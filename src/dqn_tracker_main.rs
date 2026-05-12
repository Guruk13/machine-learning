//! DQN Tracker — TUI dashboard that subscribes to live agent state.
//!
//! Architecture
//! ────────────
//!  Training threads  ──write──►  Arc<RwLock<Agent>>  ◄──read──  TUI (App)
//!
//! The TUI owns no agent data. It holds a shared reference (`Arc`) to each
//! agent and calls `.read()` on every render frame. Training threads call
//! `.write()` to push episodes or update status. No data is duplicated.
//!
//! To integrate your real agents, replace `spawn_training_thread` with your
//! own training loop and pass in the same `Arc<RwLock<Agent>>` handle.

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use rand::Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Scrollbar,
        ScrollbarOrientation,
    },
    Frame, Terminal,
};
use std::{
    collections::HashMap,
    io,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, Instant},
};
use tui_tree_widget::{Tree, TreeItem, TreeState};

// ── Agent ─────────────────────────────────────────────────────────────────────
// This is YOUR struct. The TUI never owns it — it only holds an Arc to it.

#[derive(Debug)]
pub struct Agent {
    pub id: String,
    pub generation: u32,
    pub episodes: Vec<f64>,      // reward per episode — training thread appends
    pub children: Vec<String>,   // child agent IDs (lineage)
    pub parent: Option<String>,
    pub status: AgentStatus,
    pub epsilon: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Training,
    Finished,
    Best,
}

impl Agent {
    pub fn new(id: String, generation: u32, parent: Option<String>) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            id,
            generation,
            episodes: Vec::new(),
            children: Vec::new(),
            parent,
            status: AgentStatus::Training,
            epsilon: rng.gen_range(0.05..1.0),
            learning_rate: rng.gen_range(0.0001..0.01),
        }
    }

    // Convenience helpers — called by the TUI under a read-lock
    pub fn total_reward(&self) -> f64 { self.episodes.iter().sum() }
    pub fn avg_reward(&self) -> f64 {
        if self.episodes.is_empty() { 0.0 }
        else { self.total_reward() / self.episodes.len() as f64 }
    }
    pub fn best_reward(&self) -> f64 {
        self.episodes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

// Shared handle type — what both training threads and the TUI hold
pub type SharedAgent = Arc<RwLock<Agent>>;

// ── AgentRegistry ─────────────────────────────────────────────────────────────
// Central store of Arc handles. Created once, cloned into the TUI and into
// each training thread. No agent data is ever duplicated.

#[derive(Clone)]
pub struct AgentRegistry {
    // Arc handles only — TUI reads through these, training threads write
    pub agents: HashMap<String, SharedAgent>,
    pub root_ids: Vec<String>,
    // Global best tracking (owned here since it's dashboard-only metadata)
    pub best_rewards: Arc<RwLock<Vec<(f64, f64)>>>,
    pub best_agent_id: Arc<RwLock<Option<String>>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            root_ids: Vec::new(),
            best_rewards: Arc::new(RwLock::new(Vec::new())),
            best_agent_id: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a new agent and return a clone of its shared handle.
    /// Pass the returned handle to your training thread.
    pub fn register(
        &mut self,
        id: String,
        generation: u32,
        parent: Option<String>,
    ) -> SharedAgent {
        let handle = Arc::new(RwLock::new(Agent::new(id.clone(), generation, parent.clone())));

        if generation == 0 {
            self.root_ids.push(id.clone());
        }
        if let Some(ref pid) = parent {
            if let Some(p_handle) = self.agents.get(pid) {
                p_handle.write().unwrap().children.push(id.clone());
            }
        }
        self.agents.insert(id, Arc::clone(&handle));
        handle
    }

    /// Update the global best agent. Called by the tracker thread.
    pub fn update_best(&self, candidate_id: &str, total_reward: f64, episode_idx: f64) {
        let mut best_id = self.best_agent_id.write().unwrap();

        // Demote previous best back to Training (if not Finished)
        if let Some(ref prev) = *best_id {
            if prev != candidate_id {
                if let Some(h) = self.agents.get(prev) {
                    let mut a = h.write().unwrap();
                    if a.status != AgentStatus::Finished {
                        a.status = AgentStatus::Training;
                    }
                }
            }
        }

        *best_id = Some(candidate_id.to_string());

        if let Some(h) = self.agents.get(candidate_id) {
            h.write().unwrap().status = AgentStatus::Best;
        }

        let mut br = self.best_rewards.write().unwrap();
        br.push((episode_idx, total_reward));
        if br.len() > 200 { br.remove(0); }
    }
}

// ── Training simulation ───────────────────────────────────────────────────────
// In production: replace this with your real DQN training loop.
// The only contract: call handle.write() to push episodes / update status.

fn spawn_training_thread(handle: SharedAgent, registry: AgentRegistry) {
    thread::spawn(move || {
        let mut rng = rand::thread_rng();
        let (lr, generation) = {
            let a = handle.read().unwrap();
            (a.learning_rate, a.generation)
        };
        let pace = Duration::from_millis(80 + generation as u64 * 30);
        let mut global_ep: f64 = 0.0;

        loop {
            thread::sleep(pace);

            let (ep, eps, finished) = {
                let a = handle.read().unwrap();
                (a.episodes.len() as f64, a.epsilon, a.status == AgentStatus::Finished)
            };
            if finished { break; }

            // Simulated DQN reward curve
            let trend = 200.0 * (1.0 - (-lr * 300.0 * ep).exp());
            let noise  = rng.gen_range(-30.0..30.0_f64) * (1.0 + eps);
            let reward = (trend + noise).max(-50.0);
            let new_eps = (eps * 0.998).max(0.05);

            {
                let mut a = handle.write().unwrap();
                a.episodes.push(reward);
                a.epsilon = new_eps;
                if ep > 150.0 && new_eps <= 0.06 {
                    a.status = AgentStatus::Finished;
                }
            }

            // Let the registry decide if this agent is the new global best
            let total = handle.read().unwrap().total_reward();
            let id    = handle.read().unwrap().id.clone();
            registry.update_best(&id, total, global_ep);
            global_ep += 1.0;
        }
    });
}

// ── TUI App ───────────────────────────────────────────────────────────────────
// Holds only Arc clones — zero ownership of agent data.

struct App {
    registry: AgentRegistry,
    tree_state: TreeState<String>,
    selected_id: Option<String>,
}

impl App {
    fn new(registry: AgentRegistry) -> Self {
        let first = registry.root_ids.first().cloned();
        Self { registry, tree_state: TreeState::default(), selected_id: first }
    }

    // Read agent via shared reference — no copy, no clone of data
    fn with_agent<F, R>(&self, id: &str, f: F) -> Option<R>
    where F: FnOnce(&Agent) -> R {
        self.registry.agents.get(id).map(|h| f(&h.read().unwrap()))
    }

    fn build_tree_items(&self) -> Vec<TreeItem<'static, String>> {
        fn build_item(registry: &AgentRegistry, id: &str) -> TreeItem<'static, String> {
            let handle = &registry.agents[id];
            let agent  = handle.read().unwrap();

            let status_sym = match agent.status {
                AgentStatus::Training => "[~]",
                AgentStatus::Finished => "[OK]",
                AgentStatus::Best     => "[*]",
            };
            let best_r = if agent.best_reward() == f64::NEG_INFINITY { 0.0 } else { agent.best_reward() };
            let label  = format!(
                "{} {} | ep:{:>3} | avg:{:>7.1} | best:{:>7.1}",
                status_sym, agent.id, agent.episodes.len(), agent.avg_reward(), best_r
            );
            let children_ids = agent.children.clone(); // only IDs, not data
            drop(agent); // release read-lock before recursing

            if children_ids.is_empty() {
                TreeItem::new_leaf(id.to_string(), label)
            } else {
                let children: Vec<_> = children_ids.iter()
                    .map(|cid| build_item(registry, cid))
                    .collect();
                TreeItem::new(id.to_string(), label, children).expect("unique ids")
            }
        }
        self.registry.root_ids.iter()
            .map(|id| build_item(&self.registry, id))
            .collect()
    }

    fn reward_chart_data(&self) -> Vec<(f64, f64)> {
        self.selected_id.as_deref().and_then(|id| {
            self.registry.agents.get(id).map(|h| {
                h.read().unwrap().episodes.iter()
                    .enumerate().map(|(i, &r)| (i as f64, r)).collect()
            })
        }).unwrap_or_default()
    }

    fn cumulative_reward_data(&self) -> Vec<(f64, f64)> {
        self.selected_id.as_deref().and_then(|id| {
            self.registry.agents.get(id).map(|h| {
                let mut sum = 0.0;
                h.read().unwrap().episodes.iter()
                    .enumerate()
                    .map(|(i, &r)| { sum += r; (i as f64, sum) })
                    .collect()
            })
        }).unwrap_or_default()
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &mut App) {
    let size = f.size();
    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(size);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(root[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(root[1]);

    render_tree(f, app, left[0]);
    render_best_tracker(f, app, left[1]);
    render_agent_detail(f, app, right[0]);
    render_reward_chart(f, app, right[1]);
    render_cumulative_chart(f, app, right[2]);
}

fn render_tree(f: &mut Frame, app: &mut App, area: Rect) {
    let items = app.build_tree_items();
    let tree = Tree::new(items)
        .expect("unique root ids")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" DQN Agent Lineage (read-only view) ")
                .title_bottom(" Up/Down navigate | Left/Right expand | Enter select | q quit "),
        )
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .experimental_scrollbar(Some(Scrollbar::new(ScrollbarOrientation::VerticalRight)));
    f.render_stateful_widget(tree, area, &mut app.tree_state);
}

fn render_best_tracker(f: &mut Frame, app: &App, area: Rect) {
    // Read best agent ID under lock, then release before reading agent data
    let best_id_opt = app.registry.best_agent_id.read().unwrap().clone();
    let best_id_str = best_id_opt.as_deref().unwrap_or("None");

    let (total, avg, best_ep, episodes, gen, eps_val) = best_id_opt
        .as_deref()
        .and_then(|id| app.registry.agents.get(id))
        .map(|h| {
            let a = h.read().unwrap();
            let best_r = if a.best_reward() == f64::NEG_INFINITY { 0.0 } else { a.best_reward() };
            (a.total_reward(), a.avg_reward(), best_r, a.episodes.len(), a.generation, a.epsilon)
        })
        .unwrap_or((0.0, 0.0, 0.0, 0, 0, 0.0));

    let spark: String = {
        let br = app.registry.best_rewards.read().unwrap();
        if br.len() < 2 {
            "Collecting data...".to_string()
        } else {
            let recent: Vec<f64> = br.iter().rev().take(24).rev().map(|(_, r)| *r).collect();
            let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let bars = ["\u{2581}","\u{2582}","\u{2583}","\u{2584}","\u{2585}","\u{2586}","\u{2587}","\u{2588}"];
            recent.iter().map(|&v| {
                let norm = if (max - min).abs() < 1e-6 { 0.5 } else { (v - min) / (max - min) };
                bars[((norm * 7.0) as usize).min(7)]
            }).collect()
        }
    };

    let content = vec![
        Line::from(vec![
            Span::styled(" Best Agent: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(best_id_str.to_string(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  gen:{gen}  e:{eps_val:.3}"), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled(" Total Reward: ", Style::default().fg(Color::Green)),
            Span::styled(format!("{total:.1}"), Style::default().fg(Color::White)),
            Span::styled("  Avg/ep: ", Style::default().fg(Color::Green)),
            Span::styled(format!("{avg:.1}"), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Best Ep: ", Style::default().fg(Color::Blue)),
            Span::styled(format!("{best_ep:.1}"), Style::default().fg(Color::White)),
            Span::styled("  Episodes: ", Style::default().fg(Color::Blue)),
            Span::styled(format!("{episodes}"), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Trend: ", Style::default().fg(Color::DarkGray)),
            Span::styled(spark, Style::default().fg(Color::Yellow)),
        ]),
    ];

    f.render_widget(
        Paragraph::new(content).block(
            Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" * Best Agent Tracker "),
        ),
        area,
    );
}

fn render_agent_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(ref id) = app.selected_id else {
        f.render_widget(Paragraph::new("No agent selected")
            .block(Block::default().borders(Borders::ALL).title(" Agent Detail ")), area);
        return;
    };
    let Some(handle) = app.registry.agents.get(id.as_str()) else {
        f.render_widget(Paragraph::new("Agent not found")
            .block(Block::default().borders(Borders::ALL).title(" Agent Detail ")), area);
        return;
    };

    // Single read-lock for the whole render of this panel
    let agent = handle.read().unwrap();

    let (status_str, status_color) = match agent.status {
        AgentStatus::Training => ("Training", Color::Cyan),
        AgentStatus::Finished => ("Finished", Color::Green),
        AgentStatus::Best     => ("BEST",     Color::Yellow),
    };
    let parent_str   = agent.parent.clone().unwrap_or_else(|| "(root)".to_string());
    let children_str = if agent.children.is_empty() { "none".to_string() }
                       else { agent.children.join(", ") };

    let content = vec![
        Line::from(vec![
            Span::styled(" ID: ",      Style::default().fg(Color::DarkGray)),
            Span::styled(agent.id.clone(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("  Status: ", Style::default().fg(Color::DarkGray)),
            Span::styled(status_str,   Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            Span::styled("  Gen: ",    Style::default().fg(Color::DarkGray)),
            Span::styled(agent.generation.to_string(), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled(" Parent: ",   Style::default().fg(Color::DarkGray)),
            Span::styled(parent_str,    Style::default().fg(Color::Cyan)),
            Span::styled("  Children: ",Style::default().fg(Color::DarkGray)),
            Span::styled(children_str,  Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled(" LR: ",       Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.5}", agent.learning_rate), Style::default().fg(Color::Green)),
            Span::styled("  epsilon: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.4}", agent.epsilon),        Style::default().fg(Color::Green)),
            Span::styled("  Episodes: ",Style::default().fg(Color::DarkGray)),
            Span::styled(agent.episodes.len().to_string(),       Style::default().fg(Color::White)),
            Span::styled("  Total: ",   Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}", agent.total_reward()), Style::default().fg(Color::Yellow)),
        ]),
    ];
    // read-lock released here (agent dropped)
    drop(agent);

    f.render_widget(
        Paragraph::new(content).block(
            Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(format!(" Agent: {id} "))
        ),
        area,
    );
}

fn render_reward_chart(f: &mut Frame, app: &App, area: Rect) {
    let data = app.reward_chart_data();
    if data.is_empty() {
        f.render_widget(Paragraph::new("  Waiting for data...")
            .block(Block::default().borders(Borders::ALL).title(" Episode Rewards ")), area);
        return;
    }
    let factor   = (data.len() / 300).max(1);
    let sampled: Vec<(f64, f64)> = data.iter().step_by(factor).cloned().collect();
    let min_y    = sampled.iter().map(|(_,r)| *r).fold(f64::INFINITY, f64::min) - 20.0;
    let max_y    = sampled.iter().map(|(_,r)| *r).fold(f64::NEG_INFINITY, f64::max) + 20.0;
    let max_x    = data.last().map(|(x,_)| *x).unwrap_or(1.0);

    let agent_name = app.selected_id.as_deref().unwrap_or("?");
    let dataset = Dataset::default()
        .name("reward/ep")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&sampled);

    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .title(format!(" Episode Rewards -- {agent_name} ")))
        .x_axis(Axis::default().title("episode")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([0.0, max_x])
            .labels(vec![Span::raw("0"), Span::raw(format!("{:.0}", max_x/2.0)), Span::raw(format!("{max_x:.0}"))]))
        .y_axis(Axis::default().title("reward")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([min_y, max_y])
            .labels(vec![Span::raw(format!("{min_y:.0}")), Span::raw(format!("{:.0}",(min_y+max_y)/2.0)), Span::raw(format!("{max_y:.0}"))]));

    f.render_widget(chart, area);
}

fn render_cumulative_chart(f: &mut Frame, app: &App, area: Rect) {
    let cum_data    = app.cumulative_reward_data();
    let global_data = app.registry.best_rewards.read().unwrap().clone();

    if cum_data.is_empty() && global_data.is_empty() {
        f.render_widget(Paragraph::new("  Waiting for data...")
            .block(Block::default().borders(Borders::ALL).title(" Sum Rewards ")), area);
        return;
    }

    let factor      = (cum_data.len() / 300).max(1);
    let sampled_cum: Vec<(f64, f64)> = cum_data.iter().step_by(factor).cloned().collect();
    let max_x       = cum_data.last().map(|(x,_)| *x).unwrap_or(1.0).max(1.0);

    let g_max_x = global_data.last().map(|(x,_)| *x).unwrap_or(1.0).max(1.0);
    let global_scaled: Vec<(f64, f64)> = global_data.iter()
        .map(|(x, y)| (x / g_max_x * max_x, *y)).collect();

    let all_y: Vec<f64> = sampled_cum.iter().map(|(_,y)| *y)
        .chain(global_scaled.iter().map(|(_,y)| *y)).collect();
    let min_y = all_y.iter().cloned().fold(f64::INFINITY,     f64::min) - 10.0;
    let max_y = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 10.0;

    let mut datasets = vec![
        Dataset::default().name("Cumul. Sum")
            .marker(symbols::Marker::Braille).graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan)).data(&sampled_cum),
    ];
    if !global_scaled.is_empty() {
        datasets.push(
            Dataset::default().name("Global Best")
                .marker(symbols::Marker::Dot).graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow)).data(&global_scaled),
        );
    }

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta))
            .title(" Sum of Rewards + Global Best Overlay "))
        .x_axis(Axis::default().title("episode")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([0.0, max_x])
            .labels(vec![Span::raw("0"), Span::raw(format!("{:.0}", max_x/2.0)), Span::raw(format!("{max_x:.0}"))]))
        .y_axis(Axis::default().title("reward")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([min_y, max_y])
            .labels(vec![Span::raw(format!("{min_y:.0}")), Span::raw(format!("{:.0}",(min_y+max_y)/2.0)), Span::raw(format!("{max_y:.0}"))]));

    f.render_widget(chart, area);
}

// ── Input ─────────────────────────────────────────────────────────────────────

fn handle_input(app: &mut App) -> io::Result<bool> {
    if event::poll(Duration::from_millis(0))? {
        if let Event::Key(key) = event::read()? {
            match (key.code, key.modifiers) {
                (KeyCode::Char('q'), _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(true),
                (KeyCode::Down, _) => { let items = app.build_tree_items(); app.tree_state.key_down(&items); }
                (KeyCode::Up,   _) => { let items = app.build_tree_items(); app.tree_state.key_up(&items); }
                (KeyCode::Left, _)  => { app.tree_state.key_left(); }
                (KeyCode::Right, _) => { app.tree_state.key_right(); }
                (KeyCode::Enter, _) => {
                    let sel = app.tree_state.selected();
                    if !sel.is_empty() { app.selected_id = sel.last().cloned(); }
                    app.tree_state.toggle_selected();
                }
                _ => {}
            }
        }
    }
    Ok(false)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    // 1. Build registry and register agents (your training code does this)
    let mut registry = AgentRegistry::new();

    let h_alpha        = registry.register("Alpha".to_string(),        0, None);
    let h_beta         = registry.register("Beta".to_string(),         0, None);
    let h_gamma        = registry.register("Gamma".to_string(),        0, None);
    let h_alpha_v2     = registry.register("Alpha-v2".to_string(),     1, Some("Alpha".to_string()));
    let h_alpha_v3     = registry.register("Alpha-v3".to_string(),     1, Some("Alpha".to_string()));
    let h_alpha_elite  = registry.register("Alpha-v2-elite".to_string(),2,Some("Alpha-v2".to_string()));
    let h_beta_mutant  = registry.register("Beta-mutant".to_string(),  1, Some("Beta".to_string()));

    // 2. Spawn training threads — each gets an Arc clone of its own agent
    //    and an Arc clone of the registry (for best-tracking only).
    for handle in [h_alpha, h_beta, h_gamma, h_alpha_v2, h_alpha_v3, h_alpha_elite, h_beta_mutant] {
        spawn_training_thread(handle, registry.clone());
    }

    // 3. Hand a registry clone (Arc handles only) to the TUI — no data copy
    let mut app = App::new(registry);

    // 4. Run the TUI — purely subscribes, never writes to agents
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend  = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let tick_rate    = Duration::from_millis(80);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;
        if handle_input(&mut app)? { break; }
        if last_tick.elapsed() >= tick_rate { last_tick = Instant::now(); }
        // No tick_simulation() call — training threads drive the data
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}
