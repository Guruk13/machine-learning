// dqn_tracker.rs
//
// Feature-gated TUI tracker for DQN agent performance.
// Add to Cargo.toml:
//
//   [features]
//   tracker = ["ratatui", "tui-tree-widget", "crossterm"]
//
//   [dependencies]
//   ratatui        = { version = "0.26", optional = true }
//   tui-tree-widget = { version = "0.19", optional = true }
//   crossterm      = { version = "0.27", optional = true }

use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph},
};
use tui_tree_widget::{Tree, TreeItem, TreeState};

// ─────────────────────────────────────────────────────────────────────────────
// Public data structures
// ─────────────────────────────────────────────────────────────────────────────

/// One reward sample stored in episode history.
#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    /// Episode index (global counter for this agent).
    pub episode_idx: u64,
    /// Total undiscounted return for the episode.
    pub total_reward: f32,
    /// Per-step rewards (optional, may be empty).
    pub step_rewards: Vec<f32>,
    /// Snapshot of entropy_ema at episode end.
    pub entropy_ema_snapshot: f32,
}

/// Lifecycle state of an agent in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentLifecycle {
    Alive,
    Discarded,
}

/// Registry entry – immutable identity + mutable live metrics.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    /// Stable unique ID.
    pub agent_id: u64,
    /// Human-readable label (e.g. "gen-3/child-2").
    pub label: String,
    /// Optional parent agent ID (for lineage tree).
    pub parent_id: Option<u64>,
    /// Is this agent still active?
    pub lifecycle: AgentLifecycle,
    // ── Live metrics (updated every episode) ────────────────────────────────
    pub entropy_ema: f32,
    pub score_ema: f32,
    pub episodes: u64,
    // ── History ─────────────────────────────────────────────────────────────
    /// One entry per completed episode.
    pub episode_history: Vec<EpisodeRecord>,
}

impl RegistryEntry {
    fn new(agent_id: u64, label: String, parent_id: Option<u64>) -> Self {
        Self {
            agent_id,
            label,
            parent_id,
            lifecycle: AgentLifecycle::Alive,
            entropy_ema: 0.0,
            score_ema: 0.0,
            episodes: 0,
            episode_history: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AgentRegistry  (thread-safe, cheaply cloneable handle)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AgentRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

struct RegistryInner {
    entries: HashMap<u64, RegistryEntry>,
    next_id: u64,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                entries: HashMap::new(),
                next_id: 1,
            })),
        }
    }

    /// Register a new agent; returns its stable ID.
    pub fn register(&self, label: impl Into<String>, parent_id: Option<u64>) -> u64 {
        let mut g = self.inner.lock().unwrap();
        let id = g.next_id;
        g.next_id += 1;
        g.entries
            .insert(id, RegistryEntry::new(id, label.into(), parent_id));
        id
    }

    /// Push a completed episode record for an agent.
    pub fn push_episode(&self, agent_id: u64, record: EpisodeRecord) {
        let mut g = self.inner.lock().unwrap();
        if let Some(e) = g.entries.get_mut(&agent_id) {
            e.entropy_ema = record.entropy_ema_snapshot;
            e.episodes = record.episode_idx + 1;
            e.episode_history.push(record);
        }
    }

    /// Update live score EMA without a full episode record.
    pub fn update_score_ema(&self, agent_id: u64, score_ema: f32) {
        let mut g = self.inner.lock().unwrap();
        if let Some(e) = g.entries.get_mut(&agent_id) {
            e.score_ema = score_ema;
        }
    }

    /// Mark an agent as discarded (keeps it in registry).
    pub fn discard(&self, agent_id: u64) {
        let mut g = self.inner.lock().unwrap();
        if let Some(e) = g.entries.get_mut(&agent_id) {
            e.lifecycle = AgentLifecycle::Discarded;
        }
    }

    /// Snapshot all entries (cheap clone for rendering).
    fn snapshot(&self) -> Vec<RegistryEntry> {
        self.inner
            .lock()
            .unwrap()
            .entries
            .values()
            .cloned()
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TUI application state
// ─────────────────────────────────────────────────────────────────────────────

struct App {
    registry: AgentRegistry,
    /// tui-tree-widget state (tracks open/closed nodes, selection).
    tree_state: TreeState<u64>,
    /// The agent whose reward history is shown in the secondary panel.
    selected_agent_id: Option<u64>,
    /// Last time we refreshed the snapshot from the registry.
    last_refresh: Instant,
    /// Cached snapshot (rebuilt every ~200 ms).
    snapshot: Vec<RegistryEntry>,
}

impl App {
    fn new(registry: AgentRegistry) -> Self {
        let snapshot = registry.snapshot();
        Self {
            registry,
            tree_state: TreeState::default(),
            selected_agent_id: None,
            last_refresh: Instant::now(),
            snapshot,
        }
    }

    fn refresh_if_stale(&mut self) {
        if self.last_refresh.elapsed() > Duration::from_millis(200) {
            self.snapshot = self.registry.snapshot();
            self.last_refresh = Instant::now();
        }
    }

    /// Build the tui-tree-widget items representing agent lineage.
    fn build_tree_items(&self) -> Vec<TreeItem<'static, u64>> {
        // Separate roots (no parent) from children.
        let entries = &self.snapshot;

        // Collect children per parent.
        let mut children_map: HashMap<u64, Vec<&RegistryEntry>> = HashMap::new();
        let mut roots: Vec<&RegistryEntry> = Vec::new();

        for e in entries {
            match e.parent_id {
                None => roots.push(e),
                Some(pid) => children_map.entry(pid).or_default().push(e),
            }
        }

        // Sort roots for stable ordering.
        roots.sort_by_key(|e| e.agent_id);

        fn make_item(
            entry: &RegistryEntry,
            children_map: &HashMap<u64, Vec<&RegistryEntry>>,
        ) -> TreeItem<'static, u64> {
            let status = match entry.lifecycle {
                AgentLifecycle::Alive => "●",
                AgentLifecycle::Discarded => "✕",
            };
            let color = match entry.lifecycle {
                AgentLifecycle::Alive => Color::Green,
                AgentLifecycle::Discarded => Color::DarkGray,
            };
            let label = format!(
                "{} {} [H:{:.3} S:{:.1} ep:{}]",
                status, entry.label, entry.entropy_ema, entry.score_ema, entry.episodes,
            );
            let line = Line::from(vec![Span::styled(label, Style::default().fg(color))]);

            let mut kids: Vec<&RegistryEntry> = children_map
                .get(&entry.agent_id)
                .cloned()
                .unwrap_or_default();
            kids.sort_by_key(|e| e.agent_id);

            let child_items: Vec<TreeItem<'static, u64>> =
                kids.iter().map(|c| make_item(c, children_map)).collect();

            TreeItem::new(entry.agent_id, line, child_items)
                .expect("duplicate IDs should not occur")
        }

        roots.iter().map(|r| make_item(r, &children_map)).collect()
    }

    /// Entropy EMA data for the primary panel – living agents only.
    fn alive_entropy_data(&self) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = self
            .snapshot
            .iter()
            .filter(|e| e.lifecycle == AgentLifecycle::Alive)
            .map(|e| (e.label.clone(), e.entropy_ema))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Episode reward history for the selected agent (secondary panel).
    fn selected_agent_reward_series(&self) -> Option<(String, Vec<(f64, f64)>)> {
        let id = self.selected_agent_id?;
        let entry = self.snapshot.iter().find(|e| e.agent_id == id)?;
        let points: Vec<(f64, f64)> = entry
            .episode_history
            .iter()
            .map(|r| (r.episode_idx as f64, r.total_reward as f64))
            .collect();
        Some((entry.label.clone(), points))
    }

    /// Update selected agent from tree selection.
    fn sync_selection_from_tree(&mut self) {
        self.selected_agent_id = self.tree_state.selected().first().copied();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

fn ui(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    // Top-level split: left tree (30%) | right panels (70%)
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(area);

    // Right side: primary entropy panel (50%) | secondary reward panel (50%)
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(cols[1]);

    render_lineage_tree(frame, app, cols[0]);
    render_entropy_panel(frame, app, right_rows[0]);
    render_reward_panel(frame, app, right_rows[1]);
}

fn render_lineage_tree(frame: &mut Frame, app: &mut App, area: Rect) {
    let items = app.build_tree_items();
    let tree = Tree::new(&items)
        .expect("tree items must be unique")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Agent Lineage ")
                .title_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(tree, area, &mut app.tree_state);
}

fn render_entropy_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    let data = app.alive_entropy_data();

    if data.is_empty() {
        let p = Paragraph::new("No living agents.").block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Entropy EMA – Living Agents "),
        );
        frame.render_widget(p, area);
        return;
    }

    // Build a bar-chart style with Chart (sparkline per agent on same axis).
    // Each living agent is a dataset with a single horizontal line at its entropy value.
    let palette = [
        Color::Green,
        Color::Yellow,
        Color::Cyan,
        Color::Magenta,
        Color::LightBlue,
        Color::LightRed,
    ];

    let n = data.len();
    // Represent each agent as a point at x = index.
    let series: Vec<Vec<(f64, f64)>> = data
        .iter()
        .enumerate()
        .map(|(i, (_, v))| vec![(i as f64, *v as f64)])
        .collect();

    let datasets: Vec<Dataset> = series
        .iter()
        .enumerate()
        .map(|(i, pts)| {
            Dataset::default()
                .name(data[i].0.clone())
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Bar)
                .style(Style::default().fg(palette[i % palette.len()]))
                .data(pts)
        })
        .collect();

    let max_h = data.iter().map(|(_, v)| *v).fold(0.0_f32, f32::max);
    let y_max = (max_h * 1.2).max(1.0) as f64;

    // X-axis labels: agent names (truncated).
    let x_labels: Vec<Span> = data
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            Span::styled(
                format!("{}", &name[..name.len().min(8)]),
                Style::default().fg(Color::Gray),
            )
        })
        .collect();

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Entropy EMA – Living Agents ")
                .title_style(
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
        )
        .x_axis(
            Axis::default()
                .title("agent")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, (n as f64).max(1.0)])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("H")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, y_max])
                .labels(vec![
                    Span::raw("0.0"),
                    Span::raw(format!("{:.2}", y_max / 2.0)),
                    Span::raw(format!("{:.2}", y_max)),
                ]),
        );

    frame.render_widget(chart, area);
}

fn render_reward_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    match app.selected_agent_reward_series() {
        None => {
            let p = Paragraph::new("Select an agent in the tree to view reward history.")
                .style(Style::default().fg(Color::DarkGray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Reward Over Episodes "),
                );
            frame.render_widget(p, area);
        }
        Some((label, points)) if points.is_empty() => {
            let p = Paragraph::new(format!("{} – no episodes yet.", label))
                .style(Style::default().fg(Color::DarkGray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Reward Over Episodes "),
                );
            frame.render_widget(p, area);
        }
        Some((label, points)) => {
            let x_min = points.first().map(|p| p.0).unwrap_or(0.0);
            let x_max = points.last().map(|p| p.0).unwrap_or(1.0);
            let y_min = points.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            let y_max = points.iter().map(|p| p.1).fold(f64::MIN, f64::max);
            let y_pad = ((y_max - y_min) * 0.1).max(1.0);

            let dataset = Dataset::default()
                .name(label.clone())
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&points);

            let chart = Chart::new(vec![dataset])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Reward – {} ", label))
                        .title_style(
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                )
                .x_axis(
                    Axis::default()
                        .title("episode")
                        .style(Style::default().fg(Color::DarkGray))
                        .bounds([x_min, x_max])
                        .labels(vec![
                            Span::raw(format!("{}", x_min as u64)),
                            Span::raw(format!("{}", ((x_min + x_max) / 2.0) as u64)),
                            Span::raw(format!("{}", x_max as u64)),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("R")
                        .style(Style::default().fg(Color::DarkGray))
                        .bounds([y_min - y_pad, y_max + y_pad])
                        .labels(vec![
                            Span::raw(format!("{:.1}", y_min)),
                            Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                            Span::raw(format!("{:.1}", y_max)),
                        ]),
                );

            frame.render_widget(chart, area);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event loop – run in a background thread
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the TUI in a background thread.
/// Call this from `AgentManager::new` when the `tracker` feature is active.
pub fn spawn_tracker(registry: AgentRegistry) {
    thread::spawn(move || {
        if let Err(e) = run_tui(registry) {
            eprintln!("[dqn_tracker] TUI error: {e}");
        }
    });
}

fn run_tui(registry: AgentRegistry) -> io::Result<()> {
    // Setup terminal.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(registry);

    // Pre-open tree root.
    app.tree_state.open(vec![]);

    loop {
        app.refresh_if_stale();

        terminal.draw(|f| ui(f, &mut app))?;

        // Poll with a short timeout so the UI stays live.
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,

                    // Tree navigation.
                    KeyCode::Down | KeyCode::Char('j') => {
                        app.tree_state.key_down();
                        app.sync_selection_from_tree();
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        app.tree_state.key_up();
                        app.sync_selection_from_tree();
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        app.tree_state.key_left();
                        app.sync_selection_from_tree();
                    }
                    KeyCode::Right | KeyCode::Char('l') => {
                        app.tree_state.key_right();
                        app.sync_selection_from_tree();
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        app.tree_state.toggle_selected();
                        app.sync_selection_from_tree();
                    }
                    _ => {}
                }
            }
        }
    }

    // Restore terminal.
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// AgentManager integration helpers
// ─────────────────────────────────────────────────────────────────────────────
//
// Paste the snippet below into your AgentManager impl block:
//
//  #[cfg(feature = "tracker")]
//  pub fn tracker_register(&self, label: impl Into<String>, parent_id: Option<u64>) -> u64 {
//      self.registry.register(label, parent_id)
//  }
//
//  #[cfg(feature = "tracker")]
//  pub fn tracker_push_episode(
//      &self,
//      agent_id: u64,
//      episode_idx: u64,
//      total_reward: f32,
//      step_rewards: Vec<f32>,
//      entropy_ema_snapshot: f32,
//  ) {
//      self.registry.push_episode(
//          agent_id,
//          dqn_tracker::EpisodeRecord {
//              episode_idx,
//              total_reward,
//              step_rewards,
//              entropy_ema_snapshot,
//          },
//      );
//  }
//
//  #[cfg(feature = "tracker")]
//  pub fn tracker_discard(&self, agent_id: u64) {
//      self.registry.discard(agent_id);
//  }
//
// And in AgentManager::new, after building `registry`, add:
//
//  #[cfg(feature = "tracker")]
//  dqn_tracker::spawn_tracker(registry.clone());
