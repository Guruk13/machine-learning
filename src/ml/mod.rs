use bevy::{ecs::error::warn, prelude::Component};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, activation::softmax, backend::AutodiffBackend},
};
pub mod multiagent;
use bevy::prelude::warn;

use burn::tensor::ElementConversion;

use burn::module::Param;

use std::time::{Duration, Instant};

struct Debounce {
    last_called: Option<Instant>,
    delay: Duration,
}

impl Debounce {
    fn new(ms: u64) -> Self {
        Self {
            last_called: None,
            delay: Duration::from_millis(ms),
        }
    }

    fn should_run(&mut self) -> bool {
        let now = Instant::now();
        match self.last_called {
            Some(last) if now.duration_since(last) < self.delay => false,
            _ => {
                self.last_called = Some(now);
                true
            }
        }
    }
}

// ─────────────────────────────────────────────
// 1.  POLICY NETWORK
// ─────────────────────────────────────────────
/// Input  : 6 features
/// Output : 2 action probabilities (softmax over [do-nothing, jump])
#[derive(Module, Debug)]
pub struct FlappyNet<B: Backend> {
    linear1: Linear<B>, // 6 → 16
    linear2: Linear<B>, // 16 → 16
    linear3: Linear<B>, // 16 → 2
    activation: Relu,
}

impl<B: Backend> FlappyNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            activation: Relu::new(),
            linear1: LinearConfig::new(6, 16).init(device),
            linear2: LinearConfig::new(16, 16).init(device),
            linear3: LinearConfig::new(16, 2).init(device),
        }
    }

    /// Forward pass.
    /// Input shape  : [batch, 6]
    /// Output shape : [batch, 2]  — softmax probabilities
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let logits = self.linear3.forward(x);
        softmax(logits, 1) // → action probabilities
    }
}

// ─────────────────────────────────────────────
// 2.  EPISODE STEP  (one transition for PG)
// ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct EpisodeStep {
    pub state: GameStateFeatures,
    pub action: Action,
    pub reward: f32,
}

// ─────────────────────────────────────────────
// 3.  AGENT
// ─────────────────────────────────────────────
pub struct FlappyGradientAgent<B: AutodiffBackend> {
    pub flappy: FlappyNet<B>,
    optimizer: OptimizerAdaptor<Adam, FlappyNet<B>, B>,
    device: B::Device,
    /// Steps collected in the *current* episode for this agent.
    episode: Vec<EpisodeStep>,
    /// Discount factor γ
    pub gamma: f32,
    grad_debounce: Debounce,
}

impl<B: AutodiffBackend> FlappyGradientAgent<B> {
    pub fn new(device: B::Device, gamma: f32, lr: f64) -> Self {
        let flappy = FlappyNet::new(&device);

        let optimizer = AdamConfig::new().with_epsilon(1e-7).init();

        Self {
            flappy,
            optimizer,
            device,
            episode: Vec::new(),
            gamma,
            grad_debounce: Debounce::new(2000),
        }
    }

    // ── inference ──────────────────────────────────────────────────────────

    /// Sample an action from the flappy distribution.
    /// Returns (action, log_prob) — log_prob not needed at call-site but
    /// useful for debugging / entropy logging.
    pub fn select_action(&self, state: &GameStateFeatures) -> Action {
        let input = self.state_to_tensor(&state); // [1, 6]
        // Run in no-grad context — we only need probabilities here.
        let probs = self.flappy.forward(input); // [1, 2]

        // Extract the probability of Jump (index 1).
        let p_jump: f32 = probs
            .clone()
            .slice([0..1, 1..2])
            .into_scalar()
            .elem::<f32>();

        // Stochastic sampling.
        if rand::random::<f32>() < p_jump {
            Action::Jump
        } else {
            Action::DoNothing
        }
    }

    // ── episode bookkeeping ────────────────────────────────────────────────

    /// Call once per game tick after the environment returns a reward.
    pub fn record_step(&mut self, state: GameStateFeatures, action: Action, reward: f32) {
        self.episode.push(EpisodeStep {
            state,
            action,
            reward,
        });
    }

    /// Call when the bird dies (episode ends).
    /// Runs a full REINFORCE update and clears the episode buffer.
    /// Returns the mean flappy loss for logging.
    pub fn finish_episode(&mut self) -> f32 {
        if self.episode.is_empty() {
            return 0.0;
        }

        // ── 3a. Compute discounted returns G_t ────────────────────────────
        let n = self.episode.len();
        let mut returns = vec![0.0f32; n];
        let mut running = 0.0f32;
        for t in (0..n).rev() {
            running = self.episode[t].reward + self.gamma * running;
            returns[t] = running;
        }

        // ── 3b. Normalise returns (stabilises training) ───────────────────
        let mean: f32 = returns.iter().sum::<f32>() / n as f32;
        let var: f32 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32;

        let std = var.sqrt() + 1e-8;
        let returns: Vec<f32> = returns.iter().map(|r| (r - mean) / std).collect();

        // ── 3c. Build state batch tensor  [n, 6] ─────────────────────────
        let flat: Vec<f32> = self
            .episode
            .iter()
            .flat_map(|s| s.state.to_array())
            .collect();
        let states = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device).reshape([n, 6]);

        // ── 3d. Build action indices [n] ─────────────────────────────────
        let action_idx: Vec<i64> = self.episode.iter().map(|s| s.action as i64).collect();

        // ── 3e. Forward pass → log-probs of taken actions ─────────────────
        let probs = self.flappy.forward(states); // [n, 2]

        // log(probs) — add small ε for numerical stability
        let log_probs = (probs + 1e-8f32).log();

        // Gather log_prob for the action that was actually taken.
        // burn doesn't have a gather shorthand on 2-D so we build a mask.
        let action_mask: Vec<f32> = (0..n)
            .flat_map(|i| {
                let a = action_idx[i] as usize;
                // one-hot row: [1, 0] or [0, 1]
                let mut row = [0.0f32; 2];
                row[a] = 1.0;
                row
            })
            .collect();

        let mask =
            Tensor::<B, 1>::from_floats(action_mask.as_slice(), &self.device).reshape([n, 2]);

        // Sum over action dim → selected log-probs  [n]
        let selected_log_probs = (log_probs * mask).sum_dim(1); // [n, 1]

        // ── 3f. Returns tensor [n, 1] ─────────────────────────────────────
        let returns_t =
            Tensor::<B, 1>::from_floats(returns.as_slice(), &self.device).reshape([n, 1]);

        // ── 3g. REINFORCE loss  = -mean( log π(a|s) * G_t ) ──────────────
        let loss = (selected_log_probs * returns_t).mean().neg();

        // ── 3h. Back-prop + optimiser step ───────────────────────────────
        let loss_scalar: f32 = loss.clone().into_scalar().elem::<f32>();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.flappy);
        if self.grad_debounce.should_run() {
            debug_grads(&self.flappy, &grads);
        }

        self.flappy = self.optimizer.step(1e-3, self.flappy.clone(), grads);

        // ── 3i. Reset episode buffer ──────────────────────────────────────
        self.episode.clear();

        loss_scalar
    }

    // ── helpers ────────────────────────────────────────────────────────────

    fn state_to_tensor(&self, s: &GameStateFeatures) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(s.to_array().as_slice(), &self.device).reshape([1, 6])
    }
}

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GameStateFeatures {
    pub bird_y: f32,
    pub bird_speed: f32,
    pub next_pipe_top_y: f32,
    pub next_pipe_bottom_y: f32,
    pub dist_top: f32,
    pub dist_bot: f32,
}
impl GameStateFeatures {
    /// Returns a flat [6] array — handy when building batch tensors manually.
    pub fn to_array(&self) -> [f32; 6] {
        [
            self.bird_y,
            self.bird_speed,
            self.next_pipe_top_y,
            self.next_pipe_bottom_y,
            self.dist_top,
            self.dist_bot,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    DoNothing = 0,
    Jump = 1,
}

pub struct RewardPrizes {
    pub dying: f32,
    pub pipe_cleared: f32,
    pub alive: f32,
}
impl Default for RewardPrizes {
    fn default() -> Self {
        Self {
            dying: -1.0,
            pipe_cleared: 1.0,
            alive: 0.01,
        }
    }
}

fn debug_grads<B: AutodiffBackend>(model: &FlappyNet<B>, grads: &GradientsParams) {
    /*     let layers: [(&str, &Linear<B>); 3] = [
        ("linear1", &model.linear1),
        ("linear2", &model.linear2),
        ("linear3", &model.linear3),
    ]; */
    let layers: Vec<(&str, &Param<Tensor<B, 2>>)> = vec![
        ("linear1.weight", &model.linear1.weight),
        ("linear2.weight", &model.linear2.weight),
        ("linear3.weight", &model.linear3.weight),
    ];

    for (name, layer) in layers {
        // Extract param ID from the weight tensor first
        let weight_id = layer.id;
        match grads.get(weight_id) {
            Some(grad) => {
                let data = grad.into_data();
                let values: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

                let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                warn!(
                    "Grad [{}] min={:.6} max={:.6} mean={:.6}",
                    name, min, max, mean
                );
            }
            None => warn!("Grad [{}.weight] => NONE (no gradient flowed!)", name),
        }
        // Biases — 1D
        let biases: Vec<(&str, &Option<Param<Tensor<B, 1>>>)> = vec![
            ("linear1.bias", &model.linear1.bias),
            ("linear2.bias", &model.linear2.bias),
            ("linear3.bias", &model.linear3.bias),
        ];

        for (name, maybe_bias) in biases {
            if let Some(bias) = maybe_bias {
                match grads.get(bias.id) {
                    Some(grad) => warn!("Grad [{}] bias=[{}] }", name, grad.into_data()),
                    None => warn!("Grad [{}] => NONE", name),
                }
            }
        }
    }
}
