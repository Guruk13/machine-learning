use bevy::prelude::Component;
use burn::{
    module::{Module, ModuleMapper, ParamId},
    nn::{Linear, LinearConfig, Relu},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Distribution, Tensor, activation::softmax, backend::AutodiffBackend},
};
pub mod multiagent;
use bevy::prelude::warn;
pub mod pruner;
use crate::pruner::normalised_entropy;

use burn::tensor::ElementConversion;
pub const OPTIMIZER_EPSILON: f32 = 1e-7;

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
    pub optimizer: OptimizerAdaptor<Adam, FlappyNet<B>, B>,
    device: B::Device,
    /// Steps collected in the *current* episode for this agent.
    pub episode: Vec<EpisodeStep>,
    /// Discount factor γ
    pub gamma: f32,
    entropy_sum: f32,
    pub lr: f64,
}

impl<B: AutodiffBackend> FlappyGradientAgent<B> {
    pub fn new(device: B::Device, gamma: f32, lr: f64) -> Self {
        let flappy = FlappyNet::new(&device);

        let optimizer = get_optimizer();
        Self {
            flappy,
            optimizer,
            device,
            episode: Vec::new(),
            gamma, // how much it cares for future rewards
            lr,    //aka learning rate
            entropy_sum: 0.0,
        }
    }

    // ── inference ──────────────────────────────────────────────────────────

    /// Sample an action from the flappy distribution.
    /// Returns (action, log_prob) — log_prob not needed at call-site but
    /// useful for debugging / entropy logging.
    pub fn select_action(&mut self, state: &GameStateFeatures) -> Action {
        let input = self.state_to_tensor(&state); // [1, 6]
        // Run in no-grad context — we only need probabilities here.
        let probs = self.flappy.forward(input); // [1, 2]
        let data = probs.clone().to_data();
        let raw: Vec<f32> = data.iter::<f32>().collect();

        // Average normalised entropy over the batch.
        let entropy_sum: f32 = raw
            .chunks(2) // each chunk is one row: [p_nothing, p_jump]
            .map(|row| normalised_entropy(row))
            .sum();

        self.entropy_sum += entropy_sum;

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
        /*         if self.grad_debounce.should_run() {
            debug_grads(&self.flappy, &grads);
        } */

        self.flappy = self.optimizer.step(self.lr, self.flappy.clone(), grads);

        // ── 3i. Reset episode buffer ──────────────────────────────────────
        self.episode.clear();
        loss_scalar
    }

    // ── helpers ────────────────────────────────────────────────────────────

    fn state_to_tensor(&self, s: &GameStateFeatures) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(s.to_array().as_slice(), &self.device).reshape([1, 6])
    }
    // ─────────────────────────────────────────────────────────────────────────────
    // 6.  WEIGHT PERTURBATION  (for replacement)
    // ─────────────────────────────────────────────────────────────────────────────

    /// Add Gaussian noise with std `scale` to every parameter of `net`.
    /// This gives a slightly randomised child that still starts near the parent.
    // PerturbMapper is here to edit weights (2D) and biases(1D) , hence the generic D

    pub fn perturb_weights(&mut self, scale: f32) -> FlappyNet<B> {
        let mut mapper = PerturbMapper {
            scale,
            device: self.device.clone(),
        };
        self.flappy.clone().map(&mut mapper)
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

/** In ML, gamma (γ) is the discount factor used in reinforcement learning (RL).
 * In ML, gamma (γ) is the discount factor used in reinforcement learning (RL).
 * It's a value between 0 and 1 that determines how much an agent values future rewards relative to immediate rewards.
The Bellman Equation Context
Gamma appears in the return (cumulative reward) calculation:
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
What the Value Means
GammaBehaviorγ = 0Fully myopic — only cares about the immediate next rewardγ →
1Fully far-sighted — values future rewards almost as much as current onesγ = 0.99Common default — slight preference for sooner rewards */

pub struct AgentDefault {
    pub gamma: f32,
    pub learning_rate: f64,
}

impl Default for AgentDefault {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            learning_rate: 1e-3,
        }
    }
}

pub fn get_optimizer<B: AutodiffBackend>() -> OptimizerAdaptor<Adam, FlappyNet<B>, B> {
    AdamConfig::new().with_epsilon(OPTIMIZER_EPSILON).init()
}

struct PerturbMapper<B: Backend> {
    scale: f32,
    device: B::Device,
}

// useful for per weight , per Dimmension edition (_id)
impl<B: Backend> ModuleMapper<B> for PerturbMapper<B> {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let shape = tensor.shape();
        let noise = Tensor::<B, D>::random(
            shape,
            Distribution::Normal(0.0, self.scale as f64),
            &self.device,
        );
        tensor + noise
    }
}
