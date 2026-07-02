use burn::nn::DropoutConfig;
use burn::tensor::ElementConversion;
use burn::{
    module::{Module, ModuleMapper, Param, ParamId},
    nn::{Dropout, Linear, LinearConfig, Relu},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{activation::softmax, backend::AutodiffBackend, Distribution, Tensor},
};

use super::agent_utils::{Action, AgentDefault, AgentState, AgentStats, EpisodeStep};
use super::critic::Critic;
use super::pruner::normalised_entropy;
use super::OPTIMIZER_EPSILON;

// ─────────────────────────────────────────────
// ACTOR (POLICY) NETWORK
// ─────────────────────────────────────────────
#[derive(Module, Debug)]
pub struct FlappyNet<B: Backend> {
    linear1: Linear<B>, // 5 → 16
    linear2: Linear<B>, // 16 → 2
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> FlappyNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            activation: Relu::new(),
            dropout: DropoutConfig::new(0.2).init(),
            linear1: LinearConfig::new(8, 16).init(device),
            linear2: LinearConfig::new(16, 2).init(device),
        }
    }

    /// Input  shape : [batch, 8]
    /// Output shape : [batch, 2]  — softmax probabilities
    pub fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        softmax(self.linear2.forward(x), 1)
    }
}

// ─────────────────────────────────────────────
// AGENT
// ─────────────────────────────────────────────
/// Each agent owns only its *actor* (FlappyNet + Adam).
/// The critic lives outside, in the population / game-loop struct,
/// and is passed in by `&mut` reference only when needed.
///
///   Population owns:  Vec<FlappyGradientAgent>  +  Critic
///   Per-agent:        FlappyNet, its optimizer, episode buffer, stats
///
/// This means:
///   • N agents share one V(s) — consistent baseline for all.
///   • The critic is updated once per round with all agents' data pooled.
///   • No Arc/Mutex needed — everything is single-threaded by design.
pub struct FlappyGradientAgent<B: AutodiffBackend> {
    pub flappy: FlappyNet<B>,
    pub optimizer: OptimizerAdaptor<Adam, FlappyNet<B>, B>,
    pub device: B::Device,
    pub state: AgentState,
    pub stats: AgentStats,
}

impl<B: AutodiffBackend> FlappyGradientAgent<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            flappy: FlappyNet::new(&device),
            optimizer: get_optimizer(),
            device,
            state: AgentState::new(),
            stats: AgentStats::new(None),
        }
    }

    // ── inference ──────────────────────────────────────────────────────────

    pub fn select_action(&mut self) -> Action {
        let input = self.state_to_tensor();
        let probs = self.flappy.forward(input, false);
        let data = probs.clone().into_data();
        let raw: Vec<f32> = data.iter::<f32>().collect();

        let entropy_sum: f32 = raw
            .chunks(2)
            .map(|row| {
                let e = normalised_entropy(row);
                if !e.is_finite() {
                    println!("bad entropy row: {:?}", row);
                }
                e
            })
            .sum();
        self.stats.entropy_sum += entropy_sum;

        let p_jump: f32 = probs.slice([0..1, 1..2]).into_scalar().elem::<f32>();
        if rand::random::<f32>() < p_jump {
            Action::Jump
        } else {
            Action::DoNothing
        }
    }

    // ── episode bookkeeping ────────────────────────────────────────────────

    pub fn record_step(&mut self, action: Action, reward: f32) {
        let state = self.state.get_state_features();
        self.state.episode.push(EpisodeStep {
            state,
            action,
            reward,
        });
    }

    // ── A2C episode update ─────────────────────────────────────────────────

    /// Call when the bird dies.
    ///
    /// Responsibilities
    /// ─────────────────
    /// 1. Compute discounted returns G_t from the episode buffer.
    /// 2. Query the *shared* critic for V(s_t) → advantages.
    /// 3. Update the actor with ∇ log π · A − β·H.
    /// 4. Return (flat_states, flat_returns, n) so the caller can pool
    ///    this agent's data with others for a single critic update.
    ///
    /// The critic is NOT updated here — the caller does that once after
    /// collecting data from ALL agents (see `Critic::update_batch`).
    ///
    /// Returns
    /// ───────
    /// `(actor_loss, flat_states, flat_returns, n)`
    ///   • actor_loss   — scalar for logging
    ///   • flat_states  — row-major f32, len n*8  (to be pooled)
    ///   • flat_returns — f32, len n              (to be pooled)
    ///   • n            — number of steps
    pub fn finish_episode(&mut self, critic: &Critic<B>) -> (f32, Vec<f32>, Vec<f32>, usize) {
        let n = self.state.episode.len();
        if n == 0 {
            return (0.0, vec![], vec![], 0);
        }

        // ── a. Discounted returns G_t ──────────────────────────────────────
        let mut returns = vec![0.0f32; n];
        let mut running = 0.0f32;
        for t in (0..n).rev() {
            running = self.state.episode[t].reward + AgentDefault::default().gamma * running;
            returns[t] = running;
        }

        // ── b. Flat state batch for this episode ───────────────────────────
        let flat_states: Vec<f32> = self
            .state
            .episode
            .iter()
            .flat_map(|s| s.state.to_array())
            .collect();

        // ── c. Query shared critic → V(s_t) ───────────────────────────────
        //   Returns plain f32 — no grad graph, safe to use immediately.
        let state_values = critic.values_of(&flat_states, n);

        // ── d. Advantages A_t = G_t − V(s_t), normalised ──────────────────
        let advantages_raw: Vec<f32> = returns
            .iter()
            .zip(state_values.iter())
            .map(|(g, v)| g - v)
            .collect();

        let adv_mean = advantages_raw.iter().sum::<f32>() / n as f32;
        let adv_std = (advantages_raw
            .iter()
            .map(|a| (a - adv_mean).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt()
            + 1e-8;

        let advantages: Vec<f32> = advantages_raw
            .iter()
            .map(|a| (a - adv_mean) / adv_std)
            .collect();

        // ── e. Build tensors for actor update ─────────────────────────────
        let states =
            Tensor::<B, 1>::from_floats(flat_states.as_slice(), &self.device).reshape([n, 8]);

        // ── f. Actor update ────────────────────────────────────────────────
        let actor_loss = self.update_actor(&advantages, states, n);

        // ── g. Clear episode buffer ────────────────────────────────────────

        // Return raw episode data so the caller can pool it for the critic.
        (actor_loss, flat_states, returns, n)
    }

    // ── private ────────────────────────────────────────────────────────────

    fn update_actor(&mut self, advantages: &[f32], states: Tensor<B, 2>, n: usize) -> f32 {
        let action_idx: Vec<i64> = self.state.episode.iter().map(|s| s.action as i64).collect();

        let log_probs = burn::tensor::activation::log_softmax(self.flappy.forward(states, true), 1);
        let probs = log_probs.clone().exp();

        // One-hot mask → gather log prob of the action taken
        let action_mask: Vec<f32> = (0..n)
            .flat_map(|i| {
                let mut row = [0.0f32; 2];
                row[action_idx[i] as usize] = 1.0;
                row
            })
            .collect();
        let mask =
            Tensor::<B, 1>::from_floats(action_mask.as_slice(), &self.device).reshape([n, 2]);
        let selected_log_probs = (log_probs.clone() * mask).sum_dim(1); // [n, 1]

        let adv_t = Tensor::<B, 1>::from_floats(advantages, &self.device).reshape([n, 1]);
        let entropy = (probs * log_probs).sum_dim(1).mean().neg();

        let loss = (selected_log_probs * adv_t).mean().neg() - entropy.mul_scalar(0.01f32);

        let loss_scalar: f32 = loss.clone().into_scalar().elem::<f32>();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.flappy);
        self.flappy = self.optimizer.step(
            AgentDefault::default().learning_rate,
            self.flappy.clone(),
            grads,
        );
        loss_scalar
    }

    fn state_to_tensor(&self) -> Tensor<B, 2> {
        let s = self.state.current_gamestate.unwrap();
        Tensor::<B, 1>::from_floats(s.to_array().as_slice(), &self.device).reshape([1, 8])
    }

    pub fn perturb_weights(&mut self, scale: f32) -> FlappyNet<B> {
        let mut mapper = PerturbMapper {
            scale,
            device: self.device.clone(),
        };
        self.flappy.clone().map(&mut mapper)
    }
}

pub fn get_optimizer<B: AutodiffBackend>() -> OptimizerAdaptor<Adam, FlappyNet<B>, B> {
    AdamConfig::new().with_epsilon(OPTIMIZER_EPSILON).init()
}

struct PerturbMapper<B: Backend> {
    scale: f32,
    device: B::Device,
}

impl<B: Backend> ModuleMapper<B> for PerturbMapper<B> {
    fn map_float<const D: usize>(&mut self, tensor: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let shape = tensor.val().shape();
        let noise = Tensor::<B, D>::random(
            shape,
            Distribution::Normal(0.0, self.scale as f64),
            &self.device,
        );
        tensor.map(|t| t + noise)
    }
}
