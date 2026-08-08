use burn::cubecl::device;
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
use super::pruner::normalised_entropy;
use super::OPTIMIZER_EPSILON;
use crate::MyAutodiffBackend;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Device;
use wasm_bindgen::prelude::*;
// ─────────────────────────────────────────────
// POLICY NETWORK
// ─────────────────────────────────────────────
#[derive(Module, Debug, Clone)]
#[wasm_bindgen]
pub struct FlappyNet {
    linear1: Linear<MyAutodiffBackend>, // 8 → 16
    linear2: Linear<MyAutodiffBackend>, // 16 → 2
    activation: Relu,
    dropout: Dropout,
}

impl FlappyNet {
    pub fn new(device: &Device<MyAutodiffBackend>) -> Self {
        Self {
            activation: Relu::new(),
            dropout: DropoutConfig::new(0.2).init(),
            linear1: LinearConfig::new(8, 16).init(device),
            linear2: LinearConfig::new(16, 2).init(device),
        }
    }

    /// Input  shape : [batch, 8]
    /// Output shape : [batch, 2]  — softmax probabilities
    pub fn forward(
        &self,
        x: Tensor<MyAutodiffBackend, 2>,
        _training: bool,
    ) -> Tensor<MyAutodiffBackend, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        softmax(self.linear2.forward(x), 1)
    }
}

// ─────────────────────────────────────────────
// AGENT — plain REINFORCE, no critic
// ─────────────────────────────────────────────
/// Dead simple policy-gradient agent.
///
/// Each agent owns just its policy net + optimizer + episode buffer.
/// There is no value network, no baseline queries, no pooling between
/// agents — every agent learns entirely from its own episode returns.
///
/// Update rule (REINFORCE with a "free" baseline):
///   1. Play an episode, recording (state, action, reward).
///   2. Compute discounted returns G_t.
///   3. Normalise G_t across the episode (mean 0, std 1) — this acts
///      as a cheap, no-cost baseline, replacing the critic.
///   4. Loss = −mean( log π(a_t|s_t) · G_t ) − β·H[π]
///   5. Backprop, step Adam. Done.
#[wasm_bindgen]
pub struct FlappyGradientAgent {
    pub(crate) flappy: FlappyNet,
    pub(crate) optimizer: OptimizerAdaptor<Adam, FlappyNet, MyAutodiffBackend>,
    pub(crate) device: Device<MyAutodiffBackend>,
    pub(crate) state: AgentState,
    pub(crate) stats: AgentStats,
}

#[wasm_bindgen]
impl FlappyGradientAgent {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let device = WgpuDevice::default();
        Self {
            flappy: FlappyNet::new(&device.clone),
            device: device,
            optimizer: get_optimizer(),
            state: AgentState::new(),
            stats: AgentStats::new(None),
        }
    }

    // ── inference ──────────────────────────────────────────────────────────

    pub fn select_action(&mut self) -> Action {
        let input = self.state_to_tensor();
        let probs = self.flappy.forward(input, false);
        let data = probs.clone().to_data();
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
        self.stats.total_score += reward;
    }

    // ── REINFORCE episode update ───────────────────────────────────────────

    /// Call when the bird dies.
    ///
    /// No critic involved: discounted returns are normalised in-place
    /// and used directly as the policy-gradient signal.
    ///
    /// Returns the scalar loss (for logging).
    pub fn finish_episode(&mut self) -> f32 {
        let n = self.state.episode.len();
        if n == 0 {
            return 0.0;
        }

        // ── a. Discounted returns G_t ──────────────────────────────────────
        let mut returns = vec![0.0f32; n];
        let mut running = 0.0f32;
        for t in (0..n).rev() {
            running = self.state.episode[t].reward + AgentDefault::default().gamma * running;
            returns[t] = running;
        }

        // ── b. Normalise returns (mean 0, std 1) — acts as a free baseline ──
        let mean = returns.iter().sum::<f32>() / n as f32;
        let std =
            (returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-8;
        let normalised: Vec<f32> = returns.iter().map(|r| (r - mean) / std).collect();

        // ── c. Flat state batch for this episode ────────────────────────────
        let flat_states: Vec<f32> = self
            .state
            .episode
            .iter()
            .flat_map(|s| s.state.to_array())
            .collect();

        let states =
            Tensor::<MyAutodiffBackend, 1>::from_floats(flat_states.as_slice(), &self.device)
                .reshape([n, 8]);

        // ── d. Policy update ──────────────────────────────────────────────
        let loss = self.update_policy(&normalised, states, n);

        // ── e. Clear episode buffer ──────────────────────────────────────
        self.state.episode.clear();

        loss
    }

    // ── private ────────────────────────────────────────────────────────────

    fn update_policy(
        &mut self,
        returns: &[f32],
        states: Tensor<MyAutodiffBackend, 2>,
        n: usize,
    ) -> f32 {
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
            Tensor::<MyAutodiffBackend, 1>::from_floats(action_mask.as_slice(), &self.device)
                .reshape([n, 2]);
        let selected_log_probs = (log_probs.clone() * mask).sum_dim(1); // [n, 1]

        let returns_t =
            Tensor::<MyAutodiffBackend, 1>::from_floats(returns, &self.device).reshape([n, 1]);
        let entropy = (probs * log_probs).sum_dim(1).mean().neg();

        // Loss = −mean( log π(a|s) · G ) − β·H[π]
        let loss = (selected_log_probs * returns_t).mean().neg() - entropy.mul_scalar(0.01f32);

        let loss_scalar: f32 = loss.clone().into_scalar().elem::<f32>();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<MyAutodiffBackend, _>(grads, &self.flappy);
        self.flappy = self.optimizer.step(
            AgentDefault::default().learning_rate,
            self.flappy.clone(),
            grads,
        );
        loss_scalar
    }

    fn state_to_tensor(&self) -> Tensor<MyAutodiffBackend, 2> {
        let s = self.state.current_gamestate.unwrap();
        Tensor::<MyAutodiffBackend, 1>::from_floats(s.to_array().as_slice(), &self.device)
            .reshape([1, 8])
    }

    pub fn perturb_weights(&mut self, scale: f32) -> FlappyNet {
        let mut mapper = PerturbMapper {
            scale,
            device: self.device.clone(),
        };
        self.flappy.clone().map(&mut mapper)
    }
}

pub fn get_optimizer() -> OptimizerAdaptor<Adam, FlappyNet, MyAutodiffBackend> {
    AdamConfig::new().with_epsilon(OPTIMIZER_EPSILON).init()
}

struct PerturbMapper {
    scale: f32,
    device: Device<MyAutodiffBackend>,
}

impl ModuleMapper<MyAutodiffBackend> for PerturbMapper {
    fn map_float<const D: usize>(
        &mut self,
        tensor: Param<Tensor<MyAutodiffBackend, D>>,
    ) -> Param<Tensor<MyAutodiffBackend, D>> {
        let shape = tensor.val().shape();
        let noise = Tensor::<MyAutodiffBackend, D>::random(
            shape,
            Distribution::Normal(0.0, self.scale as f64),
            &self.device,
        );
        tensor.map(|t| t + noise)
    }
}
