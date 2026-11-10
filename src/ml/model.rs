use super::agent_utils::{Action, AgentState, AgentStats, EpisodeStep};
use super::OPTIMIZER_EPSILON;
use crate::ml::multiagent::get_global_device;
use crate::think::{get_gamma, get_lr};
use burn::optim::ModuleOptimizer;
use burn::tensor::Device;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams},
    tensor::{activation::sigmoid, Tensor},
};
//use burn::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
// ─────────────────────────────────────────────
// POLICY NETWORK
// ─────────────────────────────────────────────
#[derive(Debug, Module)]
pub struct FlappyNet {
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}

impl FlappyNet {
    pub fn new() -> Self {
        let device = get_global_device();
        Self {
            activation: Relu::new(),
            linear1: LinearConfig::new(5, 10).init(&device),
            linear2: LinearConfig::new(10, 1).init(&device),
        }
    }

    /// Output shape : [batch, 1]  — P(jump)
    pub fn forward(&self, x: Tensor<2>, _training: bool) -> Tensor<2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        sigmoid(self.linear2.forward(x).sub_scalar(3.0)) // shifts the starting p(jump) from ~0.5 to ~0.05
    }
}

// ─────────────────────────────────────────────
// AGENT — plain REINFORCE, no critic
// ─────────────────────────────────────────────
//
// WASM CONSTRAINT
// ----------------
// Browsers give us a single-threaded event loop, and `wasm_bindgen_futures`
// deliberately does not provide a blocking executor (a real `block_on` would
// freeze the tab while the GPU readback promise resolves on that very same
// thread → deadlock). So we can't turn `into_scalar_async().await` into a
// true synchronous call.
//
// What we *can* do is keep the public, Bevy-facing API synchronous by never
// awaiting on the caller's stack: `spawn_local` fires the tensor work onto
// the microtask queue, and results land in a shared `Rc<RefCell<..>>` cell
// that the next sync call drains. This makes `select_action` /
// `finish_episode` safe to call from a normal (non-async) Bevy system, at
// the cost of the result being "a call or two behind" rather than
// instantaneous. An in-flight guard prevents two overlapping GPU ops (e.g.
// two gradient steps) from racing each other.
#[wasm_bindgen]
pub struct FlappyGradientAgent {
    pub(crate) flappy: Rc<RefCell<FlappyNet>>,
    pub(crate) optimizer: ModuleOptimizer,
    pub(crate) device: Device,
    pub(crate) state: AgentState,
    pub(crate) stats: AgentStats,
    pub(crate) prime: Option<FlappyNet>,
    pub(crate) prime_score: u32,
    pub(crate) prime_backed: bool,

    // background-inference bookkeeping
    pending_action: Rc<RefCell<Option<Action>>>,
    pub(crate) action_inflight: Rc<RefCell<bool>>,
    last_action: Action,
}
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::tensor::TensorData;

/// Flattens every float param to Tensor<1>, so all readbacks share one type.
struct Collector {
    flat: Vec<Tensor<1>>,
}

impl ModuleVisitor for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.flat.push(param.val().flatten::<1>(0, D - 1));
    }
}
use crate::ml::multiagent::download_bytes;
/// Restores the saved data, in the same visit order, reusing each param's shape.
struct Loader {
    data: std::vec::IntoIter<Vec<f32>>,
}

impl ModuleMapper for Loader {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let data = self.data.next().expect("saved weights don't match model");
        param.map(|t| {
            let shape = t.shape();
            let len = data.len();
            Tensor::<1>::from_data(TensorData::new(data, [len]), &t.device()).reshape(shape)
        })
    }
}
impl FlappyGradientAgent {
    pub fn record_prime(&mut self) -> Option<FlappyNet> {
        if !self.prime_backed {
            self.prime = Some(self.flappy.borrow().clone());
            self.prime_backed = true;
        }
        self.prime.clone()
    }

    // "in case there's a challenging model , check its score , if we're in this situation : model has degraded , rollback . if model has no prime, innovate."
    pub fn replace_prime(&mut self, prime: Option<FlappyNet>, chal_prime_score: u32) {
        self.reset_stats();
        self.prime_backed = true;

        if self.prime_score <= chal_prime_score {
            if let Some(p) = prime {
                *self.flappy.borrow_mut() = p;

                self.prime_score = chal_prime_score;
                return;
            }
        }
        if let Some(p) = self.prime.clone() {
            *self.flappy.borrow_mut() = p;
            return;
        }
        *self.flappy.borrow_mut() = FlappyNet::new();
    }
    pub fn save_weights(&self, filename: String) {
        let model = self.flappy.borrow().clone();

        wasm_bindgen_futures::spawn_local(async move {
            let mut collector = Collector { flat: Vec::new() };
            model.visit(&mut collector);

            let mut all: Vec<Vec<f32>> = Vec::with_capacity(collector.flat.len());
            for t in collector.flat {
                let data = match t.into_data_async().await {
                    Ok(d) => d,
                    Err(e) => {
                        web_sys::console::error_1(&format!("readback failed: {e:?}").into());
                        return;
                    }
                };
                match data.try_into_vec_as::<f32>() {
                    Ok(v) => all.push(v),
                    Err(e) => {
                        web_sys::console::error_1(&format!("convert failed: {e:?}").into());
                        return;
                    }
                }
            }

            match wincode::serialize(&all) {
                Ok(bytes) => download_bytes(&bytes, &filename),
                Err(e) => {
                    web_sys::console::error_1(&format!("serialize failed: {e:?}").into());
                }
            }
        });
    }
    pub fn load_weights(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let all: Vec<Vec<f32>> =
            wincode::deserialize(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let model = self.flappy.borrow().clone();
        let loaded = model.map(&mut Loader {
            data: all.into_iter(),
        });
        *self.flappy.borrow_mut() = loaded;
        Ok(())
    }
}

#[wasm_bindgen]
impl FlappyGradientAgent {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let device = get_global_device();
        Self {
            flappy: Rc::new(RefCell::new(FlappyNet::new())),
            prime: None,
            device: device.clone(),
            optimizer: get_optimizer(),
            state: AgentState::new(),
            stats: AgentStats::new(None),
            pending_action: Rc::new(RefCell::new(None)),
            action_inflight: Rc::new(RefCell::new(false)),
            last_action: Action::DoNothing,
            prime_score: 0,
            prime_backed: true,
        }
    }
    // "prime functions allow to only record valuable models and create new ones"

    pub fn reset_stats(&mut self) {
        self.state = AgentState::new();
        self.stats.episodes = 0;
        //agent.stats.entropy_sum = 0.;
    }
    pub fn agent_clear_episode(&mut self) {
        self.state = AgentState::new();
    }

    pub fn reset_score(&mut self) {
        self.stats.total_score = 0.0;
    }

    pub fn agent_finish_episode(&mut self) {
        self.stats.episodes += 1;
    }

    // ── inference ──────────────────────────────────────────────────────────

    /// Sync, non-blocking. Call this once per frame from a Bevy system.
    ///
    /// Returns the most recently completed inference result (may lag a
    /// frame or two behind `state`), and opportunistically kicks off a new
    /// GPU inference in the background if one isn't already running.
    pub fn select_action(&mut self) -> Action {
        // Pick up a result that finished since the last call.
        if let Some(action) = self.pending_action.borrow_mut().take() {
            //    web_sys::console::log_1(&format!("{:?}", action).into());
            self.last_action = action;
        }

        // Kick off the next inference if nothing is already in flight.
        if !*self.action_inflight.borrow() {
            *self.action_inflight.borrow_mut() = true;

            let input = self.state_to_tensor();
            let net = self.flappy.borrow().clone();
            let pending_action = self.pending_action.clone();
            let action_inflight = self.action_inflight.clone();

            spawn_local(async move {
                let probs = net.forward(input, false); // [1, 1]

                let p_jump: f32 = probs
                    .slice([0..1, 0..1])
                    .into_scalar_async()
                    .await
                    .expect("failed to read p_jump scalar from tensor");

                let action = if rand::random::<f32>() < p_jump {
                    Action::Jump
                } else {
                    Action::DoNothing
                };

                *pending_action.borrow_mut() = Some(action);
                *action_inflight.borrow_mut() = false;
            });
        }

        self.last_action
    }

    // ── episode bookkeeping (unchanged, purely CPU-side) ─────────────────────

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
    /// Computes discounted returns, normalises them, runs the forward pass,
    /// computes the REINFORCE loss, backprops, and steps the optimizer — all
    /// in one function. Uses burn's async tensor API (`into_scalar_async`)
    /// so it's WASM-safe; the caller is responsible for awaiting it (e.g.
    /// from an async-exported wasm_bindgen method, or from within a task if
    /// you add one later).
    ///
    /// Returns the scalar loss for this episode.
    pub async fn update_policy(&mut self) -> f32 {
        let n = self.state.episode.len();
        if n == 0 {
            return 0.0;
        }

        // ── a. Discounted returns G_t ──────────────────────────────────────
        let mut returns = vec![0.0f32; n];
        let mut running = 0.0f32;
        for t in (0..n).rev() {
            running = self.state.episode[t].reward + get_gamma() * running;
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

        let states = Tensor::<1>::from_floats(flat_states.as_slice(), &self.device).reshape([n, 5]);

        // 1.0 = jumped, 0.0 = did nothing (requires Action::Jump as i64 == 1)
        let action_idx: Vec<i64> = self.state.episode.iter().map(|s| s.action as i64).collect();

        // ── d. Forward pass ───────────────────────────────────────────────
        let p = self.flappy.borrow().forward(states, true); // [n, 1]
        let p = p.clamp(1e-6, 1.0 - 1e-6); // avoid ln(0)

        let jumped: Vec<f32> = action_idx.iter().map(|&a| a as f32).collect();
        let jumped = Tensor::<1>::from_floats(jumped.as_slice(), &self.device).reshape([n, 1]);

        let log_p = p.clone().log();
        let one_minus_p = p.clone().neg().add_scalar(1.0);
        let log_1mp = one_minus_p.clone().log();

        // log π(a|s) = a·ln(p) + (1−a)·ln(1−p)
        let selected_log_probs =
            jumped.clone() * log_p.clone() + jumped.neg().add_scalar(1.0) * log_1mp.clone(); // [n, 1]

        let returns_t =
            Tensor::<1>::from_floats(normalised.as_slice(), &self.device).reshape([n, 1]);

        // Bernoulli entropy: −[p·ln p + (1−p)·ln(1−p)]
        let entropy = (p * log_p + one_minus_p * log_1mp).mean().neg();

        // ── e. Loss = −mean( log π(a|s) · G ) − β·H[π] ──────────────────────
        // NOTE: Bernoulli entropy maxes at ln2 ≈ 0.69; 0.3 is a strong
        // coefficient. Try 0.01–0.05 if the policy stays near 50/50.
        let loss = (selected_log_probs * returns_t).mean().neg() - entropy.mul_scalar(0.01f32);

        let loss_scalar: f32 = loss
            .clone()
            .into_scalar_async()
            .await
            .expect("failed to read loss scalar from tensor");

        // ── f. Backprop + optimizer step ────────────────────────────────────
        let grads = loss.backward();
        let net = self.flappy.borrow().clone();
        let grads = GradientsParams::from_grads::<_>(grads, &net);
        // take an owned FlappyNet out (clones the net, not the Rc/RefCell)

        let updated = self.optimizer.step(get_lr(), net, grads);

        // write the new FlappyNet back into the same cell
        *self.flappy.borrow_mut() = updated;

        // ── g. Clear episode buffer ──────────────────────────────────────
        self.state.episode.clear();

        loss_scalar
    }

    // ── private ────────────────────────────────────────────────────────────

    fn state_to_tensor(&self) -> Tensor<2> {
        let s = self.state.current_gamestate.unwrap();
        Tensor::<1>::from_floats(s.to_array().as_slice(), &self.device).reshape([1, 5])
    }
}
impl FlappyGradientAgent {
    // Rust-only: no attribute, so wasm_bindgen never inspects it
    pub fn set_model(&mut self, net: FlappyNet) {
        *self.flappy.borrow_mut() = net;
    }
    pub fn get_model(&self) -> Rc<RefCell<FlappyNet>> {
        self.flappy.clone()
    }
}

use burn::optim::{momentum::MomentumConfig, SgdConfig};

use burn::grad_clipping::GradientClippingConfig;

pub fn get_optimizer() -> ModuleOptimizer {
    SgdConfig::new()
        .with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
        }))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init()
}
