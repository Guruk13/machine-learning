use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};

use super::OPTIMIZER_EPSILON;

// ─────────────────────────────────────────────
// CRITIC (VALUE) NETWORK
// ─────────────────────────────────────────────
/// Estimates V(s) — shared across all agents.
///
/// Because every agent observes the same 8-feature game state,
/// one value network can serve N parallel actors.  Sharing gives:
///   • A larger effective batch every update (all agents' episodes pooled).
///   • A single, consistent baseline — no per-agent drift.
///   • Far fewer parameters to maintain (1× instead of N×).
///
/// Ownership model
/// ───────────────
///   Critic lives in the population / game-loop struct, NOT inside any agent.
///   Agents call `critic.value_of()` during `finish_episode` and pass the
///   resulting advantages back to `critic.update_batch()` once per round.
#[derive(Module, Debug)]
pub struct ValueNet<B: Backend> {
    linear1: Linear<B>, // 8 → 64
    linear2: Linear<B>, // 64 → 64
    linear3: Linear<B>, // 64 → 1   (unbounded — no output activation)
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> ValueNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            activation: Relu::new(),
            dropout: DropoutConfig::new(0.2).init(),
            linear1: LinearConfig::new(8, 64).init(device),
            linear2: LinearConfig::new(64, 64).init(device),
            linear3: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Input  shape : [batch, 8]
    /// Output shape : [batch, 1]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let training = true;
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = if training { self.dropout.forward(x) } else { x };
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = if training { self.dropout.forward(x) } else { x };
        self.linear3.forward(x)
    }
}

// ─────────────────────────────────────────────
// SHARED CRITIC
// ─────────────────────────────────────────────
/// One `Critic` instance, shared by every `FlappyGradientAgent`.
///
/// Typical usage in a population loop
/// ────────────────────────────────────
/// ```rust
/// // Initialise once, outside the agent vec.
/// let mut critic = Critic::new(device.clone());
///
/// loop {  // training round
///     // Each agent plays its episode and collects (states, returns).
///     let mut batched_states:  Vec<f32> = vec![];
///     let mut batched_returns: Vec<f32> = vec![];
///     let mut total_steps = 0usize;
///
///     for agent in &mut agents {
///         let (states, returns, n) = agent.collect_episode(&mut critic);
///         batched_states.extend(states);
///         batched_returns.extend(returns);
///         total_steps += n;
///     }
///
///     // One critic update using the combined experience of all agents.
///     let critic_loss = critic.update_batch(&batched_states, &batched_returns, total_steps, lr);
/// }
/// ```
pub struct Critic<B: AutodiffBackend> {
    pub net: ValueNet<B>,
    pub optimizer: OptimizerAdaptor<Adam, ValueNet<B>, B>,
    pub device: B::Device,
}

impl<B: AutodiffBackend> Critic<B> {
    pub fn new(device: B::Device) -> Self {
        let net = ValueNet::new(&device);
        let optimizer = AdamConfig::new().with_epsilon(OPTIMIZER_EPSILON).init();
        Self {
            net,
            optimizer,
            device,
        }
    }

    // ── inference ──────────────────────────────────────────────────────────

    /// Estimate V(s) for a batch of states given as a flat f32 slice.
    ///
    /// `flat_states` must have length `n * 8` (row-major).
    /// Returns one f32 per state.
    ///
    /// Called by every agent individually during `finish_episode`; no grad
    /// graph is built on this path (the values are detached to plain f32
    /// before being handed back to the agents).
    pub async fn values_of(&self, flat_states: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(flat_states.len(), n * 8);
        let t = Tensor::<B, 1>::from_floats(flat_states, &self.device).reshape([n, 8]);
        let v = self.net.forward(t); // [n, 1]
        v.into_data().iter::<f32>().collect()
    }

    // ── batched training ───────────────────────────────────────────────────

    /// Perform one MSE update using the *combined* experience from all agents.
    ///
    /// Arguments
    /// ---------
    /// * `flat_states`  — row-major f32 slice, length `total_steps * 8`
    /// * `flat_returns` — f32 slice, length `total_steps`  (discounted G_t)
    /// * `total_steps`  — number of (state, return) pairs (sum over all agents)
    /// * `learning_rate`
    ///
    /// Returns the scalar MSE loss for logging.
    ///
    /// Design note: pooling all agents' data before the backward pass means
    /// one Adam step sees a much larger and more diverse batch than any single
    /// agent could provide, which stabilises V(s) early in training.
    pub async fn update_batch(
        &mut self,
        flat_states: &[f32],
        flat_returns: &[f32],
        total_steps: usize,
        learning_rate: f64,
    ) -> f32 {
        debug_assert_eq!(flat_states.len(), total_steps * 8);
        debug_assert_eq!(flat_returns.len(), total_steps);

        let states =
            Tensor::<B, 1>::from_floats(flat_states, &self.device).reshape([total_steps, 8]);
        let targets =
            Tensor::<B, 1>::from_floats(flat_returns, &self.device).reshape([total_steps, 1]);

        let predicted = self.net.forward(states); // [total_steps, 1]
        let loss = mse_loss(predicted, targets); // scalar

        use burn::tensor::ElementConversion;
        let loss_scalar: f32 = loss.clone().into_scalar_async().await.unwrap().elem::<f32>();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.net);
        self.net = self.optimizer.step(learning_rate, self.net.clone(), grads);

        loss_scalar
    }
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

fn mse_loss<B: AutodiffBackend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    (diff.clone() * diff).mean()
}
