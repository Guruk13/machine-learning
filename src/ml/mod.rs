// Machine Learning module for Flappy Bird AI — DQN implementation
// Architecture:
//   - Online network  : picks actions, trained every step
//   - Target network  : computes TD targets, synced from online every N steps
//   - Replay buffer   : stores (s, a, r, s', done) transitions
//   - ε-greedy policy : exploration decays from 1.0 → 0.05

use bevy::prelude::*;
use burn::{
    config::Config,
    module::AutodiffModule,
    module::Module,
    nn::{Linear, LinearConfig, Relu, loss::MseLoss, loss::Reduction},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};

use rand::{Rng, RngExt};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::VecDeque;

// ─────────────────────────────────────────────
// 1.  NETWORK
// ─────────────────────────────────────────────

/// Input  : 6 features  (bird_y, bird_speed, pipe_top_y, pipe_bottom_y, dist_top, dist_bot)
/// Output : 2 Q-values  (Q[do-nothing], Q[jump])
#[derive(Module, Debug)]
pub struct DQNModel<B: Backend> {
    linear1: Linear<B>, // 6 → 64
    linear2: Linear<B>, // 64 → 64
    linear3: Linear<B>, // 64 → 2
    activation: Relu,
}

impl<B: Backend> DQNModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            activation: Relu::new(),
            linear1: LinearConfig::new(6, 16).init(device),
            linear2: LinearConfig::new(16, 16).init(device),
            linear3: LinearConfig::new(16, 2).init(device),
        }
    }

    /// Forward pass — input shape [batch, 6], output shape [batch, 2]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.linear3.forward(x) // raw Q-values, no activation
    }
}

#[derive(Config, Debug)]
pub struct DQNModelConfig;

impl DQNModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DQNModel<B> {
        DQNModel::new(device)
    }
}

// ─────────────────────────────────────────────
// 2.  STATE FEATURES
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
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

// ─────────────────────────────────────────────
// 3.  ACTION
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    DoNothing = 0,
    Jump = 1,
}

// ─────────────────────────────────────────────
// 4.  REPLAY BUFFER
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Transition {
    pub state: GameStateFeatures,
    pub action: Action,
    pub reward: f32,
    pub next_state: GameStateFeatures,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, t: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(t);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Sample `n` random transitions (with replacement is fine for large buffers).
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<&Transition> {
        let len = self.buffer.len();
        (0..n)
            .map(|_| &self.buffer[rand::random_range(0..len)])
            .collect()
    }
}

// ─────────────────────────────────────────────
// 5.  DQN AGENT  (Bevy Resource)
// ─────────────────────────────────────────────

pub struct DQNAgent<B: AutodiffBackend> {
    // Networks
    pub online: DQNModel<B>,
    pub target: DQNModel<B::InnerBackend>, // frozen, no grad
    // Optimizer
    optimizer: OptimizerAdaptor<Adam, DQNModel<B>, B>,
    // Replay
    pub replay: ReplayBuffer,
    // Hyper-parameters
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub batch_size: usize,
    pub target_sync_every: usize,

    // Counters
    pub steps: usize,
    pub device: B::Device,
}

impl<B: AutodiffBackend> DQNAgent<B> {
    pub fn new(device: B::Device) -> Self {
        let online = DQNModel::<B>::new(&device);
        // Clone weights to target (inner backend = no autograd)
        let target = online.clone().valid();

        let optimizer = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<B, DQNModel<B>>();

        Self {
            online,
            target,
            optimizer,
            replay: ReplayBuffer::new(50_000),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.9995, // multiply each step
            batch_size: 64,
            target_sync_every: 500,
            steps: 0,
            device,
        }
    }

    // ── Action selection ──────────────────────────────────────────────────

    /// ε-greedy: explore randomly or exploit the Q-network.
    pub fn select_action(&self, state: &GameStateFeatures, rng: &mut impl Rng) -> Action {
        if rng.random::<f32>() < self.epsilon {
            if rng.random_bool(0.5) {
                Action::Jump
            } else {
                Action::DoNothing
            }
        } else {
            self.greedy_action(state)
        }
    }

    /// Pure greedy (used at inference / eval time).
    pub fn greedy_action(&self, state: &GameStateFeatures) -> Action {
        let input = Tensor::<B, 2>::from_floats([state.to_array()], &self.device);
        let q = self.online.forward(input); // [1, 2]
        // argmax over action dimension
        let action_idx: i64 = q.argmax(1).into_scalar().elem();
        if action_idx == 1 {
            Action::Jump
        } else {
            Action::DoNothing
        }
    }

    // ── Learning step ─────────────────────────────────────────────────────

    /// Store a transition and, if the buffer is full enough, run one SGD step.
    pub fn step(&mut self, transition: Transition, rng: &mut impl Rng) {
        self.replay.push(transition);
        self.steps += 1;

        // Decay ε
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        // Sync target network periodically
        if self.steps % self.target_sync_every == 0 {
            self.target = self.online.clone().valid();
            //info!("[DQN] target network synced at step {}", self.steps);
        }

        // Train only once buffer has enough samples
        if self.replay.len() < self.batch_size {
            return;
        }

        self.train_step(rng);
    }

    fn train_step(&mut self, rng: &mut impl Rng) {
        let batch = self.replay.sample(self.batch_size, rng);

        // Build tensors  [batch, 6]
        let states_data: Vec<[f32; 6]> = batch.iter().map(|t| t.state.to_array()).collect();
        let next_states_data: Vec<[f32; 6]> =
            batch.iter().map(|t| t.next_state.to_array()).collect();
        let actions: Vec<usize> = batch.iter().map(|t| t.action as usize).collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let dones: Vec<f32> = batch
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();

        let b = self.batch_size;
        let dev = &self.device.clone();

        // ── States → Q predictions (online network, with grad) ────────────
        // Flatten to [batch*6] then reshape
        let states_flat: Vec<f32> = states_data.into_iter().flatten().collect();
        let states_tensor =
            Tensor::<B, 1>::from_floats(states_flat.as_slice(), dev).reshape([b, 6]);

        let q_all = self.online.forward(states_tensor); // [b, 2]

        // Gather Q(s, a) for the taken action
        let action_indices = Tensor::<B, 1, burn::tensor::Int>::from_ints(
            actions
                .iter()
                .map(|&a| a as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            dev,
        )
        .reshape([b, 1]);

        let q_sa = q_all.gather(1, action_indices).squeeze(1); // [b]

        // ── Next states → target Q (target network, no grad) ─────────────
        let next_flat: Vec<f32> = next_states_data.into_iter().flatten().collect();

        // Target network uses inner (non-autodiff) backend
        let next_tensor =
            Tensor::<B::InnerBackend, 1>::from_floats(next_flat.as_slice(), dev).reshape([b, 6]);

        let q_next = self.target.forward(next_tensor); // [b, 2]
        let q_next_max = q_next.max_dim(1).squeeze(1); // [b]

        // ── Bellman target ────────────────────────────────────────────────
        let rewards_t = Tensor::<B::InnerBackend, 1>::from_floats(rewards.as_slice(), dev);
        let dones_t = Tensor::<B::InnerBackend, 1>::from_floats(dones.as_slice(), dev);

        // target = r + γ * max_a Q_target(s', a) * (1 - done)
        let target_q = rewards_t + q_next_max * (dones_t.neg() + 1.0) * self.gamma;

        // Bring target into autodiff graph so we can compute loss
        let target_q_ad = Tensor::<B, 1>::from_inner(target_q);

        // ── MSE loss ──────────────────────────────────────────────────────
        let loss = MseLoss::new().forward(q_sa, target_q_ad, Reduction::Mean);

        // ── Backprop + optimizer step ─────────────────────────────────────
        let gradients = loss.backward();
        let grads = GradientsParams::from_grads(gradients, &self.online);
        self.online = self.optimizer.step(1e-3, self.online.clone(), grads);
    }
}

// ─────────────────────────────────────────────
// 6.  BEVY INTEGRATION
// ─────────────────────────────────────────────

// We use the NdArray CPU backend with autodiff.
pub type MyBackend = burn::backend::Autodiff<burn::backend::NdArray>;

/// Wrap the agent in a Bevy Resource.
#[derive(Resource)]
pub struct DQNResource {
    pub agent: DQNAgent<MyBackend>,
    pub rng: rand::rngs::SmallRng,
}

impl DQNResource {
    pub fn new() -> Self {
        use rand::SeedableRng;
        //let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        Self {
            agent: DQNAgent::new(device),
            rng: StdRng::from_entropy(),
        }
    }
}

// ── Per-bird component ────────────────────────────────────────────────────────

/// Attach to every AI bird entity.
#[derive(Component, Default)]
pub struct BirdAgent {
    /// The state observed at the beginning of the current frame.
    pub last_state: Option<GameStateFeatures>,
    /// The action chosen for the current frame.
    pub last_action: Option<Action>,
    /// Cumulative reward this episode (for logging).
    pub episode_reward: f32,
}

// ── Reward shaping ────────────────────────────────────────────────────────────

/// Call this every tick to compute the reward for the current frame.
/// - alive           : +0.1  (small survival bonus per frame)
/// - passed a pipe   : +5.0
/// - died            : −10.0
pub fn compute_reward(alive: bool, passed_pipe: bool) -> f32 {
    if !alive {
        return -10.0;
    }
    let mut r = 0.1;
    if passed_pipe {
        r += 5.0;
    }
    r
}

// ─────────────────────────────────────────────
// 7.  EXAMPLE BEVY SYSTEMS
//     Wire these into your App as needed.
// ─────────────────────────────────────────────

/// Call once per physics tick, BEFORE movement is applied.
/// Reads the current game state, runs the policy, and records the chosen action.
///
/// Replace the query parameters with your actual Bevy components.
///
/// ```rust
/// app.add_systems(Update, dqn_act_system.before(bird_movement_system));
/// ```
pub fn dqn_act_system(
    mut dqn: ResMut<DQNResource>,
    mut birds: Query<(&mut BirdAgent, &BirdStateProvider)>,
) {
    let rng = &mut dqn.rng as *mut _;
    let agent = &dqn.agent;
    for (mut agent_comp, provider) in birds.iter_mut() {
        let state = provider.observe(); // you implement this
        let action = agent.select_action(&state, unsafe { &mut *rng });
        agent_comp.last_state = Some(state);
        agent_comp.last_action = Some(action);
        // Your game system should read `last_action` and apply the jump.
    }
}

/// Call once per physics tick, AFTER collision / scoring is resolved.
/// Stores the transition and triggers a training step.
///
/// ```rust
/// app.add_systems(Update, dqn_learn_system.after(collision_system));
/// ```
pub fn dqn_learn_system(
    mut dqn: ResMut<DQNResource>,
    mut birds: Query<(&mut BirdAgent, &BirdStateProvider, &BirdOutcome)>,
) {
    let rng = &mut dqn.rng as *mut _;
    for (mut agent_comp, provider, outcome) in birds.iter_mut() {
        let (Some(prev_state), Some(action)) =
            (agent_comp.last_state, agent_comp.last_action)
        else {
            continue;
        };

        let next_state = provider.observe();
        let reward = compute_reward(outcome.alive, outcome.passed_pipe);
        agent_comp.episode_reward += reward;

        let transition = Transition {
            state: prev_state,
            action,
            reward,
            next_state,
            done: !outcome.alive,
        };

        dqn.agent.step(transition, unsafe { &mut *rng });

        if !outcome.alive {
            info!(
                "[DQN] episode finished | reward={:.1} | ε={:.4} | steps={}",
                agent_comp.episode_reward, dqn.agent.epsilon, dqn.agent.steps
            );
            agent_comp.episode_reward = 0.0;
        }
    }
}

// ── Stub traits you must implement in your game code ─────────────────────────

/// Add this component (or equivalent) to each bird entity and implement `observe`.
#[derive(Component)]
pub struct BirdStateProvider;          // replace with your real component

impl BirdStateProvider {
    /// Build the feature vector from Bevy query data.
    /// In practice you'll have access to Transform, Velocity, and pipe positions.
    pub fn observe(&self) -> GameStateFeatures {
        todo!("Query your bird Transform / pipe entities here")
    }
}

/// Outcome for the current frame — filled by your collision / scoring systems.
#[derive(Component, Default)]
pub struct BirdOutcome {
    pub alive: bool,
    pub passed_pipe: bool,
}

