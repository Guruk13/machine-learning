pub mod agent_utils; // stores performance , entropy evaluation and state for a bird
pub mod critic;
pub mod model; //model
pub mod multiagent; // breaches the gap between simulation and model
pub mod pruner; // runs optimization with entropy detection
pub const OPTIMIZER_EPSILON: f32 = 1e-7;
