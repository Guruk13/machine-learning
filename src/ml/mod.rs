use bevy::reflect::Reflect;
// Machine Learning module for Flappy Bird AI
//https://burn.dev/books/burn/basic-workflow/model.html
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
};

use log::{info, warn};

// Define the neural network architecture
#[derive(Module, Debug)]
pub struct FlappyBirdModel<B: Backend> {
    // Input: [bird_y, bird_fall_rate, next_pipe_top_y, next_pipe_bottom_y, next_pipe_distance]
    // Output: [jump_probability]
    linear1: Linear<B>, // 5 input features -> 8 hidden units
    linear2: Linear<B>, // 8 hidden units -> 4 hidden units
    linear3: Linear<B>, // 4 hidden units -> 1 output (jump probability)
    activation: Relu,
}

impl<B: Backend> FlappyBirdModel<B> where
    B: Backend<BoolElem = bool> {
    /// Initialize a new model with random weights
    pub fn new(device: Option<B::Device>) -> Self {
        let device = device.unwrap_or(B::Device::default());

        Self {
            activation: Relu::new(),
            linear1: LinearConfig::new(5, 8).init(&device),
            linear2: LinearConfig::new(8, 4).init(&device),
            linear3: LinearConfig::new(4, 1).init(&device),
        }
    }

pub fn forward(&self, input: Tensor<B, 2>) -> bool {
    let x = self.linear1.forward(input);
    let x = self.activation.forward(x);
    let x = self.linear2.forward(x);
    let x = self.activation.forward(x);
    let x = sigmoid(self.linear3.forward(x)); // Tensor<B, 2> shape [1, 1]

    let flap: bool = x
        .greater_elem(0.5)      // Tensor<B, 2, Bool>  shape [1, 1]
        .reshape(-1)            // Tensor<B, 1, Bool>  shape [1]
        .squeeze(0)              // Tensor<B, 0, Bool>  scalar
        .into_scalar();          // bool                // plain Rust bool

    warn!("{:#?}", flap);
    flap
}

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

// Configuration for the model
#[derive(Config, Debug)]
pub struct FlappyBirdModelConfig {
    pub hidden_size1: usize,
    pub hidden_size2: usize,
}

impl FlappyBirdModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FlappyBirdModel<B> {
        FlappyBirdModel {
            activation: Relu::new(),
            linear1: LinearConfig::new(5, self.hidden_size1).init(device),
            linear2: LinearConfig::new(self.hidden_size1, self.hidden_size2).init(device),
            linear3: LinearConfig::new(self.hidden_size2, 1).init(device),
        }
    }
}

// Game state representation for ML input
#[derive(Debug, Clone, Copy)]
pub struct GameStateFeatures {
    pub bird_y: f32,
    pub bird_fall_rate: f32,
    pub next_pipe_top_y: f32,
    pub next_pipe_bottom_y: f32,
    pub next_pipe_distance: f32,
}

impl GameStateFeatures {
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_floats(
            [[
                self.bird_y,
                self.bird_fall_rate,
                self.next_pipe_top_y,
                self.next_pipe_bottom_y,
                self.next_pipe_distance,
            ]],
            device,
        )
    }
}
