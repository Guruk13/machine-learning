// Machine Learning module for Flappy Bird AI
//https://burn.dev/books/burn/basic-workflow/model.html
use burn::{
     config::Config, module::Module, nn::{Linear, LinearConfig, Relu}, prelude::Backend
};
use burn::tensor::Tensor;


use log::{info, warn};
use burn::tensor::activation::sigmoid;

// Define the neural network architecture
#[derive(Module, Debug)]
pub struct FlappyBirdModel<B: Backend> {

    

    // Input: [bird_y, bird_velocity, next_pipe_top_y, next_pipe_bottom_y, next_pipe_distance]
    // Output: [jump_probability]
    linear1: Linear<B>,  // 5 input features -> 8 hidden units
    linear2: Linear<B>,  // 8 hidden units -> 4 hidden units  
    linear3: Linear<B>,  // 4 hidden units -> 1 output (jump probability)
    activation: Relu,


}

impl<B: Backend> FlappyBirdModel<B> {
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
    
    /// Forward pass through the network
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let flap = sigmoid(self.linear3.forward(x)); // Output between 0 and 1
        flap

    }
    

}

// Configuration for the model
#[derive(Config, Debug)]
pub struct FlappyBirdModelConfig {
    pub hidden_size1: usize,
    pub hidden_size2: usize,
}

impl FlappyBirdModelConfig {
    pub fn init<B: Backend >(&self, device: &B::Device) -> FlappyBirdModel<B> {
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
    pub bird_velocity: f32,
    pub next_pipe_top_y: f32,
    pub next_pipe_bottom_y: f32,
    pub next_pipe_distance: f32,
}


impl GameStateFeatures {
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_floats(
            [[
                self.bird_y,
                self.bird_velocity,
                self.next_pipe_top_y,
                self.next_pipe_bottom_y,
                self.next_pipe_distance,
            ]],
            device,
        )
    }
}