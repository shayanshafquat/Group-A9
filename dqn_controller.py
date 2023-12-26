from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class CustomController:
    def __init__(self, state_dim, action_dim, hidden_dim, buffer_capacity, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dqn = NeuralNetwork(state_dim, hidden_dim, action_dim)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        # Additional initialization here

    def get_state(self) -> np.array:
        return self.drone.get_state()

    def set_action(self, action: Tuple[float, float]):
        self.drone.set_action(action)

    def get_reward(self) -> float:
        return self.drone.get_reward()

    def train(self):
        # Implement the training loop, interacting with the drone, updating the DQN, etc.

# Define DQN and ReplayBuffer classes here
class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim):
        # Initialize the weights and biases
        self.w1 = np.random.rand(state_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.rand(hidden_dim, action_dim)
        self.b2 = np.zeros(action_dim)
        self.learning_rate = 0.01

    def forward(self, state):
        # Forward pass through the network
        z1 = state.dot(self.w1) + self.b1
        a1 = np.tanh(z1)  # Activation function
        z2 = a1.dot(self.w2) + self.b2
        return z2

    def backward(self, state, action, error):
        # Backward pass to compute gradients
        z1 = state.dot(self.w1) + self.b1
        a1 = np.tanh(z1)

        dz2 = error
        dw2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.w2.T)
        dz1 = da1 * (1 - np.power(a1, 2))
        dw1 = np.dot(state.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update the weights and biases
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2


# DQN class definition (as provided earlier)

# ReplayBuffer class definition
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



def plot_metrics(metrics):
    # metrics is a dictionary containing lists of rewards, losses, etc.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(metrics['cumulative_rewards'], label='Cumulative Rewards')
    plt.title('Rewards Over Time')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(metrics['losses'], label='Loss')
    plt.title('Loss Over Time')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.title('Epsilon Over Time')
    plt.legend()

    plt.show()


