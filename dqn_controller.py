from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        pass
    def train(self):
        pass    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, x):
        # Forward pass through the network
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)  # Activation function for hidden layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.tanh(self.z2)  # Assuming tanh activation for output layer
        return self.a2
    
    def backward(self, x, y, output):
        # Backward pass (gradient descent)
        output_error = y - output  # error in output
        output_delta = output_error  # assuming linear activation for output layer

        z1_error = np.dot(output_delta, self.w2.T)
        z1_delta = z1_error * (1 - np.power(self.a1, 2))  # derivative of tanh

        # Update weights and biases
        self.w2 += np.dot(self.a1.T, output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.w1 += np.dot(x.T, z1_delta) * self.learning_rate
        self.b1 += np.sum(z1_delta, axis=0, keepdims=True) * self.learning_rate

class DQNController(FlightController):
    def __init__(self,
                state_size,
                action_size,
                hidden_size=64, 
                learning_rate=1e-4, 
                gamma=0.99, 
                epsilon=1.0, 
                epsilon_decay=0.995, 
                min_epsilon=0.01, 
                memory_size=10000):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = []  # Experience replay buffer
        self.memory_size = memory_size
        self.num_episodes = 500 
        self.batch_size = 64
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = self.build_model()

    
    def build_model(self):
        # Input size = state size, output size = number of discrete actions
        self.model = NeuralNetwork(self.state_size, self.hidden_size, self.action_size, self.learning_rate)

    def discretize_action_space(self, num_actions_per_propeller):
        # Create a grid of actions based on the number of actions for each propeller
        self.action_grid = np.linspace(0, 1, num_actions_per_propeller)
        self.discrete_actions = np.array(np.meshgrid(self.action_grid, self.action_grid)).T.reshape(-1, 2)
        return self.discrete_actions
    
    def continuous_action(self, discrete_action_index):
        return self.discrete_actions[discrete_action_index]

    def get_state(self, drone):
        # Efficiently gather the state attributes from the drone object
        target_x, target_y = drone.get_next_target()
        return np.array([drone.x, drone.y, drone.velocity_x, drone.velocity_y, drone.pitch, drone.pitch_velocity, target_x, target_y])

    def get_reward(self, drone):
        # Efficient computation of the reward using numpy operations
        distance_to_target = np.hypot(drone.x - drone.target_x, drone.y - drone.target_y)
        reward = (
            100.0 * drone.has_reached_target_last_update -
            10.0 * distance_to_target -
            1.0 * (drone.thrust_left + drone.thrust_right) -
            0.1
        )
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        # Add experience to the memory
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Remove the oldest experience if the memory is full

    def replay(self, batch_size):
        # Sample a batch of experiences from the memory and learn from them
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.forward(next_state))
            target_f = self.model.forward(state)
            target_f[0][action] = target
            # Perform a gradient descent step here to update the weights and biases
            self.model.backward(state, target, target_f)
    
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(self.discrete_actions))
        q_values = self.model.forward(state)
        return np.argmax(q_values)
    
    def step(self, state, action_index):
        # Convert discrete action to continuous action
        continuous_action = self.continuous_action(action_index)
        self.drone.set_thrust(continuous_action)
        self.drone.step_simulation(self.get_time_interval())

        next_state = self.get_state(self.drone)
        reward = self.get_reward(self.drone)
        done = self.drone.has_reached_target_last_update
        return next_state, reward, done
    
    def train(self, episodes, batch_size, drone:Drone):
        # The main training loop
        for e in range(self.episodes):
            # Reset the environment here and get the initial state
            state = self.get_state(drone)
            for time in range(self.get_max_simulation_steps()):
                action = self.act(state)
                # Convert discrete action back to continuous action
                action_index = self.discrete_actions[action]
                next_state, reward, done = self.step(state, action_index)
                self.remember(state, action_index, reward, next_state, done)
                state = next_state
                if done:
                    break
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self, filename):
        np.savez(filename, w1=self.model.w1, b1=self.model.b1, w2=self.model.w2, b2=self.model.b2)

    def load(self, filename):
        data = np.load(filename)
        self.model.w1 = data['w1']
        self.model.b1 = data['b1']
        self.model.w2 = data['w2']
        self.model.b2 = data['b2']

# Instantiate DQNController with appropriate sizes (placeholders)
state_size = 8  # Assuming 8 state features as per get_state definition
action_size = 12  # Assuming we discretize each propeller into 6 actions (up/down)
controller = DQNController(state_size, action_size)
controller.discretize_action_space(action_size)


# # ReplayBuffer class definition
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0

#     def add(self, state, action, reward, next_state, done):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def __len__(self):
#         return len(self.buffer)


# def plot_metrics(metrics):
#     # metrics is a dictionary containing lists of rewards, losses, etc.
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.plot(metrics['cumulative_rewards'], label='Cumulative Rewards')
#     plt.title('Rewards Over Time')
#     plt.legend()

#     plt.subplot(1, 3, 2)
#     plt.plot(metrics['losses'], label='Loss')
#     plt.title('Loss Over Time')
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     plt.plot(metrics['epsilons'], label='Epsilon')
#     plt.title('Epsilon Over Time')
#     plt.legend()

#     plt.show()


