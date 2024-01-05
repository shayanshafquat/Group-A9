from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-08):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

    # def update(self, params, grads):
    #     for p, g in zip(params, grads):
    #         if p not in self.cache:
    #             self.cache[p] = np.zeros_like(g)
    #         self.cache[p] = self.rho * self.cache[p] + (1 - self.rho) * g**2
    #         params[p] -= self.learning_rate * g / (np.sqrt(self.cache[p]) + self.epsilon)
    def update(self, params, grads):
        for key in params.keys():
            # Ensure that both parameters and gradients are numpy arrays
            if not (isinstance(params[key], np.ndarray) and isinstance(grads[key], np.ndarray)):
                raise TypeError("Parameters and gradients must be numpy arrays")

            # Initialize cache for the parameter if not already initialized
            if key not in self.cache:
                self.cache[key] = np.zeros_like(grads[key])

            # Update the cache and the parameter
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * np.square(grads[key])
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.optimizer = RMSpropOptimizer(learning_rate)
        print(f"w1:{self.w1.shape}\nb1:{self.b1.T.shape}\nw2:{self.w2.shape}\nb2:{self.b2.shape}")

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.tanh(z2)
        return a2, z2, a1, z1
    
    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    
    
    # def backward(self, x, y, output, z2, a1, z1):
    #     # Calculate the loss for monitoring
    #     loss = self.mse_loss(y, output)
    #     output_error = -2 * (y - output) / y.size
    #     output_delta = output_error * (1 - np.tanh(z2)**2)
    #     print(f"output_delta shape:{output_delta.shape}")
    #     z1_error = np.dot(output_delta, self.w2.T)
    #     z1_delta = z1_error * (1 - np.tanh(z1)**2)
    #     z1_delta = z1_delta.reshape(-1,1)
    #     output_delta = output_delta.reshape(-1,1)
    #     a1 = a1.reshape(-1,1)
    #     x = x.reshape(-1,1)
    #     print(f"z1_delta shape:{z1_delta.shape}")
    #     # print(f"{self.w1.shape}{x.shape}")
    #     # Gradients for weights and biases
    #     w2_grad = np.dot(output_delta,a1.T)
    #     b2_grad = np.sum(output_delta, axis=0)
    #     w1_grad = np.dot(z1_delta, x.T)
    #     b1_grad = np.sum(z1_delta, axis=0)

    #     # Update weights and biases using RMSprop
    #     self.optimizer.update({'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2},
    #                           {'w1': w1_grad, 'b1': b1_grad, 'w2': w2_grad, 'b2': b2_grad})
        
        # return loss
    
    # def backward(self, x, y, output, z2, a1, z1):
    #     loss = self.mse_loss(y, output)

    #     # Backward pass (gradient descent)
    #     output_error = y - output  # error in output
    #     output_delta = output_error  # assuming linear activation for output layer

    #     z1_error = np.dot(output_delta, self.w2.T)
    #     z1_delta = z1_error * (1 - np.power(a1, 2))  # derivative of tanh

    #     # Update weights and biases
    #     self.w2 += np.outer(a1, output_delta) * self.learning_rate
    #     self.b2 += output_delta * self.learning_rate
    #     self.w1 += np.outer(x, z1_delta) * self.learning_rate
    #     self.b1 += z1_delta * self.learning_rate

    #     return loss
    
    def backward(self, x, y_true, y_pred, z2, a1, z1):
        # Compute the gradient of the loss with respect to the output of the last layer
        d_loss_y_pred = 2 * (y_pred - y_true) / y_pred.shape[0]

        # Gradient through the second tanh activation
        d_z2 = d_loss_y_pred * (1 - np.tanh(z2)**2)

        # Gradients with respect to weights and biases of the second layer
        d_w2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)

        # Gradient through the first layer
        d_a1 = np.dot(d_z2, self.w2.T)
        d_z1 = d_a1 * (1 - np.tanh(z1)**2)

        # Gradients with respect to weights and biases of the first layer
        d_w1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        # Update the weights and biases
        self.w1 -= self.learning_rate * d_w1
        self.w2 -= self.learning_rate * d_w2
        self.b1 -= self.learning_rate * d_b1
        self.b2 -= self.learning_rate * d_b2

        # Compute the loss (optional, for monitoring purposes)
        loss = np.mean((y_true - y_pred) ** 2)

        return loss






class DQNController(FlightController):
    def __init__(self,
                state_size,
                num_actions_per_propeller,
                hidden_size=64, 
                learning_rate=1e-4, 
                gamma=0.99, 
                epsilon=1.0, 
                epsilon_decay=0.995, 
                min_epsilon=0.01, 
                memory_size=10000):
        super().__init__()
        self.state_size = state_size
        self.num_actions_per_propeller = num_actions_per_propeller
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
        self.action_space = self.discretize_action_space(self.num_actions_per_propeller)
        self.action_size = len(self.action_space)
        self.model = self.build_model()
        self.target_model = NeuralNetwork(self.state_size, self.hidden_size, self.action_size, self.learning_rate)

    
    def build_model(self):
        # Input size = state size, output size = number of discrete actions
        model = NeuralNetwork(self.state_size, self.hidden_size, self.action_size, self.learning_rate)
        return model

    def update_target_network(self):
        self.target_model.w1 = self.model.w1.copy()
        self.target_model.b1 = self.model.b1.copy()
        self.target_model.w2 = self.model.w2.copy()
        self.target_model.b2 = self.model.b2.copy()

    def discretize_action_space(self, num_actions_per_propeller):
        # Create a grid of actions based on the number of actions for each propeller
        self.action_grid = np.linspace(0, 1, num_actions_per_propeller)
        self.discrete_actions = np.array(np.meshgrid(self.action_grid, self.action_grid)).T.reshape(-1, 2)
        return self.discrete_actions    

    def get_state(self, drone):
        # Efficiently gather the state attributes from the drone object
        target_x, target_y = drone.get_next_target()
        return np.array([drone.x, drone.y, drone.velocity_x, drone.velocity_y, drone.pitch, drone.pitch_velocity, drone.thrust_left, drone.thrust_right, target_x, target_y])

    def get_reward(self, state, drone):
        # Efficient computation of the reward using numpy operations
        distance_to_target = np.hypot(drone.x - state[8], drone.y - state[9])
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

    # def replay(self, batch_size):
    #     minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
    #     minibatch = [self.memory[i] for i in minibatch_indices]  # Retrieve experiences using indices
    #     losses = []
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             q_next, _, _, _ = self.target_model.forward(next_state)
    #             target = reward + self.gamma * np.amax(q_next)       
    #         q_values, z2, a1, z1 = self.model.forward(state)
    #         # print(action)
    #         q_target = q_values.copy()
    #         # print(q_target)
    #         q_target[action] = target
    #         # Perform a gradient descent step here to update the weights and biases
    #         loss = self.model.backward(state, q_target, q_values, z2, a1, z1)
    #         losses.append(loss)
    #     average_loss = np.mean(losses)
    #     return average_loss
    def replay(self, batch_size):
        minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        # Prepare batched inputs
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Batch forward pass for current states
        q_values, z2, a1, z1 = self.model.forward(states)

        # Batch forward pass for next states
        q_next, _, _, _ = self.target_model.forward(next_states)

        # Vectorized target calculation
        targets = rewards + self.gamma * np.amax(q_next, axis=1) * (~dones)

        # Update Q-values for the actions taken
        q_target = q_values.copy()
        q_target[np.arange(batch_size), actions] = targets

        # Batch backward pass
        loss = self.model.backward(states, q_target, q_values, z2, a1, z1 )  # Add other necessary parameters
        average_loss = np.mean(loss)
        return average_loss

    def act(self, state):

        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(self.discrete_actions))
        # print(state)
        q_values,_,_,_ = self.model.forward(state)
        return np.argmax(q_values)
    
    def get_thrust(self, drone: Drone):
        state = self.get_state(drone)
        q_values, _, _, _ = self.model.forward(state)
        thrust_left, thrust_right = self.discrete_actions[np.argmax(q_values)]
        return (thrust_left, thrust_right)
    
    def step(self, drone, state, action_index):
        # Convert discrete action back to continuous action
        action = self.discrete_actions[action_index]
        drone.set_thrust(action)                   
        drone.step_simulation(self.get_time_interval())   
        # print(drone.thrust_left, drone.thrust_right) 

        next_state = self.get_state(drone)
        reward = self.get_reward(next_state, drone)
        done = drone.has_reached_target_last_update   
        return next_state, reward, done
    
    def train(self):
        for e in range(self.num_episodes):
            # Reset the environment here and get the initial state
            drone = self.init_drone()               # flight controller init_drone method used here
            state = self.get_state(drone) 

            total_reward = 0
            target_index = 0

            for time in range(self.get_max_simulation_steps()):
                action_index = self.act(state)
                # Execute the action in the simulator. Observe the next state and reward
                next_state, reward, done = self.step(drone, state, action_index)
                # Store the transition in replay memory
                self.remember(state, action_index, reward, next_state, done)
                total_reward += reward
            
                if done:
                    target_index += 1
                    if (target_index == 4):
                        print(f"Episode: {e + 1}, Total Reward: {total_reward}, Steps: {time + 1}")
                        break
                state = next_state
                if len(self.memory) > self.batch_size:
                    average_loss = self.replay(self.batch_size)
                    # print(f"Episode: {e + 1}, Step: {time + 1}, Average Loss: {average_loss:.4f}")

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            self.update_target_network()
            # Print cumulative reward at the end of each episode
            print(f"Episode: {e + 1} finished with Total Reward: {total_reward}")

            # Save the model periodically or at the end of training
            if e % 100 == 0 or e == self.num_episodes - 1:
                self.save(f"dqn_controller_{e}.npz")
        print("Training finished.")
            

    def save(self, filename):
        np.savez(filename, w1=self.model.w1, b1=self.model.b1, w2=self.model.w2, b2=self.model.b2)

    def load(self0):
        filename = "dqn_controller_400.npz"
        data = np.load(filename)
        self.model.w1 = data['w1']
        self.model.b1 = data['b1']
        self.model.w2 = data['w2']
        self.model.b2 = data['b2']
        self.update_target_network()


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


