from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
from q_learning import heuristic1, heuristic2

class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-08):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

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
        self.dropout_rate = 0.4
        print(f"w1:{self.w1.shape}\nb1:{self.b1.T.shape}\nw2:{self.w2.shape}\nb2:{self.b2.shape}")

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        a1 = self.dropout(a1, self.dropout_rate)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.tanh(z2)
        return a2, z2, a1, z1
    
    def dropout(self, X, drop_probability):
        keep_probability = 1 - drop_probability
        mask = np.random.binomial(1, keep_probability, size=X.shape)
        return mask * X / keep_probability
    
    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    

    def backward(self, x, y, output, z2, a1, z1):
        # Calculate output error using the derivative of MSE
        output_error = -2 * (y - output) / y.size  # Assuming y is a numpy array

        output_delta = output_error * (1 - np.tanh(z2)**2)

        z1_error = np.dot(output_delta, self.w2.T)
        z1_delta = z1_error * (1 - np.tanh(z1)**2)

        # Gradients for weights and biases
        w2_grad = np.dot(a1.T, output_delta)
        b2_grad = np.sum(output_delta, axis=0)
        w1_grad = np.dot(x.T, z1_delta)
        b1_grad = np.sum(z1_delta, axis=0)

        # Update weights and biases using RMSprop
        self.optimizer.update({'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2},
                            {'w1': w1_grad, 'b1': b1_grad, 'w2': w2_grad, 'b2': b2_grad})

        # Compute the loss (optional, for monitoring purposes)
        loss = np.mean((y - output) ** 2)

        return loss

class DQNController(FlightController):
    def __init__(self,
                state_size=3**4,
                action_size=2,
                hidden_size=48, 
                learning_rate=0.05, 
                gamma=0.95, 
                epsilon=1.0, 
                epsilon_decay=0.99, 
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
        self.num_episodes = 100
        self.evaluation_interval = 5
        self.batch_size = 512
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
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
 

    def get_state(self, drone:Drone):

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        velocity_y = drone.velocity_y
        pitch = drone.pitch
        pitch_velocity = drone.pitch_velocity

        # Define the discretization steps for each component
        step_size = 0.1  # e.g., 1 unit of space
        pitch_step = np.pi / 9  # e.g., 10 degrees


        # discretized_dx = 0 if dx < -0.1 else (2 if dx > 0.1 else 1)
        # discretized_dy = 0 if dy >= 0 else 1
        discretized_dy = 0 if dy < -0.1 else (2 if dy > 0.1 else 1)
        # discretized_velocity_y = 0 if drone.velocity_y >= 0 else 1
        discretized_velocity_y = 0 if velocity_y < -0.1 else (2 if velocity_y > 0.1 else 1)
        # discretized_pitch = 0 if drone.pitch >= 0 else 1 
        discretized_pitch = 0 if pitch < -0.1 else (2 if pitch > 0.1 else 1)
        # discretized_pitch_velocity = 0 if drone.pitch_velocity >= 0 else 1
        discretized_pitch_velocity = 0 if pitch_velocity < -0.2 else (2 if pitch_velocity > 0.2 else 1)
   

        # Combine into a single state index
        state_index = (discretized_dy +
                       discretized_velocity_y * 3**1 +
                       discretized_pitch * 3**2 +
                       discretized_pitch_velocity * 3**3 
                       )
        if(state_index < 0):
            print("state index is negative")

        # Total number of possible states
        num_states = 3**4

        # One-hot encode the state index
        state_vector = np.zeros(num_states)
        state_vector[state_index] = 1

        return state_vector

    def get_reward(self, drone):
        # Efficient computation of the reward using numpy operations
        target_point = drone.get_next_target()
        distance_to_target = np.hypot(drone.x - target_point[0], drone.y - target_point[1])
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

    def replay(self, batch_size, num_iterations=5):
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
        # Loop for multiple backward passes
        total_loss = 0
        for _ in range(num_iterations):
            # Call backward pass
            loss = self.model.backward(states, q_target, q_values, z2, a1, z1)
            total_loss += loss

        average_loss = total_loss / num_iterations
        return average_loss

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        # print(state)
        q_values,_,_,_ = self.model.forward(state)
        return np.argmax(q_values)
    
    def discrete_actions(self, action_index, drone):

        best_param_heuristic_1 = {'ky': 3, 'kx': 2.6, 'abs_pitch_delta': 0.1, 'abs_thrust_delta':0.4}
        best_param_heuristic_2 = {'k': 8.25, 'b': 0.3, 'k_theta': 9.5, 'b_theta': 0.3, 'theta_target': 7}
        
        ky = best_param_heuristic_1['ky']
        kx = best_param_heuristic_1['kx']
        abs_pitch_delta = best_param_heuristic_1['abs_pitch_delta']
        abs_thrust_delta = best_param_heuristic_1['abs_thrust_delta']

        k = best_param_heuristic_2['k']
        b = best_param_heuristic_2['b']
        k_theta = best_param_heuristic_2['k_theta']
        b_theta = best_param_heuristic_2['b_theta']
        theta_target = best_param_heuristic_2['theta_target']

        if action_index == 0:
            thrust_left, thrust_right = heuristic1(ky, kx, abs_pitch_delta, abs_thrust_delta, drone)
        else:
            thrust_left, thrust_right = heuristic2(k, b, k_theta, b_theta, theta_target, drone)
        return thrust_left, thrust_right
    
    def get_thrusts(self, drone: Drone):
        state = self.get_state(drone)
        q_values, _, _, _ = self.model.forward(state)
        thrust_left, thrust_right = self.discrete_actions(np.argmax(q_values), drone)
        return (thrust_left, thrust_right)
    
    def step(self, drone, state, action_index):
        # Convert discrete action back to continuous action
        action = self.discrete_actions(action_index, drone)
        drone.set_thrust(action)                   
        drone.step_simulation(self.get_time_interval())   
        # print(drone.thrust_left, drone.thrust_right) 

        next_state = self.get_state(drone)
        reward = self.get_reward(drone)
        done = drone.has_reached_target_last_update   
        if done:
            self.target_index += 1
        return next_state, reward, done
    
    def train(self):

        evaluation_epochs = []
        cumulative_rewards = []
        best_performance = float('-inf') 

        for e in range(self.num_episodes):
            # Reset the environment here and get the initial state
            drone = self.init_drone()               # flight controller init_drone method used here
            state = self.get_state(drone) 

            total_reward = 0
            self.target_index = 0
            total_loss = 0
            replay_count = 0

            for time in range(self.get_max_simulation_steps()):
                action_index = self.act(state)
                # Execute the action in the simulator. Observe the next state and reward
                next_state, reward, done = self.step(drone, state, action_index)
                # Store the transition in replay memory
                self.remember(state, action_index, reward, next_state, done)
                total_reward += reward
            
                if (self.target_index == 5):
                    print(f"Done Epoch: {e + 1}, Total Reward: {total_reward}, Steps: {time + 1}")
                    break

                state = next_state

                if len(self.memory) > self.batch_size:
                    average_loss = self.replay(self.batch_size)
                    total_loss += average_loss
                    replay_count += 1

            # Compute and print the mean loss for the episode
            if replay_count > 0:
                mean_loss = total_loss / replay_count
                print(f"Episode: {e + 1}, Mean Loss: {mean_loss:.4f}, Cumulative Reward / Step: {total_reward/self.get_max_simulation_steps()}")

            self.update_target_network()                

            # Save the model periodically or at the end of training
            if e % self.evaluation_interval == 0 or e == self.num_episodes - 1:
                performance = self.evaluate_performance()
                cumulative_rewards.append(performance)
                evaluation_epochs.append(e + 1)

                # Print cumulative reward at the end of each episode
                print(f"Epoch {e+1}: Cumulative reward / Step: {performance}")

                if performance >= best_performance:
                    best_performance = performance
                    print("Improved!")
                    self.save(f"./Results/dqn/dqn_controller_{e}.npz")

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                
        print("Training finished.")
            
    def evaluate_performance(self):
        """
        Evaluate the performance of the current controller settings.

        Returns:
            float: A performance score, higher is better.
        """
        total_performance = 0
        num_eval_episodes = 3  # Number of episodes to run for evaluation

        for _ in range(num_eval_episodes):
            drone = self.init_drone()  # Initialize the drone for each evaluation episode
            episode_performance = 0

            for _ in range(self.get_max_simulation_steps()):
                # Get the thrusts based on current ky and kx values
                thrusts = self.get_thrusts(drone)
                drone.set_thrust(thrusts)

                # Update the drone's state
                drone.step_simulation(self.get_time_interval())

                # Calculate the reward for the current step
                reward = self.get_reward(drone)
                episode_performance += reward

            # Average performance over the episode
            total_performance += episode_performance / self.get_max_simulation_steps()

        # Average performance over all evaluation episodes
        return total_performance / num_eval_episodes
    
    def save(self, filename="final"):
        np.savez(filename, w1=self.model.w1, b1=self.model.b1, w2=self.model.w2, b2=self.model.b2)

    def load(self):
        filename = "dqn_controller_499.npz"
        data = np.load(filename)
        self.model.w1 = data['w1']
        self.model.b1 = data['b1']
        self.model.w2 = data['w2']
        self.model.b2 = data['b2']
        self.update_target_network()



