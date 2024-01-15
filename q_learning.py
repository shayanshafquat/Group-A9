from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import json
import matplotlib.pyplot as plt
import os


def heuristic1(ky, kx, abs_pitch_delta, abs_thrust_delta, drone: Drone) -> Tuple[float, float]:

    target_point = drone.get_next_target()
    dx = target_point[0] - drone.x
    dy = target_point[1] - drone.y

    thrust_adj = np.clip(dy * ky, -abs_thrust_delta, abs_thrust_delta)
    target_pitch = np.clip(dx * kx, -abs_pitch_delta, abs_pitch_delta)
    delta_pitch = target_pitch-drone.pitch

    thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
    thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

    # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
    return (thrust_left, thrust_right)

def heuristic2(k, b, k_theta, b_theta, theta_target, drone: Drone) -> Tuple[float, float]:

    target_point = drone.get_next_target()
    # dx = target_point[0] - drone.x
    dy = target_point[1] - drone.y
    pitch = drone.pitch
    pitch_velocity = drone.pitch_velocity

    # Calculate the equilibrium thrust
    Teq = 0.5 * drone.g * drone.mass

    # Calculate epsilon for vertical motion
    epsilon = np.clip(k * dy - b * drone.velocity_y, Teq, -Teq)

    # Check if the drone needs to move horizontally
    if drone.get_next_target()[0] > drone.x:
        theta_target = np.radians(theta_target)  # Convert to radians for positive x direction
    elif drone.get_next_target()[0] < drone.x:
        theta_target = -np.radians(theta_target)  # Convert to radians for negative x direction
    else:
        theta_target = 0  # No pitch needed if x is at target
    
    # Calculate gamma for horizontal motion
    gamma = np.clip(k_theta * (theta_target - pitch) - b_theta * pitch_velocity, Teq, -Teq)

    # Determine T1 and T2 based on the need to move horizontally
    if theta_target == 0:
        T1 = T2 = 0.5 + epsilon 
    else:
        T1 = 0.5 + (gamma + epsilon)
        T2 = 0.5 - (gamma - epsilon)

    # Ensure that the thrust values are within the allowed range
    T1 = max(0, min(T1, 1))
    T2 = max(0, min(T2, 1))
    # print(f"T1:{T1}, T2:{T2},dy:{dy}")
    return T1, T2



class QLearningController(FlightController):
    def __init__(self,
                 state_size=81,
                 action_size=2,
                 learning_rate=0.2,
                 gamma=0.95,
                 epsilon=1):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.85
        self.min_epsilon = 0.01
        self.epochs = 2000
        self.evaluation_interval = 40

        self.q_values = np.zeros((state_size, action_size))

        self.target_index = 0

        self.k = 8.25
        self.b = 0.3
        self.k_theta = 9.5
        self.b_theta = 0.3
        self.theta_target = 7


    def discretize_with_clamp(self, value, min_val, max_val, step):
        # Clamp the value within the specified range
        clamped_value = max(min_val, min(max_val, value))
        # Discretize the clamped value
        return int((clamped_value - min_val) / step)
    
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

        # Discretize each component with clamping
        # discretized_dx = self.discretize_with_clamp(dx, -0.2, 0.2, step_size) + 4
        # discretized_dy = self.discretize_with_clamp(dy, -0.2, 0.2, step_size) + 4
        # discretized_velocity_y = 0 if drone.velocity_y >= 0 else 1
        # discretized_pitch = self.discretize_with_clamp(drone.pitch, -0.5, 0.5, pitch_step)
        # discretized_pitch_velocity = self.discretize_with_clamp(drone.pitch_velocity, -1, 1, pitch_step)

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
        return state_index


    # def generate_action_space(drone):
    #     actions = []
    #     # Load the parameters from the JSON file
    #     # with open('heuristic_controller_1_parameters.json', 'r') as file:
    #     #     best_params_heuristic_1 = json.load(file)
    #     best_param_heuristic_1 = {'ky': 3, 'kx': 2.6, 'abs_pitch_delta': 0.1, 'abs_thrust_delta':0.4}
    #     best_param_heuristic_2 = {'k': 3.75, 'b': 0.3, 'k_theta': 2.25, 'b_theta': 0.3, 'theta_target': 7}
        
    #     ky = best_param_heuristic_1['ky']
    #     kx = best_param_heuristic_1['kx']
    #     abs_pitch_delta = best_param_heuristic_1['abs_pitch_delta']
    #     abs_thrust_delta = best_param_heuristic_1['abs_thrust_delta']

    #     k = best_param_heuristic_2['k']
    #     b = best_param_heuristic_2['b']
    #     k_theta = best_param_heuristic_2['k_theta']
    #     b_theta = best_param_heuristic_2['b_theta']
    #     theta_target = best_param_heuristic_2['theta_target']

    #     # with open('heuristic_controller_2_parameters.json', 'r') as file:
    #     #     best_params_heuristic_2 = json.load(file)

    #     # for param1 in best_params_heuristic_1:
    #     #     for param2 in best_params_heuristic_2:
    #             # ... include other parameters if necessary
    #     thrust_left1, thrust_right1 = heuristic1(ky, kx, abs_pitch_delta, abs_thrust_delta, drone)
    #     thrust_left2, thrust_right2 = heuristic2(k, b, k_theta, b_theta, theta_target, drone)
    #     actions.append((thrust_left1, thrust_right1))
    #     actions.append((thrust_left2, thrust_right2))
    #     return np.array(actions)
    
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
        q_values = self.q_values[state]
        thrust_left, thrust_right = self.discrete_actions(np.argmax(q_values), drone)
        # thrust_left, thrust_right = heuristic2(self.k, self.b, self.k_theta, self.b_theta, self.theta_target, drone)
        return (thrust_left, thrust_right)

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

    def act(self, state):
        # Epsilon-greedy exploration strategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_values[state, :])
        

    def update_q_values(self, reward, next_state, done, time):
        if done:
            self.target_index += 1
            if (self.target_index == 5):
                self.q_values[self.current_state, self.current_action] += self.learning_rate * (reward - self.q_values[self.current_state, self.current_action])
        elif time == self.get_max_simulation_steps() - 1:
            self.q_values[self.current_state, self.current_action] += self.learning_rate * (reward - self.q_values[self.current_state, self.current_action])
        else:
            best_next_action = np.argmax(self.q_values[next_state, :])
            self.q_values[self.current_state, self.current_action] += self.learning_rate * (
                reward + self.gamma * self.q_values[next_state, best_next_action] - self.q_values[self.current_state, self.current_action]
            )

    def step(self, drone, state, action_index):
        # Convert discrete action back to continuous action
        action = self.discrete_actions(action_index, drone)
        # print(action)
        drone.set_thrust(action)                   
        drone.step_simulation(self.get_time_interval())   
        # print(drone.thrust_left, drone.thrust_right) 

        next_state = self.get_state(drone)
        reward = self.get_reward(drone)
        done = drone.has_reached_target_last_update   
        return next_state, reward, done
     

    def evaluate_performance(self):
        """
        Evaluate the performance of the current controller settings.

        Returns:
            float: A performance score, higher is better.
        """
        total_performance = 0
        num_eval_episodes = 10  # Number of episodes to run for evaluation

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

    def train(self):
        # Ensure the Results directory exists
        if not os.path.exists('./Results'):
            os.makedirs('./Results')

        plot_filename = (
            f"./Plots/q-learning/training_plot_"
            f"lr{self.learning_rate}_"
            f"gamma{self.gamma}_"
            f"epsdecay{self.epsilon_decay}_"
            f"epochs{self.epochs}_"
            f"evalint{self.evaluation_interval}.png"
        )
        evaluation_epochs = []
        cumulative_rewards = [] 
        
        for epoch in range(self.epochs):
            drone = self.init_drone()   
            best_performance = float('-inf')            # flight controller init_drone method used here
            
            state = self.get_state(drone)               #Discretize state
            self.current_state = state

            total_reward = 0
            self.target_index = 0

            for time in range(self.get_max_simulation_steps()):
                action_index = self.act(self.current_state)      # Epsilon-Greedy 
                # print(action_index)
                self.current_action = action_index
                # Execute the action in the simulator. Observe the next state and reward
                next_state, reward, done = self.step(drone, state, action_index)
                # print(next_state)
                total_reward += reward

                # Update Q-values
                self.update_q_values(reward, next_state, done, time)

                # Move to the next state
                self.current_state = next_state
                
                if self.target_index == 5:
                    print(f"Done Epoch: {epoch + 1}, Total Reward: {total_reward}, Steps: {time + 1}")
                    break
            
            if epoch % self.evaluation_interval == 0:
                performance = self.evaluate_performance()
                cumulative_rewards.append(performance)  # Store the cumulative reward
                evaluation_epochs.append(epoch + 1)  # Store the epoch number

                if performance >= best_performance:
                    best_performance = performance
                    self.save()
                print(f"Episode {epoch+1}: Cumulative reward: {performance}")

            # Decrease epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
        # Plotting the cumulative rewards at evaluation points
        plt.figure(figsize=(10, 6))
        plt.plot(evaluation_epochs, cumulative_rewards, label='Cumulative Reward')
        plt.xlabel('Evaluation Epochs')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Performance at Evaluation Intervals')

        xticks = [epoch for epoch in evaluation_epochs if epoch % 1000 == 0 or epoch == 1]
        plt.xticks(ticks=xticks, labels=[str(epoch) for epoch in xticks])
        plt.legend()
        
        # Saving the plot with the same filename format
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")

    def save(self):
        filename = (
            f"./Results/q-learning/q_values_"
            f"lr{self.learning_rate}_"
            f"gamma{self.gamma}_"
            f"epsdecay{self.epsilon_decay}_"
            f"epochs{self.epochs}_"
            f"evalint{self.evaluation_interval}.npy"
        )
        np.save(filename, self.q_values) 


    def load(self):
        filename = (
            f"./Results/q-learning/q_values_"
            f"lr{self.learning_rate}_"
            f"gamma{self.gamma}_"
            f"epsdecay{self.epsilon_decay}_"
            f"epochs{self.epochs}_"
            f"evalint{self.evaluation_interval}.npy"
        )
        self.q_values = np.load(filename)
        print(f"Loaded Q-values from {filename}")





