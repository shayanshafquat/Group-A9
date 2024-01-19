from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import math


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
                 state_size=32*3,
                 action_size=4,
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
        self.epochs = 1000
        self.evaluation_interval = 40

        self.q_values = np.zeros((state_size, action_size))

        self.target_index = 0
        self.reward_method = 0
    
    def get_state(self, drone: Drone):
        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        distance = math.sqrt(dx**2 + dy**2)
        velocity_y = abs(drone.velocity_y)
        pitch = drone.pitch
        pitch_velocity = abs(drone.pitch_velocity)

        # Automatically select bin sizes based on self.state_size
        if self.state_size == 96:
            dist_bin = [0.1, 0.5]
            vel_bin = [0.1]
            pitch_vel_bin = [0.1]
        elif self.state_size == 384:
            dist_bin = [0.1, 0.3, 0.7]
            vel_bin = [0.1, 0.25]
            pitch_vel_bin = [0.1, 0.3, 0.6]
        else:
            raise ValueError("Unsupported state size")

        # Discretization logic remains the same
        discretized_velocity_y = next((i for i, edge in enumerate(vel_bin) if velocity_y <= edge), len(vel_bin))
        discretized_pitch_velocity = next((i for i, edge in enumerate(pitch_vel_bin) if pitch_velocity <= edge), len(pitch_vel_bin))
        discretized_dx = 0 if dx >= 0 else 1
        discretized_dy = 0 if dy >= 0 else 1
        discretized_pitch = 0 if pitch >= 0 else 1
        discretized_dist = next((i for i, edge in enumerate(dist_bin) if distance <= edge), len(dist_bin))

        # Calculate the state index
        state_index = (discretized_dx +
                    discretized_dy * 2 +
                    discretized_velocity_y * 4 +
                    discretized_pitch * 4 * len(vel_bin) +
                    discretized_pitch_velocity * 4 * len(vel_bin) * 2 +
                    discretized_dist * 4 * len(vel_bin) * 2 * len(pitch_vel_bin))

        if state_index < 0:
            print("state index is negative")
        return state_index


    def discrete_actions(self, action_index, drone):
        # Assume self.action_space_size is set to the size of the action space (4 or 6)
        action_space_size = self.action_size

        # Load parameters from JSON files
        with open(f'./Results/tuning/all_parameters_list_heuristic_{self.reward_method}.json', 'r') as file:
            data_h1 = json.load(file)
        sorted_h1_data = sorted(data_h1, key=lambda x: x['performance'], reverse=True)

        with open(f'./Results/tuning/all_parameters_list_heuristic_2_{self.reward_method}.json', 'r') as file:
            data_h2 = json.load(file)
        sorted_h2_data = sorted(data_h2, key=lambda x: x['performance'], reverse=True)

        # Determine the number of top parameters to use based on action space size
        num_top_params = action_space_size // 2
        top_h1_params = sorted_h1_data[:num_top_params]
        top_h2_params = sorted_h2_data[:num_top_params]

        # Check the action_index and set parameters accordingly
        if action_index < num_top_params:  # Heuristic 1 parameters
            params = top_h1_params[action_index]['parameters']
            thrust_left, thrust_right = heuristic1(params['ky'],
                                                params['kx'],
                                                params['abs_pitch_delta'],
                                                params['abs_thrust_delta'],
                                                drone)
        elif action_index < action_space_size:  # Heuristic 2 parameters
            params = top_h2_params[action_index - num_top_params]['parameters']
            theta_target = 10
            thrust_left, thrust_right = heuristic2(params['k'],
                                                params['b'],
                                                params['k_theta'],
                                                params['b_theta'],
                                                theta_target,
                                                drone)
        else:
            raise ValueError("Invalid action_index")

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

        if self.reward_method == 0:
            reward = (
                100.0 * drone.has_reached_target_last_update -
                10.0 * distance_to_target -
                1 * (drone.thrust_left + drone.thrust_right) -
                0.1
            )
        else: 
            reward = (
                100.0 * drone.has_reached_target_last_update -
                10.0 * distance_to_target -
                0.1 * (drone.thrust_left + drone.thrust_right) -
                1
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
            if (self.target_index == 4):
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
        total_steps = 0
        num_eval_episodes = 3  # Number of episodes to run for evaluation
        target_index = 0

        for _ in range(num_eval_episodes):
            drone = self.init_drone()  # Initialize the drone for each evaluation episode
            episode_performance = 0
            episode_steps = 0

            for _ in range(self.get_max_simulation_steps()):
                # Get the thrusts based on current ky and kx values
                thrusts = self.get_thrusts(drone)
                drone.set_thrust(thrusts)

                # Update the drone's state
                drone.step_simulation(self.get_time_interval())

                if drone.has_reached_target_last_update:
                    target_index += 1

                # Calculate the reward for the current step
                reward = self.get_reward(drone)
                episode_performance += reward
                episode_steps += 1

                if target_index == 4:
                    break

            # Average performance over the episode
            total_performance += episode_performance / self.get_max_simulation_steps()
            total_steps += episode_steps

        # Average performance over all evaluation episodes
        return total_performance / num_eval_episodes, total_steps / num_eval_episodes

    def run_training_sequence(self):
        # Ensure the Results directory exists
        evaluation_epochs = []
        cumulative_rewards = [] 
        avg_steps = []

        best_avg_steps = float('-inf') 
        best_performance = float('-inf') 
        
        for epoch in range(self.epochs):
            drone = self.init_drone()              # flight controller init_drone method used here
            
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
                
                if self.target_index == 4:
                    # print(f"Done Epoch: {epoch + 1}, Cumulative reward / step: {total_reward/self.get_max_simulation_steps()}, Steps: {time + 1}")
                    break
            
            if epoch % self.evaluation_interval == 0 or epoch == self.epochs - 1:
                performance, steps_count = self.evaluate_performance()
                cumulative_rewards.append(performance)  # Store the cumulative reward
                evaluation_epochs.append(epoch + 1)  # Store the epoch number
                avg_steps.append(steps_count)

                print(f"Episode {epoch+1}: Cumulative reward / step: {performance}, Steps Taken: {steps_count}")

                if performance >= best_performance:
                    best_performance = performance
                    best_avg_steps = steps_count
                    self.save()
                    print(f"Improved performance: {performance}, #Steps Taken: {steps_count}")
                

            # Decrease epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
        print("Training finished")
        return cumulative_rewards, evaluation_epochs, avg_steps, best_performance, best_avg_steps
    
    def train(self):
        learning_rates = [0.05, 0.01]
        discount_factors = [0.85, 0.95]
        epsilon_decays = [0.995, 0.99]

        summary_performance = []

        total_runs = len(learning_rates) * len(discount_factors) * len(epsilon_decays)
        current_run = 0

        for lr in learning_rates:
            for df in discount_factors:
                for ed in epsilon_decays:
                    self.__init__()
                    current_run += 1
                    print(f'Running training {current_run}/{total_runs} with learning rate={lr}, discount factor={df}, epsilon decay={ed}. state size={self.state_size}, Action size={self.action_size}')

                    self.learning_rate = lr
                    self.discount_factor = df
                    self.epsilon_decay = ed

                    cumulative_rewards, evaluation_epochs, avg_steps_epochs, best_performance, steps_taken = self.run_training_sequence()

                    # Combine cumulative_rewards and evaluation_epochs and save as a numpy array
                    combined_data = np.column_stack((evaluation_epochs, cumulative_rewards, avg_steps_epochs))
                    np.save(f'./Results/q-learning/lr{lr}_df{df}_ed{ed}_{self.state_size}_{self.action_size}_{self.reward_method}.npy', combined_data)

                    # Append best performance data to the list
                    summary_performance.append({
                        'learning_rate': lr,
                        'discount_factor': df,
                        'epsilon_decay': ed,
                        'cumulative_rewards_per_step': best_performance,
                        'avg_steps_count': steps_taken
                    })

        # Save summary_performance as a CSV file
        df_summary = pd.DataFrame(summary_performance)
        df_summary.to_csv(f'./Results/q-learning/summary_performance_{self.state_size}_{self.action_size}_{self.reward_method}.csv', index=False)
        print("Saved summary and Cumulative reward results")

    def save(self):
        filename = (
            f"./Results/q-learning/q_values_"
            f"lr{self.learning_rate}_"
            f"gamma{self.gamma}_"
            f"epsdecay{self.epsilon_decay}_"
            f"epochs{self.epochs}_"
            f"evalint{self.evaluation_interval}_"
            f"reward_{self.reward_method}.npy"
        )
        np.save(filename, self.q_values) 


    def load(self):
        filename = (
            f"./Results/q-learning/q_values_"
            f"lr{self.learning_rate}_"
            f"gamma{self.gamma}_"
            f"epsdecay{self.epsilon_decay}_"
            f"epochs{self.epochs}_"
            f"evalint{self.evaluation_interval}_"
            f"reward_{self.reward_method}.npy"
        )
        self.q_values = np.load(filename)
        print(f"Loaded Q-values from {filename}")





