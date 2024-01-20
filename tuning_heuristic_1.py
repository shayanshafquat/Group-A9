import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import random
import json


class Heuristic_RL_tuning(FlightController):
    def __init__(self):
        super().__init__()
        self.initial_ky = 1
        self.initial_kx = 0.5
        self.initial_abs_pitch_delta = 0.1
        self.initial_abs_thrust_delta = 0.3
        self.ky = self.initial_ky
        self.kx = self.initial_kx
        self.min_ky = 1  # Example minimum value for ky
        self.max_ky = 3.0  # Example maximum value for ky
        self.min_kx = 0.5  # Example minimum value for kx
        self.max_kx = 3.0  # Example maximum value for kx

        self.abs_pitch_delta = self.initial_abs_pitch_delta
        self.abs_thrust_delta= self.initial_abs_thrust_delta
        """Creates a heuristic flight controller with some specified parameters

        """
        self.ky_size = 5
        self.kx_size = 5
        self.abs_pitch_delta_size = 5
        self.abs_thrust_delta_size = 5

        self.min_abs_pitch_delta = 0.05
        self.max_abs_pitch_delta = 0.5
        self.min_abs_thrust_delta = 0.1
        self.max_abs_thrust_delta = 0.5

        self.action_changes_k = np.linspace(-0.3, 0.3, num=5) 
        self.action_changes_delta = np.linspace(-0.05, 0.05, num=5)

        self.q_table = np.zeros((self.ky_size, self.kx_size, self.abs_pitch_delta_size, self.abs_thrust_delta_size, 5, 5, 5, 5))
        # self.q_table = np.zeros((self.abs_pitch_delta_size, self.abs_thrust_delta_size, 10, 10))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.episodes = 3000
        self.evaluation_interval = 30
        self.reward_method = 1 # or 1


    def get_max_simulation_steps(self):
            return 3000 # You can alter the amount of steps you want your program to run for here


    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Takes a given drone object, containing information about its current state
        and calculates a pair of thrust values for the left and right propellers.

        Args:
            drone (Drone): The drone object containing the information about the drones state.

        Returns:
            Tuple[float, float]: A pair of floating point values which respectively represent the thrust of the left and right propellers, must be between 0 and 1 inclusive.
        """

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch-drone.pitch

        thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
        thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)
    
    def reset_parameters(self):
        """Reset the ky and kx parameters to their initial values."""
        self.ky = self.initial_ky
        self.kx = self.initial_kx
        self.abs_pitch_delta = self.initial_abs_pitch_delta
        self.abs_thrust_delta = self.initial_abs_thrust_delta
    
    def get_reward(self, drone):
        # Efficient computation of the reward using numpy operations
        target_point = drone.get_next_target()
        self.distance_to_target = np.hypot(drone.x - target_point[0], drone.y - target_point[1])

        if self.reward_method == 0:
            reward = (
                100.0 * drone.has_reached_target_last_update -
                10.0 * self.distance_to_target -
                1 * (drone.thrust_left + drone.thrust_right) -
                0.1
            )
        else: 
            reward = (
                100.0 * drone.has_reached_target_last_update -
                10.0 * self.distance_to_target -
                0.1 * (drone.thrust_left + drone.thrust_right) -
                1
            )
        return reward
    
    def choose_action(self):
        if np.random.rand() <= self.epsilon:
            return self.random_action()
        else:
            q_table_index = self.get_q_table_index()
            # Find the action with the maximum Q-value
            action_index = np.unravel_index(np.argmax(self.q_table[q_table_index]), self.q_table[q_table_index].shape)
            return action_index

    # def random_action(self):
    #     # Generate a random action
    #     action_ky = random.choice(self.action_changes)
    #     action_kx = random.choice(self.action_changes)
    #     # Map the float action to its corresponding index
    #     action_ky_index = np.where(self.action_changes == action_ky)[0][0]
    #     action_kx_index = np.where(self.action_changes == action_kx)[0][0]
    #     return (action_ky_index, action_kx_index)
    
    def random_action(self):
        # Generate a random action for each parameter
        action_ky = random.choice(self.action_changes_k)
        action_kx = random.choice(self.action_changes_k)
        action_abs_pitch_delta = random.choice(self.action_changes_delta)
        action_abs_thrust_delta = random.choice(self.action_changes_delta)
        # Map the float action to its corresponding index
        action_ky_index = np.where(self.action_changes_k == action_ky)[0][0]
        action_kx_index = np.where(self.action_changes_k == action_kx)[0][0]
        action_abs_pitch_delta_index = np.where(self.action_changes_delta == action_abs_pitch_delta)[0][0]
        action_abs_thrust_delta_index = np.where(self.action_changes_delta == action_abs_thrust_delta)[0][0]
        return (action_ky_index, action_kx_index, action_abs_pitch_delta_index, action_abs_thrust_delta_index)
    

    # def get_q_table_index(self):
    #     # Calculate the discretization steps for each parameter
    #     ky_step = (self.max_ky - self.min_ky) / (self.ky_size - 1)
    #     kx_step = (self.max_kx - self.min_kx) / (self.kx_size - 1)

    #     # Discretize each parameter
    #     ky_index = int((self.ky - self.min_ky) / ky_step)
    #     kx_index = int((self.kx - self.min_kx) / kx_step)

    #     # Ensure indices are within bounds
    #     ky_index = min(ky_index, self.ky_size - 1)
    #     kx_index = min(kx_index, self.kx_size - 1)
    #     return (ky_index, kx_index)
    
    def get_q_table_index(self):
        # Calculate the discretization steps for each parameter
        ky_step = (self.max_ky - self.min_ky) / (self.ky_size - 1)
        kx_step = (self.max_kx - self.min_kx) / (self.kx_size - 1)
        abs_pitch_delta_step = (self.max_abs_pitch_delta - self.min_abs_pitch_delta) / (self.abs_pitch_delta_size - 1)
        abs_thrust_delta_step = (self.max_abs_thrust_delta - self.min_abs_thrust_delta) / (self.abs_thrust_delta_size - 1)

        # Discretize each parameter
        ky_index = min(int((self.ky - self.min_ky) / ky_step), self.ky_size - 1)
        kx_index = min(int((self.kx - self.min_kx) / kx_step), self.kx_size - 1)
        abs_pitch_delta_index = min(int((self.abs_pitch_delta - self.min_abs_pitch_delta) / abs_pitch_delta_step), self.abs_pitch_delta_size - 1)
        abs_thrust_delta_index = min(int((self.abs_thrust_delta - self.min_abs_thrust_delta) / abs_thrust_delta_step), self.abs_thrust_delta_size - 1)

        return (ky_index, kx_index, abs_pitch_delta_index, abs_thrust_delta_index)

    
    # def adjust_parameters(self, action):
    #     # Map the action indices back to float values
    #     action_ky = self.action_changes[action[0]]
    #     action_kx = self.action_changes[action[1]]

    #     # Adjust the parameters based on the action
    #     self.ky += action_ky
    #     self.kx += action_kx

    #     # Ensure parameters stay within their valid range
    #     self.ky = np.clip(self.ky, self.min_ky, self.max_ky)
    #     self.kx = np.clip(self.kx, self.min_kx, self.max_kx)

    def adjust_parameters(self, action):
        # Map the action indices back to float values
        action_ky = self.action_changes_k[action[0]]
        action_kx = self.action_changes_k[action[1]]
        action_abs_pitch_delta = self.action_changes_delta[action[2]]
        action_abs_thrust_delta = self.action_changes_delta[action[3]]

        # Adjust the parameters based on the action
        self.ky += action_ky
        self.kx += action_kx
        self.abs_pitch_delta += action_abs_pitch_delta
        self.abs_thrust_delta += action_abs_thrust_delta

        # Ensure parameters stay within their valid range
        self.ky = np.clip(self.ky, self.min_ky, self.max_ky)
        self.kx = np.clip(self.kx, self.min_kx, self.max_kx)
        self.abs_pitch_delta = np.clip(self.abs_pitch_delta, self.min_abs_pitch_delta, self.max_abs_pitch_delta)
        self.abs_thrust_delta = np.clip(self.abs_thrust_delta, self.min_abs_thrust_delta, self.max_abs_thrust_delta)

    def evaluate_performance(self):
        """
        Evaluate the performance of the current controller settings.

        Returns:
            float: A performance score, higher is better.
        """
        total_performance = 0
        total_steps = 0
        num_eval_episodes = 5  # Number of episodes to run for evaluation
        target_index = 0
        total_avg_thrusts = 0
        total_avg_dist_to_target = 0

        for _ in range(num_eval_episodes):
            drone = self.init_drone()  # Initialize the drone for each evaluation episode
            episode_performance = 0
            episode_steps = 0
            episode_avg_sum_thrust = 0
            episode_dist_to_target = 0

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
                episode_avg_sum_thrust += np.mean(thrusts)
                episode_dist_to_target += self.distance_to_target

                if target_index == 4:
                    break

            # Average performance over the episode
            total_performance += episode_performance / episode_steps
            total_steps += episode_steps
            total_avg_thrusts += episode_avg_sum_thrust / episode_steps
            total_avg_dist_to_target += episode_dist_to_target / episode_steps

        # Average performance over all evaluation episodes
        return total_performance / num_eval_episodes, total_steps / num_eval_episodes, total_avg_thrusts / num_eval_episodes, total_avg_dist_to_target / num_eval_episodes
    
    def run_training_sequence(self):
        best_performance = float('-inf')
        best_avg_steps = float('-inf')
        best_avg_thrust = float('-inf')
        best_avg_dist = float('-inf')

        best_performance, best_avg_steps, avg_thrust, avg_dist = self.evaluate_performance()
        print(f"Initial Performance:{best_performance:.2f}, Initial Steps: {best_avg_steps:.2f}, Initial avg thrust:{avg_thrust:.2f}, Initial avg distance:{avg_dist:.2f} ")

        best_parameters_list = []  # List to store the best parameters over time
        evaluation_epochs = []
        cumulative_rewards = [] 

        for episode in range(self.episodes):
            # Reset environment and parameters
            self.reset_parameters()
            drone = self.init_drone()
            total_reward = 0
            target = 0

            for t in range(self.get_max_simulation_steps()):
                current_state_index = self.get_q_table_index()
                # print(current_state_index)
                action = self.choose_action()
                # print(action)
                # print(self.action_changes)
                self.adjust_parameters(action)
                # print(self.ky, self.kx)
                new_state_index = self.get_q_table_index()
                # print(new_state_index)

                # Run simulation step
                drone.set_thrust(self.get_thrusts(drone))
                drone.step_simulation(self.get_time_interval())

                if drone.has_reached_target_last_update:
                    target += 1

                # Calculate reward and update state
                reward = self.get_reward(drone)
                total_reward += reward

                # Update Q-table
                q_table_index = current_state_index + tuple(action)
                old_value = self.q_table[q_table_index]
                next_max = np.max(self.q_table[new_state_index])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
                self.q_table[q_table_index] = new_value

                if target == 4 or t == self.get_max_simulation_steps() - 1:
                    new_value = (1 - self.learning_rate) * old_value + self.learning_rate * reward
                    self.q_table[q_table_index] = new_value
                    break

            # Decrease epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            # Evaluate and potentially save parameters
            if episode % self.evaluation_interval == 0:
                performance, steps_count, avg_thrust, avg_dist = self.evaluate_performance()  # Implement this method
                cumulative_rewards.append(performance)  # Store the cumulative reward
                evaluation_epochs.append(episode + 1)  # Store the epoch number

                if performance >= best_performance:
                    best_performance = round(performance,3)
                    best_avg_steps = round(steps_count,3)
                    best_avg_thrust = round(avg_thrust,3)
                    best_avg_dist = round(avg_dist,3)

                    current_best_parameters = {
                        'episode': episode + 1,
                        'performance': best_performance,
                        'best_avg_steps': best_avg_steps,
                        'avg_thrusts': best_avg_thrust,
                        'avg_dist': avg_dist,
                        'parameters': {
                            'ky': self.ky,
                            'kx': self.kx,
                            'abs_pitch_delta': self.abs_pitch_delta,
                            'abs_thrust_delta': self.abs_thrust_delta
                        }
                    }
                    best_parameters_list.append(current_best_parameters)
                    print(f"Parameter tuned: {current_best_parameters}")
                print(f"Episode {episode + 1}: Cumulative Reward / Step: {performance:.2f}, Steps Taken: {steps_count:.2f}, Average Thrust: {avg_thrust:.2f}, Average Distance: {avg_dist:.3f}.")
        return best_performance, best_avg_steps, best_parameters_list, best_avg_thrust, best_avg_dist

    def train(self):
        learning_rates = [0.05, 0.1]
        discount_factors = [0.95, 0.85]
        epsilon_decays = [0.9]

        all_parameters = []
        summary_performance = []

        total_runs = len(learning_rates) * len(discount_factors) * len(epsilon_decays)
        current_run = 0

        for lr in learning_rates:
            for df in discount_factors:
                for ed in epsilon_decays:
                    self.__init__()
                    current_run += 1
                    print(f'Running training {current_run}/{total_runs} with learning rate={lr}, discount factor={df}, epsilon decay={ed}, Reward method: {self.reward_method}')

                    self.learning_rate = lr
                    self.discount_factor = df
                    self.epsilon_decay = ed

                    cumulative_reward, avg_steps_count, parameters_list, avg_thrust, avg_dist = self.run_training_sequence()
                    
                    # Add hyperparameters to each entry in best_parameters_list
                    for params in parameters_list:
                        params['hyperparameters'] = {
                            'learning_rate': lr,
                            'discount_factor': df,
                            'epsilon_decay': ed
                        }
                        all_parameters.append(params)

                    summary_performance.append({
                        'learning_rate': lr,
                        'discount_factor': df,
                        'epsilon_decay': ed,
                        'cumulative_reward_per_steps': cumulative_reward,
                        'avg_steps_count': avg_steps_count,
                        'avg_thrust': avg_thrust,
                        'avg_distance': avg_dist
                    })

        # Save all best parameters
        with open(f'./Results/tuning/all_parameters_list_heuristic_{self.reward_method}.json', 'w') as file:
            json.dump(all_parameters, file, indent=4)

        # Save summary performance
        with open(f'./Results/tuning/summary_performance_heuristic_{self.reward_method}.json', 'w') as file:
            json.dump(summary_performance, file, indent=4)

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('heuristic_controller_parameters.npy')
            self.ky = parameter_array[0]
            self.kx = parameter_array[1]
            self.abs_pitch_delta = parameter_array[2]
            self.abs_thrust_delta = parameter_array[3]
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.ky, self.kx, self.abs_pitch_delta, self.abs_thrust_delta])
        np.save('heuristic_controller_parameters.npy', parameter_array)
        