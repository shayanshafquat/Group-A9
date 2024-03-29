import numpy as np
from flight_controller import FlightController
import json
import random



class Heuristic2_RL_tuning(FlightController):
    def __init__(self):
        # Define the constants for the controller
        self.initial_k = 2
        self.initial_b = 0.3
        self.initial_k_theta = 4
        self.initial_b_theta = 0.3
        self.k = self.initial_k  # Proportional constant for altitude control
        self.b = self.initial_b  # Damping constant for altitude control
        self.k_theta = self.initial_k_theta  # Proportional constant for pitch control
        self.b_theta = self.initial_b_theta  # Damping constant for pitch control
        self.max_thrust = 1.0  # Maximum thrust for each propeller
        self.theta_target = 10  # Target pitch angle in degrees     

        # Define ranges and sizes for discretization
        self.min_k = 1.0
        self.max_k = 10.0
        self.min_b = 0.1
        self.max_b = 1.0
        self.min_k_theta = 1.0
        self.max_k_theta = 10.0
        self.min_b_theta = 0.1
        self.max_b_theta = 1.0
        self.k_size = 5
        self.b_size = 5
        self.k_theta_size = 5
        self.b_theta_size = 5

         # Action space
        self.action_changes_k = np.linspace(-1, 1, num=5) 
        self.action_changes_b = np.linspace(-0.15, 0.15, num=5)

        # Initialize Q-table
        self.q_table = np.zeros((self.k_size, self.b_size, self.k_theta_size, self.b_theta_size, 5, 5, 5, 5))
        # self.q_table = np.zeros((self.b_size, self.b_theta_size, 5, 5))
        # self.q_table = np.zeros((self.k_size, self.k_theta_size, 20, 20))

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.discount_factor = 0.999
        self.episodes = 3000
        self.evaluation_interval = 30
        self.reward_method = 0

    def get_max_simulation_steps(self):
            return 3000 # You can alter the amount of steps you want your program to run for here
    
    def get_thrusts(self, drone):

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        # Calculate the equilibrium thrust
        Teq = 0.5 * drone.g * drone.mass

        # Calculate epsilon for vertical motion
        epsilon = np.clip(self.k * dy - self.b * drone.velocity_y, Teq, -Teq)

        # Check if the drone needs to move horizontally
        if drone.get_next_target()[0] > drone.x:
            theta_target = np.radians(self.theta_target)  # Convert to radians for positive x direction
        elif drone.get_next_target()[0] < drone.x:
            theta_target = -np.radians(self.theta_target)  # Convert to radians for negative x direction
        else:
            theta_target = 0  # No pitch needed if x is at target
        
        # Calculate gamma for horizontal motion
        gamma = np.clip(self.k_theta * (theta_target - drone.get_pitch()) - self.b_theta * drone.pitch_velocity, Teq, -Teq)

        # Determine T1 and T2 based on the need to move horizontally
        if theta_target == 0:
            T1 = T2 = 0.5 + epsilon / self.max_thrust
        else:
            T1 = 0.5 + (gamma + epsilon)/self.max_thrust
            T2 = 0.5 - (gamma - epsilon)/self.max_thrust

        # Ensure that the thrust values are within the allowed range
        T1 = max(0, min(T1, 1))
        T2 = max(0, min(T2, 1))
        # print(f"T1:{T1}, T2:{T2},dy:{dy}")
        return T1, T2
    
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
    
    def get_q_table_index(self):
        # Calculate the discretization steps for each parameter
        k_step = (self.max_k - self.min_k) / (self.k_size - 1)
        b_step = (self.max_b - self.min_b) / (self.b_size - 1)
        k_theta_step = (self.max_k_theta - self.min_k_theta) / (self.k_theta_size - 1)
        b_theta_step = (self.max_b_theta - self.min_b_theta) / (self.b_theta_size - 1)

        # Discretize each parameter
        k_index = min(int((self.k - self.min_k) / k_step), self.k_size - 1)
        b_index = min(int((self.b - self.min_b) / b_step), self.b_size - 1)
        k_theta_index = min(int((self.k_theta - self.min_k_theta) / k_theta_step), self.k_theta_size - 1)
        b_theta_index = min(int((self.b_theta - self.min_b_theta) / b_theta_step), self.b_theta_size - 1)

        return (k_index, b_index, k_theta_index, b_theta_index)
        # return (b_index, b_theta_index)
        # return (k_index, k_theta_index)
    
    def adjust_parameters(self, action):
        action_k = self.action_changes_k[action[0]]
        action_k_theta = self.action_changes_k[action[2]]

        action_b = self.action_changes_b[action[1]]
        action_b_theta = self.action_changes_b[action[3]]

        # Adjust all four parameters based on the action
        self.k += action_k
        self.b += action_b
        self.k_theta += action_k_theta
        self.b_theta += action_b_theta

        # Ensure parameters stay within their valid range
        # print(self.k, self.k_theta)
        self.k = np.clip(self.k, self.min_k, self.max_k)
        self.b = np.clip(self.b, self.min_b, self.max_b)
        self.k_theta = np.clip(self.k_theta, self.min_k_theta, self.max_k_theta)
        self.b_theta = np.clip(self.b_theta, self.min_b_theta, self.max_b_theta)

    def random_action(self):
        # Generate a random action for each parameter
        action_k = random.choice(self.action_changes_k)
        action_b = random.choice(self.action_changes_b)
        action_k_theta = random.choice(self.action_changes_k)
        action_b_theta = random.choice(self.action_changes_b)

        # Map the float action to its corresponding index
        action_k_index = np.where(self.action_changes_k == action_k)[0][0]
        action_b_index = np.where(self.action_changes_b == action_b)[0][0]
        action_k_theta_index = np.where(self.action_changes_k == action_k_theta)[0][0]
        action_b_theta_index = np.where(self.action_changes_b == action_b_theta)[0][0]

        return (action_k_index, action_b_index, action_k_theta_index, action_b_theta_index)
        # return (action_b_index, action_b_theta_index)
        # return (action_k_index, action_k_theta_index)

    def reset_parameters(self):
        """Reset the ky and kx parameters to their initial values."""
        self.k = self.initial_k
        self.b = self.initial_b
        self.k_theta = self.initial_k_theta
        self.b_theta = self.initial_b_theta

    def choose_action(self):
        if np.random.rand() <= self.epsilon:
            return self.random_action()
        else:
            q_table_index = self.get_q_table_index()
            # Find the action with the maximum Q-value
            action_index = np.unravel_index(np.argmax(self.q_table[q_table_index]), self.q_table[q_table_index].shape)
            return action_index
        
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

            for i in range(self.get_max_simulation_steps()):
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
                # if(reward>50):
                #     print(f"{t}:{reward}")

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
                if performance >= best_performance:
                    best_performance = round(performance,3)
                    best_avg_steps = round(steps_count,3)
                    best_avg_thrust = round(avg_thrust,3)
                    best_avg_dist = round(avg_dist,3)
        #             print(best_performance)
                    current_best_parameters = {
                        'episode': episode + 1,
                        'performance': best_performance,
                        'best_avg_steps': best_avg_steps,
                        'avg_thrusts': best_avg_thrust,
                        'avg_dist': avg_dist,
                        'parameters': {
                            'k': self.k,
                            'b': self.b,
                            'k_theta': self.k_theta,
                            'b_theta': self.b_theta
                        }
                    }
                    best_parameters_list.append(current_best_parameters)
                    print(f"Parameter tuned: {current_best_parameters}")
                print(f"Episode {episode + 1}: Mean Cumulative Reward / Step: {performance:.2f}, Steps Taken: {steps_count:.2f}, Average Thrust:{avg_thrust:.2f}, Average Distance: {avg_dist:.3f}.")

        return best_performance, best_avg_steps, best_parameters_list, best_avg_thrust, best_avg_dist
        # with open('heuristic_controller_2.1_parameters.json', 'w') as file:
        #     json.dump(best_parameters_list, file, indent=4)

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
        with open(f'./Results/tuning/all_parameters_list_heuristic_2_{self.reward_method}.json', 'w') as file:
            json.dump(all_parameters, file, indent=4)

        # Save summary performance
        with open(f'./Results/tuning/summary_performance_heuristic_2_{self.reward_method}.json', 'w') as file:
            json.dump(summary_performance, file, indent=4)

    def save(self):
        # Save parameters to a file
        parameters = {
            'k': self.initial_k,
            'b': self.initial_b,
            'k_theta': self.initial_k_theta,
            'b_theta': self.initial_b_theta
        }
        with open('custom_controller_parameters.json', 'w') as file:
            json.dump(parameters, file, indent=4)


    def load(self):
        # Loading is not applicable in this heuristic controller
        try:
            with open(f'./Results/tuning/all_parameters_list_heuristic_2_{self.reward_method}.json', 'r') as file:
                data_h2 = json.load(file)
            sorted_h2_data = sorted(data_h2, key=lambda x: x['performance'], reverse=True)[0]
            params = sorted_h2_data['parameters']
            self.k = params['k']
            self.b = params['b']
            self.k_theta = params['k_theta']
            self.b_theta = params['b_theta']
        except:
            print("Could not load parameters, sticking with default parameters.")


