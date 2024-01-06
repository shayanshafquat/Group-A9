import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import random

class HeuristicController(FlightController):
    def __init__(self):
        super().__init__()
        self.initial_ky = 1.0
        self.initial_kx = 0.5
        self.ky = self.initial_ky
        self.kx = self.initial_kx
        self.min_ky = 0.1  # Example minimum value for ky
        self.max_ky = 2.0  # Example maximum value for ky
        self.min_kx = 0.05  # Example minimum value for kx
        self.max_kx = 2.0  # Example maximum value for kx

        self.abs_pitch_delta = 0.1
        self.abs_thrust_delta = 0.3
        """Creates a heuristic flight controller with some specified parameters

        """
        self.ky_size = 10
        self.kx_size = 10

        self.action_changes = np.linspace(-1, 1, num=20) 

        self.q_table = np.zeros((self.ky_size, self.kx_size, 20, 20))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.episodes = 1000
        self.evaluation_interval = 20
        self.performance_threshold = 0



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
    
    # def get_state(self, drone):
    #     # Define the discretization steps for each component
    #     position_step = 1.0  # e.g., 1 unit of space
    #     velocity_step = 0.5  # e.g., 0.5 units of velocity
    #     pitch_step = np.pi / 18  # e.g., 10 degrees

    #     # Discretize each component
    #     discretized_x = int(drone.x / position_step)
    #     discretized_y = int(drone.y / position_step)
    #     discretized_velocity_x = int(drone.velocity_x / velocity_step)
    #     discretized_velocity_y = int(drone.velocity_y / velocity_step)
    #     discretized_pitch = int(drone.pitch / pitch_step)

    #     # Combine into a single state
    #     state = (discretized_x, discretized_y, discretized_velocity_x, discretized_velocity_y, discretized_pitch)
    #     return state
    
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
    
    def choose_action(self):
        if np.random.rand() <= self.epsilon:
            return self.random_action()
        else:
            q_table_index = self.get_q_table_index()
            # Find the action with the maximum Q-value
            action_index = np.unravel_index(np.argmax(self.q_table[q_table_index]), self.q_table[q_table_index].shape)
            return action_index

    def random_action(self):
        # Generate a random action
        action_ky = random.choice(self.action_changes)
        action_kx = random.choice(self.action_changes)
        # Map the float action to its corresponding index
        action_ky_index = np.where(self.action_changes == action_ky)[0][0]
        action_kx_index = np.where(self.action_changes == action_kx)[0][0]
        return (action_ky_index, action_kx_index)
    
    # def choose_action(self):
    #     if np.random.rand() <= self.epsilon:
    #         return self.random_action()
    #     else:
    #         q_table_index = self.get_q_table_index()
    #         return np.unravel_index(np.argmax(self.q_table[q_table_index]), self.q_table[q_table_index].shape)

    # def random_action(self):
    #     return (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
    def get_q_table_index(self):
        # Calculate the discretization steps for each parameter
        ky_step = (self.max_ky - self.min_ky) / (self.ky_size - 1)
        kx_step = (self.max_kx - self.min_kx) / (self.kx_size - 1)

        # Discretize each parameter
        ky_index = int((self.ky - self.min_ky) / ky_step)
        kx_index = int((self.kx - self.min_kx) / kx_step)

        # Ensure indices are within bounds
        ky_index = min(ky_index, self.ky_size - 1)
        kx_index = min(kx_index, self.kx_size - 1)
        return (ky_index, kx_index)
    
    def adjust_parameters(self, action):
        # Map the action indices back to float values
        action_ky = self.action_changes[action[0]]
        action_kx = self.action_changes[action[1]]

        # Adjust the parameters based on the action
        self.ky += action_ky
        self.kx += action_kx

        # Ensure parameters stay within their valid range
        self.ky = np.clip(self.ky, self.min_ky, self.max_ky)
        self.kx = np.clip(self.kx, self.min_kx, self.max_kx)

    # def adjust_parameters(self, action):
    #     # Calculate the discretization steps for each parameter
    #     ky_step = (self.max_ky - self.min_ky) / self.ky_size
    #     kx_step = (self.max_kx - self.min_kx) / self.kx_size

    #     # Adjust the parameters based on the action
    #     self.ky += action[0] * ky_step
    #     self.kx += action[1] * kx_step

    #     # Ensure parameters stay within their valid range
    #     self.ky = np.clip(self.ky, self.min_ky, self.max_ky)
    #     self.kx = np.clip(self.kx, self.min_kx, self.max_kx)

    
    def train(self):
        best_performance = float('-inf')
        best_parameters = (self.ky, self.kx)

        for episode in range(self.episodes):
            # Reset environment and parameters
            self.reset_parameters()
            drone = self.init_drone()
            total_reward = 0

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

                # Calculate reward and update state
                reward = self.get_reward(drone)
                total_reward += reward

                # Update Q-table
                q_table_index = current_state_index + tuple(action)
                old_value = self.q_table[q_table_index]
                next_max = np.max(self.q_table[new_state_index])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
                self.q_table[q_table_index] = new_value

                # Decrease epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            # Evaluate and potentially save parameters
            if episode % self.evaluation_interval == 0:
                performance = self.evaluate_performance()  # Implement this method
                if performance > best_performance:
                    best_performance = performance
                    best_parameters = (self.ky, self.kx)
                    print(f"best performance:{best_performance}Best parameters:{best_parameters}")
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        
        if best_performance >= self.performance_threshold:
            self.ky, self.kx = best_parameters
            self.save()

    def evaluate_performance(self):
        """
        Evaluate the performance of the current controller settings.

        Returns:
            float: A performance score, higher is better.
        """
        total_performance = 0
        num_eval_episodes = 5  # Number of episodes to run for evaluation

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
        