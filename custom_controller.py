from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    # Original code
    # def __init__(self):
        #pass
    #def train(self):
       # pass    
    #def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        #return (0.5, 0.5) # Replace this with your custom algorithm
    #def load(self):
        #pass
    #def save(self):
        #pass


    def __init__(self, states, actions, learn_rate=0.1, discount=0.9, p_exploration=0.1):
        self.states = states
        self.actions = actions
        self.learn_rate = learn_rate
        self.discount = discount
        self.p_exploration = p_exploration

        # Initialize Q-values to zeros
        self.q_values = np.zeros((states, actions))

        # Initialize state and action variables
        self.current_state = None
        self.current_action = None

    def select_action(self, state):
        # Epsilon-greedy exploration strategy
        if np.random.rand() < self.p_exploration:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.q_values[state, :])

    def update_q_values(self, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_values[next_state, :])
        self.q_values[self.current_state, self.current_action] += self.learn_rate * (
                reward + self.discount * self.q_values[next_state, best_next_action] - self.q_values[self.current_state, self.current_action]
        )

    # Training using Q-learning algorithm for specified number of episodes
    def train(self, episodes):
        for episode in range(episodes):
            # Reset environment and get initial state
            self.current_state = initial_state  # Replace with your logic to get the initial state

            while not self.target_reached(self.current_state):  # Target condition
                # Select an action
                self.current_action = self.select_action(self.current_state)

                # Take the selected action and observe the new state and reward
                next_state, reward = take_action(self.current_state, self.current_action)  # Implement this function based on your dynamics

                # Update Q-values
                self.update_q_values(reward, next_state)

                # Move to the next state
                self.current_state = next_state

    #def take_action(self, current_state, action):
        # Placeholder for your drone dynamics
        # Update the state based on the selected action and return the new state and a reward
        # This is where you simulate the effect of the drone taking the selected action
        # Replace the following line with your actual logic
        #new_state = current_state
        #reward = 0  # Replace with the appropriate reward calculation
        #return new_state, reward

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        # Assuming the state representation is a tuple of relevant drone information
        current_state = (drone.position_x, drone.position_y, drone.velocity_x, drone.velocity_y, drone.pitch_angle, drone.angular_pitch_velocity)

        # Store the current state for Q-learning update
        self.current_state = current_state

        # Select action using Q-learning
        action = self.select_action(current_state)

        # Assuming action is a value between 0 and num_actions-1
        thrust_value = (action / (self.num_actions - 1))  # Scaling to the range [0, 1]

        # Return the same thrust value for both propellers
        return thrust_value, thrust_value
    
    def target_reached(self,current_state):

            # Define your target criteria based on the problem
            target_x, target_y = get_target_coordinates()  # Replace with your target logic

            # Check if the drone has reached the target based on some criteria
            distance_to_target = np.sqrt((position_x - target_x)**2 + (position_y - target_y)**2)

            # Adjust as needed (how close drone gets to target to consider it as having reached it)
            threshold_distance = 0.1
            return distance_to_target < threshold_distance

    def load(self):
        # Load any saved model or data
        pass

    def save(self):
        # Save the trained Q-values or any other necessary data
        pass