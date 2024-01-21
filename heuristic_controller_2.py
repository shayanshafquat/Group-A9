import numpy as np
from flight_controller import FlightController



class HeuristicController2(FlightController):
    def __init__(self):
        # Define the constants for the controller
        self.k = 2  # Proportional constant for altitude control
        self.b = 0.3  # Damping constant for altitude control
        self.k_theta = 4  # Proportional constant for pitch control
        self.b_theta = 0.3  # Damping constant for pitch control
        self.max_thrust = 1.0  # Maximum thrust for each propeller
        self.theta_target = 7  # Target pitch angle in degrees     

    def get_max_simulation_steps(self):
            return 3000 # You can alter the amount of steps you want your program to run for here
    
    def get_thrusts(self, drone):

        target_point = drone.get_next_target()
        dy = target_point[1] - drone.y
        pitch = drone.pitch
        pitch_velocity = drone.pitch_velocity

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
        gamma = np.clip(self.k_theta * (theta_target - pitch) - self.b_theta * pitch_velocity, Teq, -Teq)

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
    
    def train(self):
        # Training is not applicable in this heuristic controller

        pass

    def save(self):
        # Saving is not applicable in this heuristic controller
        pass

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('heuristic_controller_parameters_2.npy')
            self.k = parameter_array[0]
            self.b = parameter_array[1]
            self.k_theta = parameter_array[2]
            self.b_theta = parameter_array[3]
            self.theta_target = parameter_array[4]
        except:
            print("Could not load parameters, sticking with default parameters.")

