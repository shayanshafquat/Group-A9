from flight_controller import FlightController

class CustomController(FlightController):
    def __init__(self):
        # Define the constants for the controller
        self.k = 0.1  # Proportional constant for altitude control
        self.b = 0.1  # Damping constant for altitude control
        self.k_theta = 0.1  # Proportional constant for pitch control
        self.b_theta = 0.1  # Damping constant for pitch control
        self.max_thrust = 1.0  # Maximum thrust for each propeller
        self.theta_target = 10  # Target pitch angle in degrees

    def get_thrusts(self, drone):
        # Calculate the equilibrium thrust
        Teq = 0.5 * drone.g * drone.mass

        # Calculate epsilon for vertical motion
        epsilon = self.clip(-self.k * (drone.get_next_target()[1] - drone.y) - self.b * drone.velocity_y, -Teq, Teq)

        # Check if the drone needs to move horizontally
        if drone.get_next_target()[0] > drone.x:
            theta_target = np.radians(self.theta_target)  # Convert to radians for positive x direction
        elif drone.get_next_target()[0] < drone.x:
            theta_target = -np.radians(self.theta_target)  # Convert to radians for negative x direction
        else:
            theta_target = 0  # No pitch needed if x is at target
        
        # Calculate gamma for horizontal motion
        gamma = self.clip(-self.k_theta * (theta_target - drone.get_pitch()) - self.b_theta * drone.pitch_velocity, -Teq, Teq)

        # Determine T1 and T2 based on the need to move horizontally
        if theta_target == 0:
            T1 = T2 = 0.5 + epsilon / drone.max_thrust
        else:
            T1 = 0.5 + (gamma + epsilon) / drone.max_thrust
            T2 = 0.5 - (gamma - epsilon) / drone.max_thrust

        # Ensure that the thrust values are within the allowed range
        T1 = max(0, min(T1, 1))
        T2 = max(0, min(T2, 1))

        return T1, T2

    def clip(self, value, min_value, max_value):
        # Clip the value to be within the min and max bounds
        return max(min_value, min(value, max_value))

    # Implement any additional methods required by the FlightController interface
    def init_drone(self):
        # Initialize your drone here
        return Drone()  # Assuming the Drone class has no required initialization arguments
    
    def get_max_simulation_steps(self):
        # Return the maximum number of simulation steps
        return 1000  # This is an example value, you should adjust it according to your needs
    
    def get_time_interval(self):
        # Return the time interval for each simulation step
        return 0.016  # This represents a simulation step of 60Hz, adjust as necessary

    def train(self):
        # Training is not applicable in this heuristic controller
        pass

    def save(self):
        # Saving is not applicable in this heuristic controller
        pass

    def load(self):
        # Loading is not applicable in this heuristic controller
        pass
