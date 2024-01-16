import pygame
from drone import Drone
from pygame import Rect
import numpy as np
import math
from typing import Tuple
from flight_controller import FlightController


#---------------------WRITE YOUR OWN CODE HERE------------------------#
from heuristic_controller import HeuristicController
from heuristic_controller_new import CustomController
from dqn_controller import DQNController, RMSpropOptimizer, NeuralNetwork
from tuning import HeuristicController
# from tuning_2 import CustomController
# from custom_controller import CustomController
from q_learning import QLearningController, heuristic1, heuristic2

def generate_controller() -> FlightController:
    # return HeuristicController() # <--- Replace this with your own written controller

    return DQNController()
    # return QLearningController()
    # return CustomController()
    # return controller

def is_training() -> bool:
    return True # <--- Replace this with True if you want to train, false otherwise
def is_saving() -> bool:
    return False # <--- Replace this with True if you want to save the results of training, false otherwise

#---------------------------------------------------------------------#
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 480


def get_scale():
    return min(SCREEN_HEIGHT, SCREEN_WIDTH)

def convert_to_screen_coordinate(x,y):
    scale = get_scale()
    return (x*scale + SCREEN_WIDTH/2, -y*scale + SCREEN_HEIGHT/2)

def convert_to_screen_size(game_size):    
    scale = get_scale()
    return game_size*scale

def convert_to_game_coordinates(x,y):
    scale = get_scale()
    return ((x - SCREEN_WIDTH/2)/scale, (y - SCREEN_HEIGHT/2)/scale)

def main(controller: FlightController):

    # Initialise pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Load the relevant graphics into pygame
    drone_img = pygame.image.load('graphics/drone_small.png')
    background_img = pygame.image.load('graphics/background.png')
    target_img = pygame.image.load('graphics/target.png')

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Initalise the drone
    drone = controller.init_drone()
    
    simulation_step_counter = 0
    max_simulation_steps = controller.get_max_simulation_steps()
    delta_time = controller.get_time_interval()

    # drone_x = []
    # drone_y = []
    # drone_velocity_y = []
    # drone_pitch_velocity = []
    # drone_pitch = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        # --- Begin Physics --- #
        # Get the thrust information from the controller
        drone.set_thrust(controller.get_thrusts(drone))
        # Update the simulation
        drone.step_simulation(delta_time)
        # if(drone.thrust_left!=0.5 or drone.thrust_right!=0.5):
        #     print(drone.thrust_left, drone.thrust_right)
        # drone_x.append(drone.x)
        # drone_y.append(drone.y)
        # drone_velocity_y.append(drone.velocity_y)
        # drone_pitch_velocity.append(drone.pitch_velocity)
        # drone_pitch.append(drone.pitch)


        # --- Begin Drawing --- #

        # Refresh the background
        screen.blit(background_img, (0,0))
        # Draw the current drone on the screen
        draw_drone(screen, drone, drone_img)
        # Draw the next target on the screen
        draw_target(drone.get_next_target(), screen, target_img)

        # Actually displays the final frame on the screen
        pygame.display.flip()

        # Makes sure that the simulation runs at a target 60FPS
        clock.tick(60)

        # Checks whether to reset the current drone
        simulation_step_counter+=1
        if (simulation_step_counter>max_simulation_steps):
            # print(f"min:{min(drone_x)}\n{min(drone_y)}\n{min(drone_velocity_y)}\n{min(drone_pitch_velocity)}\n{min(drone_pitch)}\n\n")
            # print(f"max:{max(drone_x)}\n{max(drone_y)}\n{max(drone_velocity_y)}\n{max(drone_pitch_velocity)}\n{max(drone_pitch)}")
            drone = controller.init_drone() # Reset the drone
            simulation_step_counter = 0

    

def draw_target(target_point, screen, target_img):
    target_size = convert_to_screen_size(0.1)
    point_x, point_y = convert_to_screen_coordinate(*target_point)
    screen.blit(pygame.transform.scale(target_img, (int(target_size), int(target_size))), (point_x-target_size/2, point_y-target_size/2))

def draw_drone(screen: pygame.Surface, drone: Drone, drone_img: pygame.Surface):
    drone_x, drone_y = convert_to_screen_coordinate(drone.x, drone.y)
    drone_width = convert_to_screen_size(0.3)
    drone_height = convert_to_screen_size(0.15)
    drone_rect = Rect(drone_x-drone_width/2, drone_y-drone_height/2, drone_width, drone_height)
    drone_scaled_img = pygame.transform.scale(drone_img, (int(drone_width), int(drone_height)))
    drone_scaled_center = drone_scaled_img.get_rect(topleft = (drone_x-drone_width/2, drone_y-drone_height/2)).center
    rotated_drone_img = pygame.transform.rotate(drone_scaled_img, -drone.pitch * 180 / math.pi)
    drone_scaled_rect = rotated_drone_img.get_rect(center=drone_scaled_center)
    screen.blit(rotated_drone_img, drone_scaled_rect)

if __name__ == "__main__":

    controller = generate_controller()
    if is_training():
        controller.train()
        if is_saving():
            controller.save()        
    else:
        controller.load()
    # print(controller.q_values)
    main(controller)