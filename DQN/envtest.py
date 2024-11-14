import gymnasium
from GridWorld import GridWorld 
import pygame
import numpy as np



a = 0
quit = False

action = 0

def register_input():
    global quit, restart, a

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a = 1
            if event.key == pygame.K_RIGHT:
                a = 0
            if event.key == pygame.K_UP:
                a = 3
            if event.key == pygame.K_DOWN:
                a = 2
rng = np.random.default_rng()

env = GridWorld(render_mode="human")
obs, info= env.reset()

total_reward = 0.0
steps = 0
restart = False
while not quit:

    register_input()

    observation, reward, terminated, truncated, info = env.step(a)
    print(reward)

    a = 1
    #print vision as a 9x9 grid with spaces
    #
    vision = observation["vision"]
    # print(vision)
   
    if terminated:
        print("Environment reset")
        env.reset()
    if quit:
        break
    pygame.time.Clock().tick(5)

    if truncated:
        print("Truncated")
        break
env.close()