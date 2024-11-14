import numpy as np
from scipy.linalg import block_diag
import pygame
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from GridWorldConstants import *

from Robber import Robber
from Cops import Cop

class GridWorld(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, init_state=None, goal=None, size=GRID_SIZE):
    
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward = 0 # The reward for the current state
        self.time = 0 # The time step
        same = True #boolean to check if robber and bank are in the same position
        self.frames = []

        #for rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(-1000, 100, shape=((VISION_RADIUS*2 + 1)*(VISION_RADIUS*2 + 1),), dtype=float),  #vision is the 9x9 grid around the robber
                "robber_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int), #robbers (x,y) position
                "dist":  spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=float), #distance between robber and bank
                "bank_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int), #bank (x,y) position
            }
        )

        self.action_space = spaces.Discrete(4)
        self._action_to_direction= {
            0: np.array([1, 0]),  # right
            1: np.array([-1, 0]),  # left
            2: np.array([0, 1]),  # down
            3: np.array([0, -1]),  # up
        }

        #Bank location is always in the middle
        self.bank_location = np.array([np.floor((size - 1)/2), np.floor((size-1)/2)], dtype=int)
        self.goal = np.copy(self.bank_location)

        # Initialize cops and robber
        self.cops = []
        self.cop_position = []
        self.old_cop_position = []
        self.num_cops = NUM_COPS

        #generate random cop positions
        for i in range(self.num_cops):
            temp = np.array([np.random.randint(0, size), np.random.randint(0, size)], dtype=int)
            while np.array_equal(temp, self.bank_location):
                temp = np.array([np.random.randint(0, size), np.random.randint(0, size)], dtype=int)
            self.cop_position.append(temp)
            self.old_cop_position.append(np.copy(self.cop_position[i]))
            self.cops.append(Cop(self.cop_position[i], self.bank_location))

        #generate random robber position and make sure it is not the same as the bank and cop positions
        while same:
            self.robber_position = np.array([np.random.randint(0, size), np.random.randint(0, size)], dtype=int)
            for i in range(self.num_cops):
                if np.array_equal(self.robber_position, self.cop_position[i]):
                    same = True
                    break
                else:
                    same = False
            if np.array_equal(self.robber_position, self.bank_location):
                same = True
        #create robber 
        self.robber = Robber(self.robber_position, self.bank_location, self.cop_position)
        self.old_robber_pos = np.copy(self.robber_position)

    
    def _get_obs(self):
        return {"vision": self.robber.vision, "robber_pos": self.robber.position, "dist": np.array([self.robber.dist]), "bank_pos": self.bank_location}

   
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        same = True
        self.reward = 0
        self.time = 0
        self.frames = []

        #reset cops to original positions
        for i in range(self.num_cops):
            self.cops[i].reset()
        
        #rest robber to random position
        while same:
            self.robber_position = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)], dtype=int)
            for i in range(self.num_cops):
                if np.array_equal(self.robber_position, self.cops[i].position):
                    same = True
                    break
                else:
                    same = False
            if np.array_equal(self.robber_position, self.bank_location):
                same = True
        self.old_robber_pos = np.copy(self.robber_position)
        self.robber.reset(self.robber_position, self.bank_location, self.cop_position)
        observation = self.robber.get_state()

        if self.render_mode == "human":
            self._render_frame()


        return self._get_obs(), {}

            
    def step(self, action):
        self.reward = 0
        self.time += 1
        truncated = False
        self.old_robber_pos = np.copy(self.robber.position)
        
        for i in range(self.num_cops):
            self.old_cop_position[i] = np.copy(self.cop_position[i])   
        for i in range(self.num_cops):
            self.cop_position[i] = self.cops[i].step()
        observation, reward, terminated = self.robber.step(action, self.cop_position, self.old_cop_position)
        if self.render_mode == "human":
            self._render_frame()
        self.reward += reward
        if self.time == TIMEOUT:
            truncated = True
            self.reward -= TIMEOUT_PENALTY
        else:
            self.reward -= self.time
        return self._get_obs(), self.reward, terminated, truncated, {}


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Drawing out the robber's vision with light grey
        min_coord = self.robber.position - self.robber.radius

        if min_coord[0] < 0:
            min_coord[0] = 0
        if min_coord[1] < 0:
            min_coord[1] = 0
        
        max_coord = self.robber.position + self.robber.radius
        if max_coord[0] > self.size - 1:
            max_coord[0] = self.size - 1
        if max_coord[1] > self.size - 1:
            max_coord[1] = self.size - 1
        
        for y in range(min_coord[1], max_coord[1] + 1):
            for x in range(min_coord[0] , max_coord[0] + 1):
                pygame.draw.rect(
                    canvas,
                    (211, 211, 211),
                    pygame.Rect(
                        pix_square_size * np.array([x, y]),
                        (pix_square_size, pix_square_size)
                    )
                )

        # Marks bank position with yellow
        pygame.draw.rect(
            canvas,
            (235, 206, 61),
            pygame.Rect(
                pix_square_size * self.bank_location,
                (pix_square_size, pix_square_size)
            )
        )

        # For each cop, draw a blue circle
        for cop in self.cops:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (cop.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        
        # Draw robber (red circle)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.robber.position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )


        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            frame_data = pygame.image.tostring(self.window, 'RGB')
            frame_surface = pygame.image.fromstring(frame_data, (self.window_size, self.window_size), 'RGB')
            temp_frame = pygame.surfarray.array3d(frame_surface)
            temp_frame = np.rot90(temp_frame, 3)
            temp_frame = np.flip(temp_frame, axis=1)
            self.frames.append(temp_frame)
            
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )   
    
    
        
    @property
    def n_actions(self):
        return self._n_actions

    @property 
    def n_states(self):
        return self._n_states

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        

           




