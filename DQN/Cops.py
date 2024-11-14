import numpy as np
from GridWorldConstants import *

class Cop:
    def __init__(self, pos, bank):
        self.position = pos
        self.initial_pos = np.copy(pos)
        self.initial_step_dir = np.array([0, 0])
        self.step_dir = np.array([0, 0])
        self.old_pos = np.array([])
        self.corners = {}

        #Calculates the "corners" of cop's path based on its position relative to the bank 
        vec = self.position - bank
        vec = int(vec[0]), int(vec[1])
        abs_vec = np.absolute(vec)
        radius = max(abs_vec[0], abs_vec[1])
        self.corners[(bank[0] - radius, bank[1] - radius)] = (0, 1)
        self.corners[(bank[0] - radius, bank[1] + radius)] = (1, 0)
        self.corners[(bank[0] + radius, bank[1] + radius)] = (0, -1)
        self.corners[(bank[0] + radius, bank[1] - radius)] = (-1, 0)
        temp = np.array([])
        key = (self.position[0], self.position[1])

        if key in self.corners:
            self.step_dir[0] = self.corners[key][0]
            self.step_dir[1] = self.corners[key][1]
        else:
            if abs(vec[1]) == radius:
                if vec[1] > 0:
                    temp = np.array([-radius, radius])
                else:
                    temp = np.array([radius, -radius])
            else:
                if vec[0] > 0:
                    temp = np.array([radius, radius])
                else:
                    temp = np.array([-radius, -radius])

            temp = bank + temp

            key = (temp[0], temp[1])
            self.step_dir[0] = self.corners[key][0]
            self.step_dir[1] = self.corners[key][1]

        # Get initial step direction - used for resetting
        self.initial_step_dir = np.copy(self.step_dir)

    def reset(self):
        self.position[0] = self.initial_pos[0]
        self.position[1] = self.initial_pos[1]

        self.step_dir[0] = self.initial_step_dir[0]
        self.step_dir[1] = self.initial_step_dir[1]
    
    def step(self):
        self.old_pos = self.position
        # self.position = self.position + self.step_dir
        # key = (self.position[0], self.position[1])
        # if key in self.corners:
        #     self.step_dir[0] = self.corners[key][0]
        #     self.step_dir[1] = self.corners[key][1]

        self.position = self.position + self.step_dir
        key = (self.position[0], self.position[1])
        if key in self.corners:
            self.step_dir[0] = self.corners[key][0]
            self.step_dir[1] = self.corners[key][1]
        
        return self.position
    
    def __str__(self):
        return "Cop position: " + str(self.position) + " Step direction: " + str(self.step_dir) + " Corners: " + str(self.corners) + "\n"
        