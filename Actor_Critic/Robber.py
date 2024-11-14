import numpy as np
from GridWorldConstants import * 

class Robber:
    def __init__(self, position, bank_pos, cops_pos):
        self._to_state = {}
        self.position = position
        self.bank_pos = bank_pos
        self.radius = VISION_RADIUS
        self.dist = np.linalg.norm(self.position - self.bank_pos)
        self.reward = 0
        self.goal = np.copy(self.bank_pos)
        self.old_pos = np.copy(self.position)
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([-1, 0]),  # left
            2: np.array([0, 1]),  # down
            3: np.array([0, -1]),  # up
        }



         # Vision set up
        state_num = 0
        for y in range(-VISION_RADIUS, VISION_RADIUS+1):
            for x in range(-VISION_RADIUS, VISION_RADIUS+1):
                self._to_state[(x, y)] = state_num
                state_num = state_num + 1

        self._to_cell = {v: k for k, v in self._to_state.items()}
        self.vision = np.zeros(len(self._to_cell))

        # Mark its current position with a "1"
        self.vision[self._to_state[(0, 0)]] = 1

        # Mark out of bounds positions with a "-1000"
        xpos = self.position[0]
        ypos = self.position[1]
        for y in range(-self.radius, self.radius+1):
            for x in range(-self.radius, self.radius+1):
                if (xpos + x < 0 or xpos + x > GRID_SIZE - 1 or ypos + y < 0 or ypos + y > GRID_SIZE - 1):
                    self.vision[self._to_state[(x, y)]] = -5

        # Mark Cop positions in vision with a "-1000"
        for cop_position in cops_pos:
            vec = cop_position - self.position
            vec = (vec[0], vec[1])

            if vec in self._to_state:
                # print("Cop spotted at position: ", vec)
                # print("Corresponding index: ", self._to_state[vec])
                self.vision[self._to_state[vec]] = -1000
        
        # Mark Bank position in vision with a "100"
        vec = self.goal - self.position
        vec = (vec[0], vec[1])
        if vec in self._to_state:
            # print("Bank spotted at position: ", vec)
            # print("Corresponding index: ", self._to_state[vec])
            self.vision[self._to_state[vec]] = 100
        
        # print("Vision: ", self.vision)


    def step(self, action, cops_pos, cops_old_pos):
        #update robber position
        self.old_pos = np.copy(self.position)
        self.position = np.clip(self.position + self._action_to_direction[action], 0, GRID_SIZE - 1)

        #update robber vision
        self.vision = np.zeros(len(self._to_cell))
        # Mark its current position with a "1"
        self.vision[self._to_state[(0, 0)]] = 1
        # Mark out of bounds positions with a "-1000"
        xpos = self.position[0]
        ypos = self.position[1]
        for y in range(-self.radius, self.radius+1):
            for x in range(-self.radius, self.radius+1):
                if (xpos + x < 0 or xpos + x > GRID_SIZE - 1 or ypos + y < 0 or ypos + y > GRID_SIZE - 1):
                    self.vision[self._to_state[(x, y)]] = -5

        # Mark Cop positions in vision with a "-1000"
        for cop_position in cops_pos:
            vec = cop_position - self.position
            vec = (vec[0], vec[1])

            if vec in self._to_state:
                # print("Cop spotted at position: ", vec)
                # print("Corresponding index: ", self._to_state[vec])
                self.vision[self._to_state[vec]] = -50
        
        # Mark Bank position in vision with a "100"
        vec = self.goal - self.position
        vec = (vec[0], vec[1])
        if vec in self._to_state:
            # print("Bank spotted at position: ", vec)
            # print("Corresponding index: ", self._to_state[vec])
            self.vision[self._to_state[vec]] = 100

        #update robber distance from bank
        prev_dist = self.dist
        self.dist = np.linalg.norm(self.position - self.bank_pos)


        #update robber reward
        self.reward = 0
        
        #if robber is at the bank, reward is 100
        if np.array_equal(self.position, self.bank_pos):
            self.reward = BANK_REWARD
            observation = [self.vision, self.position, self.dist]
            return observation, self.reward, True
        
        #if robber is caught by a cop, reward is -100
        for i in range(len(cops_pos)):
            if np.array_equal(self.position, cops_pos[i]):
                self.reward = CAUGHT_PUNISHMENT
                observation = [self.vision, self.position, self.dist]
                return observation, self.reward, True
            if np.array_equal(self.position, cops_old_pos[i]) and np.array_equal(self.old_pos, cops_pos[i]):
                self.reward = CAUGHT_PUNISHMENT
                observation = [self.vision, self.position, self.dist]
                return observation, self.reward, True
        
        # Moved further
        if (prev_dist <= self.dist):
            self.reward -= MOVE_FURTHER
        else:
            self.reward -= MOVE_CLOSER
        
        return [self.vision, self.position, self.dist], self.reward, False
            


    def reset(self, position, bank_position, cops_position):
        self.position = position
        self.bank_pos = bank_position
        self.dist = np.linalg.norm(self.position - self.bank_pos)
        self.reward = 0
        self.goal = np.copy(self.bank_pos)
        self.old_pos = np.copy(self.position)

        # Clear out its vision
        for i in range(0, len(self.vision)):
            self.vision[i] = 0
        
        # Mark its current position with a "1"
        self.vision[self._to_state[(0, 0)]] = 1

        # Mark out of bounds positions with a "-1000"
        xpos = self.position[0]
        ypos = self.position[1]
        for y in range(-VISION_RADIUS, VISION_RADIUS+1):
            for x in range(-VISION_RADIUS, VISION_RADIUS+1):
                if (xpos + x < 0 or xpos + x > GRID_SIZE - 1 or ypos + y < 0 or ypos + y > GRID_SIZE - 1):
                    self.vision[self._to_state[(x, y)]] = -5

        # Mark Cop positions in vision with a "-1000"
        for cop_position in cops_position:
            vec = cop_position - self.position
            vec = (vec[0], vec[1])

            if vec in self._to_state:
                # print("Cop spotted at position: ", vec)
                # print("Corresponding index: ", self._to_state[vec])
                self.vision[self._to_state[vec]] = -1000
        
        # Mark Bank position in vision with a "100"
        vec = self.goal - self.position
        vec = (vec[0], vec[1])
        if vec in self._to_state:
            # print("Bank spotted at position: ", vec)
            # print("Corresponding index: ", self._to_state[vec])
            self.vision[self._to_state[vec]] = 100
    
    def get_state(self):
        return self.vision, self.position, self.dist
        

       



        

        
