import numpy as np

class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.rows, self.cols = maze.shape
        self.start_pos = (1, 0)
        self.goal_pos = (self.rows-2, self.cols-1)
        self.current_pos = self.start_pos
        self.steps_taken = 0
        self.max_steps = self.rows * self.cols * 2
        
    def reset(self):
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.current_pos
        
    def step(self, action):
        self.steps_taken += 1
        row, col = self.current_pos
        
        if action == 0:  # up
            new_pos = (row-1, col)
        elif action == 1:  # right
            new_pos = (row, col+1)
        elif action == 2:  # down
            new_pos = (row+1, col)
        elif action == 3:  # left
            new_pos = (row, col-1)
        else:
            raise ValueError(f"Invalid action: {action}")
            
        if (0 <= new_pos[0] < self.rows and 
            0 <= new_pos[1] < self.cols and 
            self.maze[new_pos] == 0):
            self.current_pos = new_pos
            
        if self.current_pos == self.goal_pos:
            reward = 100
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -100
            done = True
        else:
            reward = -1
            done = False
            
        return self.current_pos, reward, done, {}
    
    def get_state_space_size(self):
        return self.rows, self.cols
