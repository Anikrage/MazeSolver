import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MazeEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, maze):
        """
        Initialize the maze environment.
        
        Args:
            maze: A 2D numpy array where 0 represents paths and 1 represents walls.
        """
        super(MazeEnvironment, self).__init__()
        self.maze = maze
        self.rows, self.cols = maze.shape

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Tuple((spaces.Discrete(self.rows), spaces.Discrete(self.cols)))

        # Set default start and goal positions
        self.start_pos = (1, 0)
        self.goal_pos = (self.rows - 2, self.cols - 1)

        # Ensure start and goal are valid paths
        assert self.maze[self.start_pos] == 0, "Start position must be a path"
        assert self.maze[self.goal_pos] == 0, "Goal position must be a path"

        self.current_pos = self.start_pos
        self.steps_taken = 0
        self.max_steps = self.rows * self.cols * 2

    def reset(self, seed=None, options=None):
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.current_pos, {}

    def step(self, action):
        self.steps_taken += 1
        row, col = self.current_pos
        
        if action == 0:      # up
            new_pos = (row - 1, col)
        elif action == 1:    # right
            new_pos = (row, col + 1)
        elif action == 2:    # down
            new_pos = (row + 1, col)
        elif action == 3:    # left
            new_pos = (row, col - 1)
        else:
            raise ValueError("Invalid action. Must be 0, 1, 2, or 3.")
        
        # Check validity: within bounds and not a wall
        if (0 <= new_pos[0] < self.rows and 
            0 <= new_pos[1] < self.cols and 
            self.maze[new_pos] == 0):
            self.current_pos = new_pos
        
        if self.current_pos == self.goal_pos:
            reward = 100
            done = True
            info = {'success': True}
        elif self.steps_taken >= self.max_steps:
            reward = -100
            done = True
            info = {'timeout': True}
        else:
            reward = -1
            done = False
            info = {}
        
        truncated = False
        return self.current_pos, reward, done, truncated, info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(self.maze, cmap='binary')
        plt.plot(self.current_pos[1], self.current_pos[0], 'ro', markersize=10)
        plt.title("Maze Environment")
        plt.show()

    def close(self):
        pass
