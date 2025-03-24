import numpy as np

class MazeEnvironment:
    def __init__(self, maze):
        """
        Initialize the maze environment.
        
        Args:
            maze: A 2D numpy array where 0 represents paths and 1 represents walls.
        """
        self.maze = maze
        self.rows, self.cols = maze.shape
        
        # Find start and end positions
        self.start_pos = (1, 0)  # Default start position
        self.goal_pos = (self.rows-2, self.cols-1)  # Default goal position
        
        # Ensure the start and end positions are valid
        assert self.maze[self.start_pos] == 0, "Start position must be a path"
        assert self.maze[self.goal_pos] == 0, "Goal position must be a path"
        
        self.current_pos = self.start_pos
        self.steps_taken = 0
        self.max_steps = self.rows * self.cols * 2  # Prevent infinite loops
        
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            The initial state (position).
        """
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.current_pos
        
    def step(self, action):
        """
        Take a step in the maze based on the action.
        
        Args:
            action: An integer representing the action to take:
                   0: up, 1: right, 2: down, 3: left
        
        Returns:
            A tuple (new_state, reward, done, info)
        """
        self.steps_taken += 1
        
        # Get current position
        row, col = self.current_pos
        
        # Determine new position based on action
        if action == 0:  # up
            new_pos = (row-1, col)
        elif action == 1:  # right
            new_pos = (row, col+1)
        elif action == 2:  # down
            new_pos = (row+1, col)
        elif action == 3:  # left
            new_pos = (row, col-1)
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, or 3.")
            
        # Check if the move is valid (within bounds and not a wall)
        if (0 <= new_pos[0] < self.rows and 
            0 <= new_pos[1] < self.cols and 
            self.maze[new_pos] == 0):
            self.current_pos = new_pos
            
        # Calculate reward and check if done
        if self.current_pos == self.goal_pos:
            reward = 100  # Large positive reward for reaching the goal
            done = True
            info = {'success': True}
        elif self.steps_taken >= self.max_steps:
            reward = -100  # Large negative reward for taking too many steps
            done = True
            info = {'timeout': True}
        else:
            reward = -1  # Small penalty for each step to encourage efficiency
            done = False
            info = {}
            
        return self.current_pos, reward, done, info
    
    def get_state_space_size(self):
        """
        Returns the size of the state space (rows, cols).
        """
        return self.rows, self.cols
