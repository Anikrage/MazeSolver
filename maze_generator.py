import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class MazeGenerator:
    def __init__(self, size, seed=None):
        """
        Initialize the maze generator with a given size and random seed.
        
        Args:
            size: The size of the maze (actual dimensions will be 2*size+1)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.maze = None
        self.rng = np.random.RandomState(seed)
        
    def generate_dfs_maze(self):
        """
        Generate a maze using the Depth-First Search algorithm.
        Returns a 2D numpy array where 0=path and 1=wall.
        """
        size = self.size
        grid = np.ones((size*2+1, size*2+1))
        visited = np.zeros((size, size), dtype=bool)
        
        def dfs(row, col):
            visited[row][col] = True
            maze_row, maze_col = 2*row+1, 2*col+1
            grid[maze_row][maze_col] = 0
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.rng.shuffle(directions)
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < size and 0 <= new_col < size and 
                    not visited[new_row][new_col]):
                    grid[maze_row + dr][maze_col + dc] = 0
                    dfs(new_row, new_col)
        
        # Use random starting point
        start_row, start_col = self.rng.randint(0, size), self.rng.randint(0, size)
        dfs(start_row, start_col)
        
        grid[1][0] = 0  # Start point (entrance)
        grid[size*2-1][size*2] = 0  # End point (exit)
        
        self.maze = grid
        return grid
    
    def visualize_maze(self, path=None):
        """
        Visualize the maze and optionally show a path through it.
        
        Args:
            path: Optional list of coordinates representing a path through the maze
        """
        if self.maze is None:
            print("No maze to visualize. Generate a maze first.")
            return
            
        plt.figure(figsize=(10, 10))
        cmap = colors.ListedColormap(['white', 'black'])  # 0 = path, 1 = wall
        plt.imshow(self.maze, cmap=cmap)
        
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
            plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10)  # Start
            plt.plot(path_array[-1, 1], path_array[-1, 0], 'bo', markersize=10)  # End
        
        plt.title('Maze with Optimal Path' if path else 'Maze')
        plt.grid(False)
        plt.show()
