import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class MazeGenerator:
    def __init__(self, size):
        """
        Initialize the maze generator with a given size.
        The actual maze dimensions will be (2*size+1) x (2*size+1).
        """
        self.size = size
        self.maze = None
        
    def generate_binary_tree_maze(self):
        """
        Generate a maze using the binary tree algorithm.
        """
        # Create a grid with walls (1) everywhere
        size = self.size
        grid = np.ones((size*2+1, size*2+1))
        
        # Set the paths (0) where the agent can move
        for i in range(1, size*2, 2):
            for j in range(1, size*2, 2):
                grid[i][j] = 0
                
                # Randomly carve either north or east (except at boundaries)
                if i == size*2-1 and j == size*2-1:
                    continue
                    
                if i == size*2-1:
                    grid[i][j+1] = 0  # Only east available
                elif j == size*2-1:
                    grid[i+1][j] = 0  # Only north available
                else:
                    if np.random.randint(0, 2) == 0:
                        grid[i+1][j] = 0  # North
                    else:
                        grid[i][j+1] = 0  # East
        
        # Set start and end points
        grid[1][0] = 0  # Start point (entrance)
        grid[size*2-1][size*2] = 0  # End point (exit)
        
        self.maze = grid
        return grid
    
    def generate_dfs_maze(self):
        """
        Generate a maze using the Depth-First Search algorithm.
        This creates more winding paths than the binary tree algorithm.
        """
        size = self.size
        # Create a grid with walls (1) everywhere
        grid = np.ones((size*2+1, size*2+1))
        
        # Mark all cells as unvisited
        visited = np.zeros((size, size), dtype=bool)
        
        # DFS function to carve paths
        def dfs(row, col):
            visited[row][col] = True
            
            # Convert to maze coordinates
            maze_row, maze_col = 2*row+1, 2*col+1
            grid[maze_row][maze_col] = 0
            
            # Define possible directions: (row_offset, col_offset)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            np.random.shuffle(directions)
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new cell is valid and unvisited
                if (0 <= new_row < size and 0 <= new_col < size and 
                    not visited[new_row][new_col]):
                    
                    # Remove the wall between current cell and new cell
                    grid[maze_row + dr][maze_col + dc] = 0
                    
                    # Continue DFS from the new cell
                    dfs(new_row, new_col)
        
        # Start DFS from a random cell
        start_row, start_col = np.random.randint(0, size), np.random.randint(0, size)
        dfs(start_row, start_col)
        
        # Set start and end points
        grid[1][0] = 0  # Start point (entrance)
        grid[size*2-1][size*2] = 0  # End point (exit)
        
        self.maze = grid
        return grid
    
    def visualize_maze(self, path=None):
        """
        Visualize the maze and optionally show a path through it.
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
        
        plt.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.title('Maze with Optimal Path' if path else 'Maze')
        plt.show()
