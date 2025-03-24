import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        """
        Plot the training history.
        
        Args:
            history: A dictionary containing the training history.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot episode rewards
        axs[0].plot(history['episode_rewards'])
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].grid(True)
        
        # Plot episode lengths
        axs[1].plot(history['episode_lengths'])
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        axs[1].grid(True)
        
        # Plot exploration rate
        axs[2].plot(history['exploration_rates'])
        axs[2].set_title('Exploration Rate')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Epsilon')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_q_values(q_table, maze):
        """
        Visualize the Q-values as arrows in the maze.
        
        Args:
            q_table: The Q-table from the agent.
            maze: The maze as a 2D numpy array.
        """
        rows, cols, _ = q_table.shape
        
        # Create a figure
        plt.figure(figsize=(12, 12))
        
        # Plot the maze
        plt.imshow(maze, cmap='binary')
        
        # Plot arrows for each cell
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if maze[i, j] == 0:  # Only show arrows on paths
                    q_values = q_table[i, j]
                    max_q = np.max(q_values)
                    
                    # Only show arrows for cells with learned values
                    if max_q > 0:
                        # Plot an arrow for the best action
                        best_action = np.argmax(q_values)
                        
                        # Define arrow directions (dx, dy)
                        if best_action == 0:  # up
                            dx, dy = 0, -0.3
                        elif best_action == 1:  # right
                            dx, dy = 0.3, 0
                        elif best_action == 2:  # down
                            dx, dy = 0, 0.3
                        elif best_action == 3:  # left
                            dx, dy = -0.3, 0
                            
                        plt.arrow(j, i, dx, dy, head_width=0.2, head_length=0.15, 
                                  fc='red', ec='red', width=0.05)
        
        plt.title('Q-values Visualization')
        plt.grid(False)
        plt.show()
