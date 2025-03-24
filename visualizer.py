import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, maze, agent, environment):
        """
        Initialize the visualizer.
        
        Args:
            maze: The maze as a 2D numpy array.
            agent: The Q-learning agent.
            environment: The maze environment.
        """
        self.maze = maze
        self.agent = agent
        self.environment = environment
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.cmap = plt.cm.colors.ListedColormap(['white', 'black'])
        
    def plot_training_history(self, history):
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
    
    def visualize_q_values(self):
        """
        Visualize the Q-values as a heatmap.
        """
        q_values = np.max(self.agent.q_table, axis=2)
        plt.figure(figsize=(12, 12))
        plt.imshow(q_values, cmap='hot')
        plt.colorbar(label='Max Q-value')
        plt.title('Q-values Visualization')
        plt.grid(False)
        plt.show()
    
    def update_plot(self, frame):
        """
        Update the plot for animation.
        
        Args:
            frame: The current frame number.
        
        Returns:
            A list of artists to update.
        """
        self.ax.clear()
        self.ax.imshow(self.maze, cmap=self.cmap)
        
        # Draw arrows for Q-values
        for i in range(1, self.maze.shape[0]-1):
            for j in range(1, self.maze.shape[1]-1):
                if self.maze[i, j] == 0:  # Only show arrows on paths
                    q_values = self.agent.q_table[i, j]
                    max_q = np.max(q_values)
                    
                    # Only show arrows for cells with learned values
                    if max_q > 0:
                        # Plot an arrow for the best action
                        best_action = np.argmax(q_values)
                        
                        if best_action == 0:  # up
                            self.ax.arrow(j, i, 0, -0.3, head_width=0.2, color='red')
                        elif best_action == 1:  # right
                            self.ax.arrow(j, i, 0.3, 0, head_width=0.2, color='red')
                        elif best_action == 2:  # down
                            self.ax.arrow(j, i, 0, 0.3, head_width=0.2, color='red')
                        elif best_action == 3:  # left
                            self.ax.arrow(j, i, -0.3, 0, head_width=0.2, color='red')
        
        # Draw current best path
        path = self.agent.get_best_path(self.environment)
        if path and len(path) > 1:
            path_array = np.array(path)
            self.ax.plot(path_array[:, 1], path_array[:, 0], 'b-')
            self.ax.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10)  # Start
            self.ax.plot(path_array[-1, 1], path_array[-1, 0], 'bo', markersize=10)  # End
        
        # Add score text
        self.ax.text(0.02, 0.98, f"Episode: {frame}", transform=self.ax.transAxes,
                     verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        return self.ax,
        
    def close(self):
        """
        Close any resources used by the visualizer.
        """
        plt.close()
