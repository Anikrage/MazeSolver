import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, maze):
        """
        Initialize the visualizer.
        
        Args:
            maze: The maze as a 2D numpy array.
        """
        self.maze = maze
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.cmap = plt.cm.colors.ListedColormap(['white', 'black'])
        self.ax.imshow(self.maze, cmap=self.cmap)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def update_plot(self, data):
        """
        Update the plot with the latest training data.
        
        Args:
            data: Dictionary from the trainer with keys:
                  'episode', 'total_reward', 'steps', 'exploration_rate', 'q_table', 'best_path'
        """
        self.ax.clear()
        self.ax.imshow(self.maze, cmap=self.cmap)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        q_table = data['q_table']
        # Draw arrows for cells with positive Q-values
        for i in range(1, self.maze.shape[0] - 1):
            for j in range(1, self.maze.shape[1] - 1):
                if self.maze[i, j] == 0:
                    q_values = q_table[i, j]
                    if np.max(q_values) > 0:
                        best_action = np.argmax(q_values)
                        dx, dy = 0, 0
                        if best_action == 0:
                            dy = -0.3
                        elif best_action == 1:
                            dx = 0.3
                        elif best_action == 2:
                            dy = 0.3
                        elif best_action == 3:
                            dx = -0.3
                        self.ax.arrow(j, i, dx, dy, head_width=0.2, color='red', length_includes_head=True)
        
        # Draw best path
        best_path = data['best_path']
        if best_path and len(best_path) > 1:
            path_array = np.array(best_path)
            self.ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2)
            self.ax.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10)
            self.ax.plot(path_array[-1, 1], path_array[-1, 0], 'bo', markersize=10)
        
        # Display training info
        info_text = (f"Episode: {data['episode']}\n"
                     f"Reward: {data['total_reward']}\n"
                     f"Steps: {data['steps']}\n"
                     f"Epsilon: {data['exploration_rate']:.4f}")
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        return self.ax,

    def animate_training(self, train_gen, interval=50):
        """
        Animate the training process.
        
        Args:
            train_gen: A generator yielding training data per episode.
            interval: Delay between frames in milliseconds.
            
        Returns:
            The FuncAnimation object.
        """
        ani = FuncAnimation(self.fig, self.update_plot, frames=train_gen, interval=interval, repeat=False)
        return ani

    def plot_training_history(self, history):
        """
        Plot training history (episode rewards, lengths, and exploration rates).
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        axs[0].plot(history['episode_rewards'])
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)
        
        axs[1].plot(history['episode_lengths'])
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        axs[1].grid(True)
        
        axs[2].plot(history['exploration_rates'])
        axs[2].set_title('Exploration Rate')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Epsilon')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def visualize_q_values(self, q_table):
        """
        Visualize the Q-values as a heatmap.
        """
        q_values = np.max(q_table, axis=2)
        plt.figure(figsize=(12, 12))
        plt.imshow(q_values, cmap='hot')
        plt.colorbar(label='Max Q-value')
        plt.title('Q-values Heatmap')
        plt.grid(False)
        plt.show()
