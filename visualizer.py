import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(history['episode_rewards'])
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
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
    
    @staticmethod
    def visualize_q_values(q_table, maze):
        rows, cols, _ = q_table.shape
        plt.figure(figsize=(12, 12))
        plt.imshow(maze, cmap='binary')
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if maze[i, j] == 0:
                    q_values = q_table[i, j]
                    max_q = np.max(q_values)
                    if max_q > 0:
                        best_action = np.argmax(q_values)
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

class RealTimeVisualizer:
    def __init__(self, maze, agent, environment):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.maze = maze
        self.agent = agent
        self.environment = environment
        
        self.cmap = plt.cm.colors.ListedColormap(['white', 'black'])
        self.img = self.ax.imshow(self.maze, cmap=self.cmap)
        
        self.path_line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.start_point = self.ax.plot(1, 0, 'go', markersize=10)[0]
        self.goal_point = self.ax.plot(maze.shape[1]-1, maze.shape[0]-2, 
                                      'bo', markersize=10)[0]
        
        self.arrows = []
                # Add text elements for scores
        self.score_text = self.ax.text(
            0.02, 0.95, 
            "Episode: 0\nScore: 0\nAvg Score: 0\nEps: 1.0000",
            transform=self.ax.transAxes,
            color='black',
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    def update_scores(self, episode, score, avg_score, exploration_rate):
        """Update the score display without full redraw"""
        self.score_text.set_text(
        f"Episode: {episode}\n"
        f"Score: {score:.0f}\n"
        f"Avg Score: {avg_score:.0f}\n"
        f"Eps: {exploration_rate:.4f}"
    )

    def update_plot(self, frame):
        for arrow in self.arrows:
            arrow.remove()
        self.arrows.clear()
        
        for i in range(1, self.maze.shape[0]-1):
            for j in range(1, self.maze.shape[1]-1):
                if self.maze[i, j] == 0:
                    q_values = self.agent.q_table[i, j]
                    if np.max(q_values) > 0:
                        best_action = np.argmax(q_values)
                        dx, dy = self._get_arrow_direction(best_action)
                        arrow = self.ax.arrow(j, i, dx*0.3, dy*0.3, 
                                            head_width=0.2, color='red')
                        self.arrows.append(arrow)
        
        path = self.agent.get_best_path(self.environment)
        if len(path) > 1:
            path_array = np.array(path)
            self.path_line.set_data(path_array[:, 1], path_array[:, 0])
            
        return [self.img, self.path_line] + self.arrows
    
    def _get_arrow_direction(self, action):
        if action == 0:  # Up
            return 0, -0.3
        elif action == 1:  # Right
            return 0.3, 0
        elif action == 2:  # Down
            return 0, 0.3
        elif action == 3:  # Left
            return -0.3, 0
        return 0, 0
