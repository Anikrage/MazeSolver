import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Trainer:
    def __init__(self, environment, agent):
        """
        Initialize the trainer.
        
        Args:
            environment: The maze environment.
            agent: The Q-learning agent.
        """
        self.environment = environment
        self.agent = agent
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': []
        }
        self.visualizer = None
        self.ani = None
        
    def train(self, episodes=5000, realtime_animation=False, update_interval=5):
        """
        Train the agent on the environment.
        
        Args:
            episodes: The number of episodes to train for.
            realtime_animation: Whether to show real-time animation.
            update_interval: How often to update the animation.
            
        Returns:
            The training history.
        """
        if realtime_animation:
            from visualizer import Visualizer
            self.visualizer = Visualizer(self.environment.maze, self.agent, self.environment)
            self.ani = FuncAnimation(self.visualizer.fig, self.visualizer.update_plot, 
                                    frames=range(1, episodes+1, update_interval),
                                    interval=100, blit=True)
            plt.draw()
            plt.pause(0.1)
        
        for episode in range(1, episodes + 1):
            state = self.environment.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
            # Decay exploration rate at the end of each episode
            self.agent.decay_exploration()
            
            # Record history
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['exploration_rates'].append(self.agent.exploration_rate)
            
            # Update animation if needed
            if realtime_animation and episode % update_interval == 0:
                plt.pause(0.01)  # Force GUI update
                
            # Print progress
            if episode % 100 == 0 or episode == 1:
                print(f"Episode {episode}/{episodes}, "
                      f"Reward: {total_reward}, "
                      f"Steps: {steps}, "
                      f"Exploration Rate: {self.agent.exploration_rate:.4f}")
                
        return self.training_history
