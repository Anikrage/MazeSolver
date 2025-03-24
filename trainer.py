import numpy as np
import time

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
        
    def train(self, episodes=5000, render_interval=None, render_delay=0.1):
        """
        Train the agent on the environment.
        
        Args:
            episodes: The number of episodes to train for.
            render_interval: If not None, render the environment every render_interval episodes.
            render_delay: Delay between renders in seconds.
            
        Returns:
            The training history.
        """
        print(f"Starting training for {episodes} episodes...")
        start_time = time.time()
        
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
            
            # Print progress
            if episode % 100 == 0 or episode == 1:
                print(f"Episode {episode}/{episodes}, "
                      f"Reward: {total_reward}, "
                      f"Steps: {steps}, "
                      f"Exploration Rate: {self.agent.exploration_rate:.4f}")
                
            # Render if requested
            if render_interval and episode % render_interval == 0:
                path = self.agent.get_best_path(self.environment)
                self.render_maze_with_path(path)
                time.sleep(render_delay)
                
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")
        return self.training_history
    
    def render_maze_with_path(self, path):
        """
        Render the maze with the current best path.
        
        Args:
            path: A list of positions representing the path.
        """
        # This assumes the environment has a maze attribute with a visualize_maze method
        if hasattr(self.environment, 'maze') and hasattr(self.environment.maze, 'visualize'):
            self.environment.maze.visualize(path)
        else:
            print("Cannot render maze: environment does not have the required attributes.")
