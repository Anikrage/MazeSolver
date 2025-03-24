import numpy as np

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
    
    def train_generator(self, episodes=5000):
        """
        Generator that trains the agent one episode at a time and yields current training data.
        
        Yields a dictionary with:
          - episode number
          - total_reward for the episode
          - steps taken
          - current exploration rate
          - current Q-table copy
          - best path (computed from the current Q-table)
        """
        for episode in range(1, episodes + 1):
            state, _ = self.environment.reset()
            state = (int(state[0]), int(state[1]))
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _, _ = self.environment.step(action)
                next_state = (int(next_state[0]), int(next_state[1]))
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            self.agent.decay_exploration()
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['exploration_rates'].append(self.agent.exploration_rate)
            
            yield {
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'exploration_rate': self.agent.exploration_rate,
                'q_table': self.agent.q_table.copy(),
                'best_path': self.agent.get_best_path(self.environment)
            }
