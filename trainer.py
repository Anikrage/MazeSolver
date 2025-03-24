import numpy as np
from datetime import datetime

# Import SummaryWriter for TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class Trainer:
    def __init__(self, environment, agent, log_tensorboard=False):
        """
        Initialize the trainer.
        
        Args:
            environment: The maze environment.
            agent: The Q-learning agent.
            log_tensorboard: Boolean flag to enable TensorBoard logging.
        """
        self.environment = environment
        self.agent = agent
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': []
        }
        self.log_tensorboard = log_tensorboard
        if self.log_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=f'runs/maze_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        else:
            self.writer = None
    
    def train_generator(self, episodes=5000, checkpoint_interval=500):
        """
        Generator that trains the agent one episode at a time and yields training data.
        
        Yields a dictionary with keys:
          - episode
          - total_reward
          - steps
          - exploration_rate
          - q_table (copy)
          - best_path (from current Q-table)
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
            
            # Log to TensorBoard if enabled
            if self.writer:
                self.writer.add_scalar('Reward/Episode', total_reward, episode)
                self.writer.add_scalar('Steps/Episode', steps, episode)
                self.writer.add_scalar('Exploration Rate', self.agent.exploration_rate, episode)
            
            # Optional: save checkpoint
            if episode % checkpoint_interval == 0 and self.writer:
                self.writer.add_text('Checkpoint', f'Episode {episode}', episode)
            
            yield {
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'exploration_rate': self.agent.exploration_rate,
                'q_table': self.agent.q_table.copy(),
                'best_path': self.agent.get_best_path(self.environment)
            }
        
        if self.writer:
            self.writer.close()
