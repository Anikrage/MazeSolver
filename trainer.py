import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualizer import RealTimeVisualizer

class Trainer:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': []
        }
        self.fig = None
        self.visualizer = None

    def train(self, episodes=5000, realtime_animation=True, update_interval=10):
        if realtime_animation:
            self.visualizer = RealTimeVisualizer(
                self.environment.maze, self.agent, self.environment
            )
            self.ani = FuncAnimation(
                self.visualizer.fig, 
                self.visualizer.update_plot, 
                frames=episodes,
                interval=30,  # Reduced from 50ms to 30ms
                blit=True,
                repeat=False,
                cache_frame_data=False  # Reduces memory usage
            )
        
        
# Store performance metrics
        scores = []
        avg_scores = []
        
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
                
            self.agent.decay_exploration()
            
            # Update metrics
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:])  # Rolling 100-episode average
            avg_scores.append(avg_score)
            
            # Update training history
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['exploration_rates'].append(
                self.agent.exploration_rate
            )
            self.training_history['average_scores'] = avg_scores
            
            # Update visualization less frequently
            if realtime_animation and episode % update_interval == 0:
                self.visualizer.update_scores(
                    episode=episode,
                    score=total_reward,
                    avg_score=avg_score,
                    exploration_rate=self.agent.exploration_rate
                )
                plt.pause(0.001)  # Force GUI update
                
            if episode % 100 == 0:
                print(f"Ep {episode}/{episodes} | "
                      f"Score: {total_reward:4.0f} | "
                      f"Avg: {avg_score:4.0f} | "
                      f"Eps: {self.agent.exploration_rate:.4f}")
                
        return self.training_history