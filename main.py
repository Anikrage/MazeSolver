import numpy as np
import matplotlib.pyplot as plt
from maze_generator import MazeGenerator
from maze_environment import MazeEnvironment
from q_learning_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer

def main():
    np.random.seed(42)
    
    print("Generating maze...")
    maze_size = 10
    generator = MazeGenerator(maze_size)
    maze = generator.generate_dfs_maze()
    
    print("Displaying the generated maze...")
    generator.visualize_maze()
    
    environment = MazeEnvironment(maze)
    
    state_space = environment.get_state_space_size()
    agent = QLearningAgent(state_space)
    
    trainer = Trainer(environment, agent)
    
    print("Starting the training process...")
    num_episodes = 1000
    training_history = trainer.train(episodes=num_episodes, realtime_animation=True)
    
    plt.show()  # Keep the animation window open
    
    print("Visualizing training history...")
    Visualizer.plot_training_history(training_history)
    
    print("Finding the best path...")
    best_path = agent.get_best_path(environment)
    print("Visualizing the best path...")
    generator.visualize_maze(path=best_path)
    
    print("Visualizing Q-values...")
    Visualizer.visualize_q_values(agent.q_table, maze)
    
    print("\nTraining Statistics:")
    print(f"Final exploration rate: {agent.exploration_rate:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(training_history['episode_rewards'][-100:]):.2f}")
    print(f"Average steps (last 100 episodes): {np.mean(training_history['episode_lengths'][-100:]):.2f}")
    print(f"Best path length: {len(best_path)}")

if __name__ == "__main__":
    main()
