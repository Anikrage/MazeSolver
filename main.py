import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from maze_generator import MazeGenerator
from maze_environment import MazeEnvironment
from q_learning_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Maze Reinforcement Learning')
    parser.add_argument('--maze_size', type=int, default=10, help='Size of the maze')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--update_interval', type=int, default=5, help='Visualization update interval (episodes)')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: generate different maze each time)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up random generator - using None will generate different mazes each time
    seed_message = f"with seed {args.seed}" if args.seed is not None else "with random seed (different maze each run)"
    print(f"Starting Maze RL Project {seed_message}")
    
    start_time = time.time()
    
    # Generate maze with proper randomization
    print(f"Generating maze of size {args.maze_size}x{args.maze_size}...")
    generator = MazeGenerator(args.maze_size, seed=args.seed)
    maze = generator.generate_dfs_maze()
    
    # Display initial maze
    print("Displaying the generated maze...")
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='binary')
    plt.title("Generated Maze")
    plt.colorbar(label='0=Path, 1=Wall')
    plt.show()
    
    # Create environment, agent, and visualizer
    environment = MazeEnvironment(maze)
    agent = QLearningAgent(environment.get_state_space_size(), seed=args.seed)
    visualizer = Visualizer(maze, agent, environment)
    
    # Train the agent
    print(f"Starting training for {args.episodes} episodes...")
    trainer = Trainer(environment, agent)
    
    # Training with real-time visualization if enabled
    if args.visualize:
        training_history = trainer.train(
            episodes=args.episodes,
            realtime_animation=True,
            update_interval=args.update_interval
        )
        plt.show()  # Keep visualization window open
    else:
        # Training without visualization (faster)
        training_history = trainer.train(
            episodes=args.episodes,
            realtime_animation=False
        )
    
    # Find best path after training
    best_path = agent.get_best_path(environment)
    path_length = len(best_path)
    
    # Training summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Maze size: {args.maze_size}x{args.maze_size}")
    print(f"Episodes trained: {args.episodes}")
    print(f"Final exploration rate: {agent.exploration_rate:.6f}")
    print(f"Best path length: {path_length}")
    print(f"Last 100 episodes - Avg reward: {np.mean(training_history['episode_rewards'][-100:]):.2f}")
    print(f"Last 100 episodes - Avg steps: {np.mean(training_history['episode_lengths'][-100:]):.2f}")
    print("="*50)
    
    # Visualize final results
    print("\nVisualizing training history...")
    visualizer.plot_training_history(training_history)
    
    print("Visualizing best path...")
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary')
    path_array = np.array(best_path)
    plt.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
    plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10)  # Start
    plt.plot(path_array[-1, 1], path_array[-1, 0], 'bo', markersize=10)  # End
    plt.title(f'Final Optimal Path (Length: {path_length})')
    plt.show()
    
    print("Visualizing Q-values heatmap...")
    visualizer.visualize_q_values()
    
    # Save model if requested
    if args.save_model:
        timestamp = int(time.time())
        model_name = f'q_table_maze{args.maze_size}_{args.episodes}ep_{timestamp}.npy'
        np.save(model_name, agent.q_table)
        print(f"Model saved to {model_name}")
    
    visualizer.close()
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()
