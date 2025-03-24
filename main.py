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
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--log_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: generate different maze each time)')
    return parser.parse_args()

def main():
    args = parse_args()
    seed_message = f"with seed {args.seed}" if args.seed is not None else "with random seed (different maze each run)"
    print(f"Starting Maze RL Project {seed_message}")
    
    start_time = time.time()
    
    # Generate maze
    print(f"Generating maze of size {args.maze_size}x{args.maze_size}...")
    generator = MazeGenerator(args.maze_size, seed=args.seed)
    maze = generator.generate_dfs_maze()
    
    # Display generated maze
    print("Displaying the generated maze...")
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='binary')
    plt.title("Generated Maze")
    plt.colorbar(label='0=Path, 1=Wall')
    plt.show()
    
    # Create Gymnasium environment and agent
    environment = MazeEnvironment(maze)
    agent = QLearningAgent((environment.rows, environment.cols), seed=args.seed)
    
    # Initialize Trainer with optional TensorBoard logging
    trainer = Trainer(environment, agent, log_tensorboard=args.log_tensorboard)
    
    if args.visualize:
        visualizer = Visualizer(maze)
        # Use the training generator for smooth animation
        ani = visualizer.animate_training(trainer.train_generator(episodes=args.episodes), interval=50)
        plt.show()  # This call will block and show the animation window
    else:
        # Run training without visualization
        for _ in trainer.train_generator(episodes=args.episodes):
            pass
    
    # After training, extract the best path
    best_path = agent.get_best_path(environment)
    path_length = len(best_path)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Maze size: {args.maze_size}x{args.maze_size}")
    print(f"Episodes trained: {args.episodes}")
    print(f"Final exploration rate: {agent.exploration_rate:.6f}")
    print(f"Best path length: {path_length}")
    print("="*50)
    
    # Plot training history
    visualizer.plot_training_history(trainer.training_history)
    
    # Visualize final optimal path
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary')
    path_array = np.array(best_path)
    plt.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
    plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10)
    plt.plot(path_array[-1, 1], path_array[-1, 0], 'bo', markersize=10)
    plt.title(f'Final Optimal Path (Length: {path_length})')
    plt.show()
    
    # Visualize Q-values heatmap
    visualizer.visualize_q_values(agent.q_table)
    
    # Save model if requested
    if args.save_model:
        timestamp = int(time.time())
        model_name = f'q_table_maze{args.maze_size}_{args.episodes}ep_{timestamp}.npy'
        np.save(model_name, agent.q_table)
        print(f"Model saved to {model_name}")

if __name__ == "__main__":
    main()
