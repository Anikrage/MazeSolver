# Maze Reinforcement Learning Project

## Overview
This project implements a Q-learning algorithm to solve procedurally generated mazes. It features dynamic maze generation with a different maze generated each time, reinforcement learning for path-finding, and comprehensive visualization tools to monitor the learning process.

## Features
- **Procedural maze generation** using Depth-First Search algorithm (different maze each run)
- Q-learning implementation for path-finding
- Real-time visualization of the learning process
- Comprehensive training metrics and visualizations
- Flexible command-line interface for customization

## Requirements
- Python 3.7+
- NumPy
- Matplotlib

## Project Structure
- `main.py`: Main script to run the project
- `maze_generator.py`: Maze generation algorithms
- `maze_environment.py`: Reinforcement learning environment
- `q_learning_agent.py`: Q-learning agent implementation
- `trainer.py`: Training loop and logic
- `visualizer.py`: Visualization tools for maze, path, and metrics


### Command-line Options
- `--maze_size`: Size of the maze (default: 10)
- `--episodes`: Number of training episodes (default: 1000)
- `--visualize`: Enable real-time visualization
- `--update_interval`: Visualization update interval in episodes (default: 5)
- `--save_model`: Save the trained Q-table
- `--seed`: Set random seed for reproducibility (default: None, generates different maze each time)

## Maze Generation
By default, the program generates a different maze each time it runs. This is achieved using:
- Dynamic random seeding
- Depth-First Search (DFS) with randomized exploration order
- Random starting points for maze generation

If you want to reproduce the same maze for multiple runs (e.g., for comparing different learning parameters), use the `--seed` option with a specific value.

## Output
The program will display:
1. The initially generated maze
2. Real-time training visualization (if enabled)
3. Training summary statistics
4. Training history plots
5. Best path found in the maze
6. Q-values heatmap

## How It Works

### Maze Generation
The maze is generated using a randomized depth-first search algorithm, which creates a perfect maze (one with no loops or isolated walls).

### Reinforcement Learning
The agent uses Q-learning to find the optimal path through the maze:
- **States**: Cell positions in the maze
- **Actions**: Up, right, down, left
- **Rewards**: -1 for each step, +100 for reaching the goal, -100 for timeout
- **Q-learning parameters**: Learning rate = 0.1, Discount factor = 0.99

### Visualization
The visualization shows:
- The maze structure (white = paths, black = walls)
- Red arrows indicating the learned policy (best actions)
- Blue line showing the current best path
- Start (green) and goal (blue) positions

## Customization
You can modify hyperparameters like learning rate, discount factor, and exploration rate in the `q_learning_agent.py` file.

## License
This project is licensed under the MIT License - see the LICENSE file for details.