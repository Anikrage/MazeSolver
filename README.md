Maze Reinforcement Learning Project
Overview

This project implements a Q-learning algorithm to solve procedurally generated mazes. It includes maze generation, a reinforcement learning environment, a Q-learning agent, and visualization tools to monitor the learning process and results.
Features

    Procedural maze generation using Depth-First Search algorithm

    Q-learning implementation for path-finding

    Real-time visualization of the learning process

    Comprehensive training metrics and visualizations

    Flexible command-line interface for customization

Requirements

    Python 3.7+

    NumPy

    Matplotlib

    TensorBoard (optional, for advanced logging)

Installation

    Clone the repository:

text

Install the required packages:

    text
    pip install numpy matplotlib tensorboard

Usage
Basic Run

To run the project with default settings:

text
python main.py

Command-line Options

    --maze_size: Size of the maze (default: 10)

    --episodes: Number of training episodes (default: 1000)

    --visualize: Enable real-time visualization

    --update_interval: Visualization update interval in episodes (default: 5)

    --save_model: Save the trained Q-table

    --seed: Set random seed for reproducibility (default: 42)

Examples

    Run with real-time visualization:

text
python main.py --visualize

Train on a larger maze for more episodes:

text
python main.py --maze_size 15 --episodes 2000 --visualize

Faster visualization updates:

text
python main.py --visualize --update_interval 2

Train and save the model:

    text
    python main.py --episodes 5000 --save_model

Project Structure

    main.py: Main script to run the project

    maze_generator.py: Maze generation algorithms

    maze_environment.py: Reinforcement learning environment

    q_learning_agent.py: Q-learning agent implementation

    trainer.py: Training loop and logic

    visualizer.py: Visualization tools for maze, path, and metrics

Output

The program will display:

    The initially generated maze

    Real-time training visualization (if enabled)

    Training summary statistics

    Training history plots

    Best path found in the maze

    Q-values heatmap

Customization

You can modify hyperparameters like learning rate, discount factor, and exploration rate in the q_learning_agent.py file.
Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
License

This project is licensed under the MIT License - see the LICENSE file for details.