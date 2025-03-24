import numpy as np

class QLearningAgent:
    def __init__(self, state_space, action_space=4):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_space: A tuple (rows, cols) representing the size of the state space.
            action_space: The number of possible actions (default is 4: up, right, down, left).
        """
        self.rows, self.cols = state_space
        self.action_space = action_space
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.rows, self.cols, self.action_space))
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state (position) in the maze.
        
        Returns:
            The chosen action.
        """
        # Epsilon-greedy policy
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state])
            
    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the observed transition.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is done.
        """
        # Update Q-value using the Q-learning formula
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
            
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def decay_exploration(self):
        """
        Decay the exploration rate.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)
    
    def get_best_path(self, environment):
        """
        Get the best path from start to goal based on the learned Q-table.
        
        Args:
            environment: The maze environment.
            
        Returns:
            A list of positions representing the path.
        """
        state = environment.reset()
        path = [state]
        done = False
        
        while not done:
            action = np.argmax(self.q_table[state])
            next_state, _, done, _ = environment.step(action)
            path.append(next_state)
            state = next_state
            
            # Prevent infinite loops
            if len(path) > environment.rows * environment.cols:
                break
                
        return path
