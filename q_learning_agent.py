import numpy as np

class QLearningAgent:
    def __init__(self, state_space, action_space=4, seed=None):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_space: A tuple (rows, cols) representing the size of the state space.
            action_space: Number of possible actions (default 4).
            seed: Random seed for reproducibility.
        """
        self.rows, self.cols = state_space
        self.action_space = action_space
        self.rng = np.random.RandomState(seed)
        self.q_table = np.zeros((self.rows, self.cols, self.action_space))
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        state = (int(state[0]), int(state[1]))
        if self.rng.random() < self.exploration_rate:
            return self.rng.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state[0], state[1]])
            
    def learn(self, state, action, reward, next_state, done):
        state = (int(state[0]), int(state[1]))
        next_state = (int(next_state[0]), int(next_state[1]))
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = 0 if done else np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
    def get_best_path(self, environment):
        state, _ = environment.reset()
        state = (int(state[0]), int(state[1]))
        path = [state]
        done = False
        while not done:
            action = np.argmax(self.q_table[state[0], state[1]])
            next_state, _, done, _, _ = environment.step(action)
            next_state = (int(next_state[0]), int(next_state[1]))
            path.append(next_state)
            state = next_state
            if len(path) > environment.rows * environment.cols:
                break
        return path
