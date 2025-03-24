import numpy as np

class QLearningAgent:
    def __init__(self, state_space, action_space=4):
        self.rows, self.cols = state_space
        self.action_space = action_space
        self.q_table = np.zeros((self.rows, self.cols, self.action_space))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
    def choose_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state])
            
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)
    
    def get_best_path(self, environment):
        state = environment.reset()
        path = [state]
        done = False
        while not done:
            action = np.argmax(self.q_table[state])
            next_state, _, done, _ = environment.step(action)
            path.append(next_state)
            state = next_state
            if len(path) > environment.rows * environment.cols:
                break
        return path
