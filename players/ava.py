import numpy as np
class QLearning:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration factor

        # Initialize the Q-table to small random values
        self.q_table = np.random.uniform(low=-1, high=1, size=(3, 3, 3, 3, 3, 3, 3, 3, 3, 9))

        # Initialize the last state and last action to None
        self.last_state = None
        self.last_action = None

    def move(self, board):
        # Flatten the board to a 1D array
        state = np.array(board).flatten()

        # Choose an action
        if np.random.rand() < self.epsilon:  # exploration
            action = np.random.choice(9)
        else:  # exploitation
            action = np.argmax(self.q_table[tuple(state)])

        # Remember the state and action
        self.last_state = state
        self.last_action = action

        # Return the action as a tuple
        return divmod(action, 3)

    def train(self, reward, **kwargs):
        # Get the current state and Q-value
        state = self.last_state
        action = self.last_action
        q_value = self.q_table[tuple(state) + (action,)]

        # Calculate the target Q-value
        target_q_value = reward
        if reward == 0:  # game not over
            target_q_value = reward + self.gamma * np.max(self.q_table[tuple(state)])

        # Update the Q-value
        self.q_table[tuple(state) + (action,)] = (1 - self.alpha) * q_value + self.alpha * target_q_value

    def reset(self):
        # Reset the last state and last action
        self.last_state = None
        self.last_action = None