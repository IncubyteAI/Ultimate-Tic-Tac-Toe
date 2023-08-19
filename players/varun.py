import random

class TicTacVarun(ModelWrapper):
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

        # Custom rewards for different outcomes
        self.reward_win = 5
        self.reward_draw = 2
        self.reward_loss = -2
        self.reward_illegal = -10

        self.q_values = {}

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, next_action) for next_action in [(i, j) for i in range(3) for j in range(3)])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_values[(state, action)] = new_q

    def check_winner(self, player):
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
        for j in range(3):
            if all(self.board[i][j] == player for i in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)) or all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def ai_move_random(self):
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        return row, col

    def ai_move(self):
      current_state = tuple([tuple(row) for row in self.board])
      possible_moves = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == " "]
    
      if not possible_moves:
        return random.randint(0, 2), random.randint(0, 2)  # If no available moves, choose a random move
    
      q_values_for_moves = [self.get_q_value(current_state, move) for move in possible_moves]
      max_q_value = max(q_values_for_moves)
      best_moves = [move for move, q_value in zip(possible_moves, q_values_for_moves) if q_value == max_q_value]
      return random.choice(best_moves)

    def play_and_train(self):
        player = "X"
        ai = self
        current_turn = ai
        while True:
            if current_turn == player:
                row, col = self.ai_move_random()
            else:
                if random.uniform(0, 1) < self.exploration_rate:
                    row, col = self.ai_move_random()
                else:
                    row, col = self.ai_move()
            if self.board[row][col] != " ":
                continue

            self.board[row][col] = current_turn

            if self.check_winner(current_turn):
                if current_turn == player:
                    self.update_q_value(tuple([tuple(row) for row in self.board]), (row, col), self.reward_loss, None)
                else:
                    self.update_q_value(tuple([tuple(row) for row in self.board]), (row, col), self.reward_win, None)
                break
            elif all(self.board[i][j] != " " for i in range(3) for j in range(3)):
                self.update_q_value(tuple([tuple(row) for row in self.board]), (row, col), self.reward_draw, None)
                break

            current_turn = player if current_turn == ai else ai

    def move(self, board):
        self.board = board  # Update the internal board
        return self.ai_move()  # Return the AI's move

    def train(self):
        num_episodes = 0
        for episode in range(num_episodes):
            for i in range(3):
                for j in range(3):
                    self.board[i][j] = " "
            self.play_and_train()

# Create an instance of TicTacVarun and train the model
model = TicTacVarun()
model.train()

