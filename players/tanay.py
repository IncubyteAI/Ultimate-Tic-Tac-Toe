import random
from util.result import Result
class TicTacTanay:
  def __init__(self, alpha=0.9, gamma=0.9, epsilon=0.1):
    self.alpha = alpha  # learning rate
    self.gamma = gamma  # discount factor
    self.epsilon = epsilon  # exploration rate
    self.q_table = {}  # initialize Q-table
    self.last_board = None
    self.last_move = None
    self.points = 0

  def get_q(self, state, action):
    # return Q value for given state-action pair
    return self.q_table.get((str(state), action), 0.6)

  def get_move(self, board):
    # exploration
    if random.random() < self.epsilon:
      move = (random.randint(0, 2), random.randint(0, 2))
    # else exploitation
    else:
      q_values = {move: self.get_q(board, move) for move in [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]}
      move = max(q_values, key=q_values.get)

    self.last_board = board
    self.last_move = move
    return move

  def move(self, board):
    return self.get_move(board)

  def train(self, result):
    if result == Result.WIN:
      final_reward = 1
    elif result == Result.LOSS:
      final_reward = -1
    elif result == Result.DRAW:
      final_reward = 0.5
    elif result == Result.ILLEGAL:
      final_reward = -5

    if self.last_board is not None and self.last_move is not None:
      old_q = self.get_q(self.last_board, self.last_move)
      max_future_q = max([self.get_q(self.last_board, move) for move in [(i, j) for i in range(3) for j in range(3) if self.last_board[i][j] == 0]], default=0)

      new_q = old_q + self.alpha * (final_reward + self.gamma * max_future_q - old_q)
      self.q_table[(str(self.last_board), self.last_move)] = new_q