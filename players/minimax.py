import copy
class Board:
  def __init__(self, board=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]):
    self.board = board
  def move(self, pos, color):
    new_board = copy.deepcopy(self.board)
    new_board[pos[0]][pos[1]] = color
    return Board(new_board)
  def check_row(self, i):
    for j in range(3):
      if self.board[i][j] != self.board[i][0]:
        return 0
    return self.board[i][0]
  def check_col(self, j):
    for i in range(3):
      if self.board[i][j] != self.board[0][j]:
        return 0
    return self.board[0][j]
  def winner(self) -> int:
    for i in range(3):
      if self.check_row(i) > 0:
        return self.check_row(i)
      if self.check_col(i) > 0:
        return self.check_col(i)
    if self.board[0][0] != 0:
      works = True
      for i in range(3):
        if self.board[i][i] != self.board[0][0]:
          works = False
          break
      if works:
        return self.board[0][0]
    if self.board[0][2] != 0:
      works = True
      for i in range(3):
        if self.board[i][2 - i] != self.board[0][2]:
          works = False
          break
      if works:
        return self.board[0][2]
    for i in range(3):
      for j in range(3):
        if self.board[i][j] == 0:
          return 0
    return 3
  def __getitem__(self, index):
    return self.board[index]
  def __str__(self) -> str:
    return str(self.board)
  def __hash__(self):
    return hash(str(self.board))
  def __eq__(self, other):
    return self.board == other.board

class Minimax:
  def __init__(self):
    self.cache = {}
    self.next_move = {}
    self.search(Board(), 1)
  def search(self, state: Board, color) -> int:
    # print(state)
    if state in self.cache:
      return self.cache[state]
    elif state.winner() > 0:
      # print(state)
      self.next_move[state] = (-1, -1)
      W = state.winner()
      if W == 3:
        self.cache[state] = 0
      elif W == color:
        self.cache[state] = 1
      else:
        self.cache[state] = -1
      return self.cache[state]
    else:
      mn = 2
      next_color = 1 if color == 2 else 2
      for i in range(3):
        for j in range(3):
          if state[i][j] != 0:
            continue
          next_state = state.move((i, j), color)
          if self.search(next_state, next_color) < mn:
            mn = self.cache[next_state]
            self.next_move[state] = (i, j)
      self.cache[state] = -mn
      return self.cache[state]
  def move(self, board):
    return self.next_move[Board(board)]
    # ^ to use this with the central host, uncomment the line above and comment the line below
    # return self.next_move[board]
  def train(self, result, **kwargs):
    pass