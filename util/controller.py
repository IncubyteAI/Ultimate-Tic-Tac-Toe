import copy
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

if not 'server_mode' in locals():
  print("Running w/o Server")
  server_mode = False
else:
  print("Running in Server")
  
def pretty_print(board):
  fig, ax = plt.subplots()
  cmap = colors.ListedColormap(['white', 'red', 'blue'])
  bounds = [-0.5,0.5,1.5,2.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  ax.imshow(np.array(board).reshape((3,3)), cmap=cmap, norm=norm)
  # ax.imshow(np.array(board).reshape((3,3)), cmap=cmap)

  ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
  ax.set_xticks(np.arange(-.5, 3, 1))
  ax.set_yticks(np.arange(-.5, 3, 1))
  plt.show()
  
class Controller:
  def __init__(self, verbose: bool):
    self.board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    self.verbose = verbose
  def play(self, move: tuple[int, int], color):
    """attempt to play move defined by tuple (x, y)"""
    if self.board[move[0]][move[1]] != 0:
      if self.verbose:
        print("Illegal move detected.")
        print(f"AI {color} attempted to play a move at {move}")
      return False
    else:
      self.board[move[0]][move[1]] = color
      return True


  #This is the host function for checking if the game has ended
  def checkIfDone(self):
    """
    1->Player 1 wins
    2->Player 2 wins
    0->Draw
    -1->Unfinished
    """
    # check horizontal spaces
    for row in self.board:
        if row.count(row[0]) == len(row) and row[0] != 0:
            return row[0]

    # check vertical spaces
    for col in range(len(self.board[0])):
        check = []
        for row in self.board:
            check.append(row[col])
        if check.count(check[0]) == len(check) and check[0] != 0:
            return check[0]

    # check / diagonal spaces
    if self.board[0][0] != 0 and self.board[0][0] == self.board[1][1] == self.board[2][2]:
        return self.board[0][0]

    # check \ diagonal spaces
    if self.board[0][2] != 0 and self.board[0][2] == self.board[1][1] == self.board[2][0]:
        return self.board[0][2]

    # check if board is full (draw)
    if all(all(row) for row in self.board):
        return 0

    # if the game is not over, return -1
    return -1
  def show_board(self):
    pretty_print(self.board)
  def get_board(self):
     return copy.deepcopy(self.board)
