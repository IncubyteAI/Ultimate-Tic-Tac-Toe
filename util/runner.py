from controller import Controller
from util import Result
def run_episode(verbose: bool, first, second, train=True)->int:
  cnt = 0
  controller = Controller(verbose=verbose)
  # illegal = False
  # result = 0
  p1 = Result.NONE
  p2 = Result.NONE
  while controller.checkIfDone() == -1:
    first_move = first.move(controller.get_board())
    if controller.play(first_move, 1) == False:
      p1 = Result.ILLEGAL
      p2 = Result.WIN
      break
    else:
      cnt += 1
    if controller.checkIfDone() == -1:
      second_move = second.move(controller.get_board())
      if controller.play(second_move, 2) == False:
        p1 = Result.WIN
        p2 = Result.ILLEGAL
        break
      else:
        cnt += 1
  if verbose:
    controller.show_board()
  if p1 == Result.NONE:
    winner = controller.checkIfDone()
    if winner == 1:
      p1 = Result.WIN
      p2 = Result.LOSS
    elif winner == 2:
      p1 = Result.LOSS
      p2 = Result.WIN
    else:
      p1 = Result.DRAW
      p2 = Result.DRAW   
  if p1 == Result.DRAW:
    first.points += 1
    second.points += 1
    if verbose:
      print("game was drawn")
  elif p1 == Result.WIN:
    first.points += 2
    if verbose:
      print("player 1 won")
  elif p2 == Result.WIN:
    second.points += 2
    if verbose:
      print("player 2 won")
  first.train(p1)
  second.train(p2)
  return cnt