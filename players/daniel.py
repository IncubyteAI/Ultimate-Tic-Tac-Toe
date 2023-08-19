import torch
from torch.optim import Adam
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
from util.result import Result
class NN(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, board_size),
            nn.Softmax(dim=0),
        )
    def forward(self, x):
        return self.layers(x)
class Player:
    def __init__(self, board_length=3, gamma=1, lr=0.001, passive=0.3, draw=1, illegal=0.3, entropy_weight=0.75, verbose=False):
        self.board_length = board_length
        self.board_size = board_length ** 2
        self.nn = NN(self.board_size)
        self.training = True
        self.gamma = gamma
        self.lr = lr
        self.optimizer = Adam(self.nn.parameters(), lr=self.lr)
        self.log_probs = []
        self.rewards = []
        self.passive = passive
        self.draw = draw
        self.illegal = illegal
        self.batch = []
        self.batch_size = 1
        self.entropies = []
        self.entropy_weight = entropy_weight
        self.verbose = verbose
    def get_action(self, num):
        return (num // self.board_length, num % self.board_length)
    def move(self, state: list[list[int]]):
        for i in range(self.board_length):
            for j in range(self.board_length):
                if state[i][j] == 2:
                    state[i][j] = -1
        state = torch.tensor(state).flatten().to(torch.float32) # .to(self.device)
        if self.training:
            probs = self.nn(state) # self.nn.forward(state)
            self.entropies.append(-torch.sum(probs * torch.log(probs)))
            m = Categorical(probs)
            action_num = m.sample()
            self.log_probs.append(m.log_prob(action_num))
            self.rewards.append(self.passive)
            return self.get_action(int(action_num.item()))
        else:
            return self.get_action(int(torch.argmax(self.nn(state)).item()))
    def score(self, result: Result):
        if result == Result.WIN:
            return 1.5 * self.draw
        elif result == Result.LOSS:
            return -1
        elif result == Result.ILLEGAL:
            return -self.illegal
        else:
            return self.draw
    def train(self, result: Result, game_num: int):
        self.rewards[-1] = self.score(result)
        if game_num % 2000 == 0 and self.verbose:
            print(self.rewards)
            print(result, self.score(result))
        loss = torch.tensor(0.0)
        G = deque()
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            G.appendleft(R)
        G = torch.tensor(G)
        for log_prob, R in zip(self.log_probs, G):
            loss -= log_prob * R
        if game_num % 5000 == 0 and self.verbose:
            print(loss.item(), torch.stack(self.entropies).mean().item())
        loss -= torch.stack(self.entropies).mean() * self.entropy_weight
        self.batch.append(loss)
        if len(self.batch) == self.batch_size:
            batch_loss = torch.stack(self.batch).mean()
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del self.batch[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.entropies[:]