import torch
from torch import autograd
from torch.optim import Adam
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from util import Result
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
    def __init__(self, board_length, gamma, lr, passive, draw, illegal):
        self.board_length = board_length
        self.board_size = board_length ** 2
        # self.device = torch.device("cpu")
        self.nn = NN(self.board_size)
        # self.nn.to(self.device)
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
        self.c = 0
        self.entropies = []
    def get_action(self, num):
        return (num // self.board_length, num % self.board_length)
    def move(self, state: list[list[int]]):
        # print(state)
        for i in range(self.board_length):
            for j in range(self.board_length):
                if state[i][j] == 2:
                    state[i][j] = -1
        state = torch.tensor(state).flatten().to(torch.float32)# .to(self.device)
        if self.training:
            probs = self.nn(state) * (1 - self.er) + self.er / self.board_size
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
            return self.draw
        elif result == Result.LOSS:
            return -self.illegal
        elif result == Result.ILLEGAL:
            return -self.illegal
        else:
            return self.draw
    def train(self, result: Result):
        self.rewards[-1] = self.score(result)
        # if self.score(result) < 0:
        #     self.bad.append(result)
        #     if len(self.bad) > self.max_bad:
        #         self.nn = NN(self.board_size)
        #         self.optimizer = Adam(self.nn.parameters(), lr=0.001)
        #         del self.bad[:]
        # else:
        #     self.bad = []
        self.c += 1
        if self.c % 5000 == 0:
            print(self.rewards, self.mx)
            print(result, self.score(result))
            # print(self.log_probs)
        loss = torch.tensor(0.0)
        G = deque()
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            G.appendleft(R)

        # self.cur_R.append(R)
        # if len(self.cur_R) == self.avg_len:
        #     self.R.append(sum(self.cur_R) / self.avg_len)
        #     del self.cur_R[:]
        G = torch.tensor(G)
        for log_prob, R in zip(self.log_probs, G):
            loss -= log_prob * R
        loss -= torch.stack(self.entropies).mean()
        self.batch.append(loss)
        if len(self.batch) == self.batch_size:
            batch_loss = torch.stack(self.batch).mean()
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del self.batch[:]
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        del self.log_probs[:]
        del self.rewards[:]
        del self.entropies[:]
        # self.loss = torch.tensor(0.0, device=self.device)