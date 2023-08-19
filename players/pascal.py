import torch
import torch.nn as nn
import torch.nn.functional as F
from util.result import Result

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.Softmax(dim=0)
        )
    def forward(self, x):
        return self.network(x)

class Pascal():
    def __init__(self):
        self.model = NN()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.probs = []
    def move(self, board:list[list[int]]):
        for i in range(3):
            for j in range(3):
                if board[i][j] == 2:
                    board[i][j] = -1
        output = self.model.forward(torch.tensor(board).to(torch.float32))
        M = torch.distributions.Categorical(output)
        move = M.sample()
        self.probs.append(M.log_prob(move))
        r = int(move.item())
        finalmove = (r // 3, r % 3)
        return finalmove
    def train(self, result: Result):
        reward = 0
        if result == Result.WIN:
            reward = 1
        elif result == Result.DRAW:
            reward = 0.5
        elif result == Result.LOSS:
            reward = -1
        else:
            reward = 0.3
            self.probs[-1]*-1.5
        loss = -reward*torch.stack(self.probs).sum()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.probs = torch.tensor(0)
