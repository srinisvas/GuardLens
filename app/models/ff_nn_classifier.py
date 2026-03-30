import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNNClassifier(nn.Module):

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()

        self.fc1  = nn.Linear(input_dim, 256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, 64)
        self.skip = nn.Linear(256, 128)
        self.out  = nn.Linear(64, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop(F.relu(self.fc1(x)))

        x2 = self.drop(F.relu(self.fc2(x1) + self.skip(x1)))

        x3 = self.drop(F.relu(self.fc3(x2)))

        return self.out(x3)

