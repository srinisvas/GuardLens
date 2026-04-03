import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckFFNN(nn.Module):

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.pre = nn.Linear(input_dim, 256)

        self.b1 = nn.Linear(256, 128)
        self.b2 = nn.Linear(128, 64)
        self.b3 = nn.Linear(64, 128)

        self.post = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop(F.relu(self.pre(x))) # 768 -> 256
        x2 = self.drop(F.relu(self.b1(x1))) # 256 -> 128
        x3 = self.drop(F.relu(self.b2(x2))) #
        x4 = self.drop(F.relu(self.b3(x3))) # 64 -> 128
        x5 = self.drop(F.relu(self.post(x4))) # 128 -> 64
        return self.out(x5)

class ParallelMultiPathFFNN(nn.Module):

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.pre = nn.Linear(input_dim, 256)
        self.path_a = nn.Linear(256, 128)
        self.path_b1 = nn.Linear(256, 128)
        self.path_b2 = nn.Linear(128, 64)
        self.path_b3 = nn.Linear(64, 128)

        self.fusion = nn.Linear(256, 128)
        self.post = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop(F.relu(self.pre(x))) # 768 -> 256
        x2 = self.drop(F.relu(self.path_a(x1))) # 256 -> 128
        x3 = self.drop(F.relu(self.path_b1(x1))) # 256 -> 128
        x4 = self.drop(F.relu(self.path_b2(x3))) # 128 -> 64
        x5 = self.drop(F.relu(self.path_b3(x4))) # 64 -> 128

        fused = torch.cat([x2, x5], dim=-1) # 128 + 128 -> 256
        fused = self.drop(F.relu(self.fusion(fused))) # 256 -> 128
        fused = self.drop(F.relu(self.post(fused))) # 128 -> 64

        return self.out(fused)


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

        x1 = self.drop(F.relu(self.fc1(x))) # 768 -> 256
        x2 = self.drop(F.relu(self.fc2(x1) + self.skip(x1))) # 256 -> 128 with skip connection
        x3 = self.drop(F.relu(self.fc3(x2))) # 128 -> 64

        return self.out(x3)

