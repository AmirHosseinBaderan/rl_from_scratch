import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)