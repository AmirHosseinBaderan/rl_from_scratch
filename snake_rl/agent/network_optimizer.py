import torch
from torch import nn


class NetworkOptimizer:
    def __init__(
            self,
            network: nn.Module,
            learning_rate: float
    ):
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0.0")

        self.network = network
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate
        )

    def step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
