import random

import torch

from environment import Action


class EpsilonGreedySelector:
    def __init__(
        self,
        epsilon: float,
        action_size: int,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        if action_size <= 0:
            raise ValueError(
                "action_size must be greater than 0."
            )

        self.epsilon = epsilon
        self.action_size = action_size

    def select(
        self,
        q_values: torch.Tensor,
    ) -> Action:
        if q_values.ndim != 1:
            raise ValueError(
                "q_values must be a 1D tensor."
            )

        if q_values.shape[0] != self.action_size:
            raise ValueError(
                "q_values size must match action_size."
            )

        if random.random() < self.epsilon:
            return Action(
                random.randrange(self.action_size)
            )

        return Action(
            torch.argmax(q_values).item()
        )

    def decay(
            self,
            amount: float,
            minimum: float,
    ) -> None:
        if amount < 0.0:
            raise ValueError(
                "amount must be greater than or equal to 0."
            )

        if not 0.0 <= minimum <= 1.0:
            raise ValueError(
                "minimum must be between 0 and 1."
            )

        self.epsilon = max(
            minimum,
            self.epsilon - amount,
        )