from dataclasses import dataclass

import numpy as np
import torch

from .experience import Experience


@dataclass(frozen=True)
class ExperienceBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


def create_batch(
    experiences: list[Experience],
) -> ExperienceBatch:
    if not experiences:
        raise ValueError(
            "experiences cannot be empty."
        )

    states = torch.tensor(
        np.stack(
            [experience.state for experience in experiences]
        ),
        dtype=torch.float32,
    )

    actions = torch.tensor(
        [
            int(experience.action)
            for experience in experiences
        ],
        dtype=torch.long,
    )

    rewards = torch.tensor(
        [
            experience.reward
            for experience in experiences
        ],
        dtype=torch.float32,
    )

    next_states = torch.tensor(
        np.stack(
            [
                experience.next_state
                for experience in experiences
            ]
        ),
        dtype=torch.float32,
    )

    dones = torch.tensor(
        [
            experience.done
            for experience in experiences
        ],
        dtype=torch.bool,
    )

    return ExperienceBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )