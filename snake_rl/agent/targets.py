import torch


def compute_bellman_target(
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
) -> torch.Tensor:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between 0.0 and 1.0"
        )

    max_next_q_values = next_q_values.max(
        dim=1
    ).values

    targets = rewards + (
            gamma
            * max_next_q_values
            * (~dones)
    )

    return targets
