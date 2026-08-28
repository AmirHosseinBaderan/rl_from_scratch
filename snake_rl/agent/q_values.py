import torch


def select_action_q_values(
        q_values: torch.Tensor,
        actions: torch.Tensor,
) -> torch.Tensor:
    if q_values.ndim != 2:
        raise ValueError(
            "q_values must be a 2D tensor"
        )

    if actions.ndim != 1:
        raise ValueError(
            "actions must be a 1D tensor"
        )

    if q_values.shape[0] != actions.shape[0]:
        raise ValueError(
            "Number of actions must match batch size"
        )

    return q_values.gather(
        dim=1,
        index=actions.unsqueeze(1),
    ).squeeze()
