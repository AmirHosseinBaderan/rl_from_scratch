import torch
from torch import nn


def compute_loss(
        predicted_q_values: torch.Tensor,
        target_q_values: torch.Tensor,
) -> torch.Tensor:
    if predicted_q_values.shape != target_q_values.shape:
        raise ValueError(
            "Predicted and target Q-values"
            "must have same shape"
        )

    if predicted_q_values.ndim != 1:
        raise ValueError(
            "Q-values must be a 1D Tensor"
        )

    loss_function = nn.SmoothL1Loss()
    return loss_function(predicted_q_values, target_q_values)
