import torch
import pytest

from agent import compute_loss


def test_compute_loss_returns_scalar():
    predicted = torch.tensor(
        [2.0, 5.0, 10.0]
    )

    target = torch.tensor(
        [3.0, 4.0, 7.0]
    )

    loss = compute_loss(
        predicted,
        target,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_loss_is_zero_for_equal_values():
    values = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    loss = compute_loss(
        values,
        values,
    )

    assert torch.isclose(
        loss,
        torch.tensor(0.0),
    )

def test_compute_loss_rejects_shape_mismatch():
    predicted = torch.tensor(
        [1.0, 2.0]
    )

    target = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(ValueError):
        compute_loss(
            predicted,
            target,
        )