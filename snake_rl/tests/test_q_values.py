import torch

from agent import select_action_q_values


def test_select_action_q_values():
    q_values = torch.tensor(
        [
            [1.0, 5.0, 2.0, 3.0],
            [4.0, 2.0, 8.0, 1.0],
            [7.0, 3.0, 1.0, 6.0],
        ]
    )

    actions = torch.tensor(
        [1, 2, 3]
    )

    selected = select_action_q_values(
        q_values=q_values,
        actions=actions,
    )

    expected = torch.tensor(
        [5.0, 8.0, 6.0]
    )

    assert torch.equal(
        selected,
        expected,
    )

def test_select_action_q_values_rejects_invalid_q_values():
    q_values = torch.zeros(4)

    actions = torch.tensor([0])

    try:
        select_action_q_values(
            q_values=q_values,
            actions=actions,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_select_action_q_values_rejects_batch_mismatch():
    q_values = torch.zeros(2, 4)

    actions = torch.tensor([0])

    try:
        select_action_q_values(
            q_values=q_values,
            actions=actions,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )