import torch

from agent import QNetwork


def test_q_network_output_shape():
    network = QNetwork(
        state_size=12,
        action_size=4,
    )

    state = torch.zeros(1, 12)

    q_values = network(state)

    assert q_values.shape == (1, 4)

def test_q_network_supports_batch():
    network = QNetwork(
        state_size=12,
        action_size=4,
    )

    states = torch.zeros(32, 12)

    q_values = network(states)

    assert q_values.shape == (32, 4)