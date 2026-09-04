import torch

from agent import QNetwork, TargetNetwork


def test_target_network_has_same_architecture():
    online_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    target_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    assert type(online_network) is type(target_network)


def test_target_network_can_copy_online_weights():
    online_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    target_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    target_network.load_state_dict(
        online_network.state_dict()
    )

    for online_parameter, target_parameter in zip(
            online_network.parameters(),
            target_network.parameters(),
    ):
        assert torch.equal(
            online_parameter,
            target_parameter,
        )

def test_target_network_is_independent_from_online_network():
    online_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    target_network = TargetNetwork(
        state_size=12,
        action_size=4,
    )

    target_network.copy_from(online_network)

    with torch.no_grad():
        for parameter in online_network.parameters():
            parameter.add_(1.0)

    for online_parameter, target_parameter in zip(
            online_network.parameters(),
            target_network.network.parameters(),
    ):
        assert not torch.equal(
            online_parameter,
            target_parameter,
        )

def test_target_network_can_copy_from_online_network():
    online_network = QNetwork(
        state_size=12,
        action_size=4,
    )

    target_network = TargetNetwork(
        state_size=12,
        action_size=4,
    )

    target_network.copy_from(online_network)

    for online_parameter, target_parameter in zip(
            online_network.parameters(),
            target_network.network.parameters(),
    ):
        assert torch.equal(
            online_parameter,
            target_parameter,
        )