import torch

from agent import NetworkOptimizer


def test_optimizer_updates_network_parameters():
    network = torch.nn.Linear(1, 1)

    optimizer = NetworkOptimizer(
        network=network,
        learning_rate=0.01,
    )

    before = network.weight.detach().clone()

    input_value = torch.tensor([[2.0]])
    target = torch.tensor([[10.0]])

    prediction = network(input_value)

    loss = torch.nn.functional.mse_loss(
        prediction,
        target,
    )

    optimizer.step(loss)

    after = network.weight.detach().clone()

    assert not torch.equal(
        before,
        after,
    )

def test_optimizer_rejects_invalid_learning_rate():
    network = torch.nn.Linear(1, 1)

    try:
        NetworkOptimizer(
            network=network,
            learning_rate=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_optimizer_reduces_loss():
    network = torch.nn.Linear(1, 1)

    optimizer = NetworkOptimizer(
        network=network,
        learning_rate=0.01,
    )

    input_value = torch.tensor([[2.0]])
    target = torch.tensor([[10.0]])

    prediction = network(input_value)

    initial_loss = torch.nn.functional.mse_loss(
        prediction,
        target,
    )

    for _ in range(100):
        prediction = network(input_value)

        loss = torch.nn.functional.mse_loss(
            prediction,
            target,
        )

        optimizer.step(loss)

    prediction = network(input_value)

    final_loss = torch.nn.functional.mse_loss(
        prediction,
        target,
    )

    assert final_loss < initial_loss