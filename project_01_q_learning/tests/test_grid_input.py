from environment import GridWorldConfig
from input import GridInput


def test_get_start_and_goal(monkeypatch):
    values = iter([
        "0 0",
        "4 4",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(values),
    )

    config = GridWorldConfig(
        rows=5,
        cols=5,
    )

    grid_input = GridInput(
        config=config,
    )

    result = grid_input.get_start_and_goal(
        obstacles=set()
    )

    assert result.start == (0, 0)
    assert result.goal == (4, 4)

def test_start_cannot_be_obstacle(monkeypatch):
    values = iter([
        "1 1",
        "0 0",
        "0 0",
        "4 4",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(values),
    )

    config = GridWorldConfig(
        rows=5,
        cols=5,
    )

    grid_input = GridInput(
        config=config,
    )

    result = grid_input.get_start_and_goal(
        obstacles={(1, 1)}
    )

    assert result.start == (0, 0)
    assert result.goal == (4, 4)
