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

    grid_input = GridInput(
        rows=5,
        cols=5,
    )

    start, goal = grid_input.get_start_and_goal(
        obstacles=set()
    )

    assert start == (0, 0)
    assert goal == (4, 4)

def test_start_cannot_be_obstacle(monkeypatch):
    values = iter([
        "1 1",
        "0 0",
        "4 4",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(values),
    )

    grid_input = GridInput(
        rows=5,
        cols=5,
    )

    start, goal = grid_input.get_start_and_goal(
        obstacles={(1, 1)}
    )

    assert start == (0, 0)
    assert goal == (4, 4)