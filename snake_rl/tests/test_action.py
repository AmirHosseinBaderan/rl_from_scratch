from environment import Action, Direction, to_direction


def test_action_values():
    assert Action.UP == 0
    assert Action.DOWN == 1
    assert Action.LEFT == 2
    assert Action.RIGHT == 3


def test_action_to_direction():
    assert to_direction(Action.UP) == Direction.UP
    assert to_direction(Action.DOWN) == Direction.DOWN
    assert to_direction(Action.LEFT) == Direction.LEFT
    assert to_direction(Action.RIGHT) == Direction.RIGHT