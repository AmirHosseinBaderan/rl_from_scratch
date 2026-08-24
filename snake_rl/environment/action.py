from enum import IntEnum

from .direction import Direction


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def to_direction(action: Action) -> Direction:
    mapping = {
        Action.UP: Direction.UP,
        Action.DOWN: Direction.DOWN,
        Action.LEFT: Direction.LEFT,
        Action.RIGHT: Direction.RIGHT,
    }

    return mapping[action]