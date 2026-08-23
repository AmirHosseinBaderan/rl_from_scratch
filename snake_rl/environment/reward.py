from enum import IntEnum


class Reward(IntEnum):
    FOOD = 10
    COLLISION = -10
    STEP = -1