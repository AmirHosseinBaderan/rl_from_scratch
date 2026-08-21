from enum import Enum

import numpy as np


class Direction(Enum):
    UP = np.array([0, 1])
    DOWN = np.array([0, -1])
    LEFT = np.array([-1, 0])
    RIGHT = np.array([1, 0])

    @property
    def vector(self) -> np.ndarray:
        return self.value.copy()