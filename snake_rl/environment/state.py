from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnakeState:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (12,):
            raise ValueError(
                "SnakeState.values must have shape (12,)"
            )

    def as_array(self) -> np.ndarray:
        return self.values.copy()