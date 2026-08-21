from dataclasses import dataclass

import numpy as np


@dataclass
class Food:
    position: np.ndarray

    def __post_init__(self) -> None:
        if self.position.shape != (2,):
            raise ValueError("Food position must have shape (2,).")