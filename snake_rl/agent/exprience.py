from dataclasses import dataclass

import numpy as np

from environment import Action


@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool