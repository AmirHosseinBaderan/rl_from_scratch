from dataclasses import dataclass

from .state import SnakeState


@dataclass(frozen=True)
class StepResult:
    state: SnakeState
    reward: int
    done: bool