from collections import deque
import random

from .exprience import Experience


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than 0."
            )

        self.capacity = capacity
        self._buffer: deque[Experience] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if batch_size > len(self._buffer):
            raise ValueError(
                "batch_size cannot be greater than "
                "the number of stored experiences."
            )

        return random.sample(
            self._buffer,
            batch_size,
        )