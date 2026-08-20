from dataclasses import dataclass


@dataclass
class TrainingConfig:
    episodes: int = 10_000
    checkpoint_interval: int = 500
    log_interval: int = 10
