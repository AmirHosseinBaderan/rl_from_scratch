from dataclasses import dataclass
from agent import DQNAgent
from environment import SnakeEnvironment

from .episode import EpisodeResult, run_episode


@dataclass(frozen=True)
class TrainingResult:
    episodes: int
    rewards: list[float]
    steps: list[int]

class Trainer:
    def __init__(
            self,
            environment: SnakeEnvironment,
            agent: DQNAgent,
            episodes: int,
            batch_size: int,
    ):
        if episodes <= 0:
            raise ValueError(
                "episodes mist be greater than -"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size mist be greater than 0"
            )

        self.environment = environment
        self.agent = agent
        self.episodes = episodes
        self.batch_size = batch_size

    def train(self) -> TrainingResult:
        rewards: list[float] = []
        steps: list[int] = []

        for _ in range(self.episodes):
            result: EpisodeResult = run_episode(
                environment=self.environment,
                agent=self.agent,
                batch_size=self.batch_size,
            )

            rewards.append(result.total_reward)
            steps.append(result.steps)

        return TrainingResult(
            episodes=self.episodes,
            rewards=rewards,
            steps=steps,
        )
