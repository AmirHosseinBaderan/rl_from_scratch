from pathlib import Path

import numpy as np

from agent import QLearningAgent
from environment import GridWorldConfig


class CheckpointManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.npz"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.npz"

    def save(
            self,
            agent: QLearningAgent,
            config: GridWorldConfig,
            episode: int,
            best_reward: float,
            path: str | Path,
    ):
        path = Path(path)

        obstacles = np.array(
            list(config.obstacles),
            dtype=np.int64,
        )

        np.savez(
            path,
            q_table=agent.q_table,
            epsilon=agent.epsilon,
            episode=episode,
            best_reward=best_reward,

            rows=config.rows,
            cols=config.cols,

            start=np.array(config.start),
            goal=np.array(config.goal),

            obstacles=obstacles,
        )

    def save_latest(
            self,
            agent: QLearningAgent,
            config: GridWorldConfig,
            episode: int,
            best_reward: float,
    ):
        self.save(
            agent=agent,
            config=config,
            episode=episode,
            best_reward=best_reward,
            path=self.latest_path,
        )

    def save_best(
            self,
            agent: QLearningAgent,
            config: GridWorldConfig,
            episode: int,
            best_reward: float,
    ):
        self.save(
            agent=agent,
            config=config,
            episode=episode,
            best_reward=best_reward,
            path=self.best_path,
        )

    def load(
            self,
            agent: QLearningAgent,
            path: str | Path,
    ) -> dict:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = np.load(
            path,
            allow_pickle=False,
        )

        agent.q_table = checkpoint["q_table"].copy()
        agent.epsilon = float(
            checkpoint["epsilon"]
        )

        obstacles = frozenset(
            tuple(position)
            for position in checkpoint["obstacles"]
        )

        config = GridWorldConfig(
            rows=int(checkpoint["rows"]),
            cols=int(checkpoint["cols"]),
            start=tuple(checkpoint["start"]),
            goal=tuple(checkpoint["goal"]),
            obstacles=obstacles,
        )

        return {
            "episode": int(checkpoint["episode"]),
            "best_reward": float(
                checkpoint["best_reward"]
            ),
            "config": config,
        }

    def load_latest(
            self,
            agent: QLearningAgent,
    ) -> dict:
        return self.load(
            agent=agent,
            path=self.latest_path,
        )

    def load_best(
            self,
            agent: QLearningAgent,
    ) -> dict:
        return self.load(
            agent=agent,
            path=self.best_path,
        )
