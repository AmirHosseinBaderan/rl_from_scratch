from pathlib import Path

import numpy as np

from agent import QLearningAgent


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
            episode: int,
            best_reward: float,
            path: str | Path,
    ):
        path = Path(path)

        np.savez(
            path,
            q_table=agent.q_table,
            epsilon=agent.epsilon,
            episode=episode,
            best_reward=best_reward,
        )

    def save_latest(
            self,
            agent: QLearningAgent,
            episode: int,
            best_reward: float,
    ):
        self.save(
            agent=agent,
            episode=episode,
            best_reward=best_reward,
            path=self.latest_path,
        )

    def save_best(
            self,
            agent: QLearningAgent,
            episode: int,
            best_reward: float,
    ):
        self.save(
            agent=agent,
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

        checkpoint = np.load(path)

        agent.q_table = checkpoint["q_table"].copy()
        agent.epsilon = float(checkpoint["epsilon"])

        return {
            "episode": int(checkpoint["episode"]),
            "best_reward": float(checkpoint["best_reward"]),
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
