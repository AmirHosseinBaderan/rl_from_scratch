from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir)
        )

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        epsilon: float,
        success_rate: float,
    ):
        self.writer.add_scalar(
            "episode/reward",
            reward,
            episode,
        )

        self.writer.add_scalar(
            "episode/length",
            length,
            episode,
        )

        self.writer.add_scalar(
            "training/epsilon",
            epsilon,
            episode,
        )

        self.writer.add_scalar(
            "training/success_rate",
            success_rate,
            episode,
        )

    def close(self):
        self.writer.close()