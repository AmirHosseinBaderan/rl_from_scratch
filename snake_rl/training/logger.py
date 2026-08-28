from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir)
        )

    def log_scalar(
            self,
            tag:str,
            value:float,
            step:int
    ):
        self.writer.add_scalar(
            tag,
            value,
            step
        )

    def close(self):
        self.writer.close()
