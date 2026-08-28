from pathlib import Path

from training import TensorBoardLogger


def test_logger_creates_log_directory(tmp_path: Path):
    log_dir = tmp_path / "runs"

    logger = TensorBoardLogger(log_dir)

    assert log_dir.exists()

    logger.close()

def test_logger_writes_scalar(tmp_path: Path):
    log_dir = tmp_path / "runs"

    logger = TensorBoardLogger(log_dir)

    logger.log_scalar(
        tag="training/reward",
        value=10.0,
        step=1,
    )

    logger.close()

    assert any(log_dir.iterdir())