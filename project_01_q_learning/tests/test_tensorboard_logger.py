from pathlib import Path

from training import TensorBoardLogger


def test_tensorboard_logger():
    logdir = Path("./logs")
    logger = TensorBoardLogger(logdir)

    logger.log_episode(
        episode=1,
        reward=10.0,
        length=8,
        epsilon=0.9,
        success_rate=1.0,
    )

    logger.close()

    event_files = list(logdir.iterdir())

    assert len(event_files) > 0
