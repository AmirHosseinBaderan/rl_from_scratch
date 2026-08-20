from ..training import TensorBoardLogger


def test_tensorboard_logger(tmp_path):
    logger = TensorBoardLogger(tmp_path)

    logger.log_episode(
        episode=1,
        reward=10.0,
        length=8,
        epsilon=0.9,
        success_rate=1.0,
    )

    logger.close()

    event_files = list(tmp_path.iterdir())

    assert len(event_files) > 0
