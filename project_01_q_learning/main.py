from pathlib import Path

from agent import QLearningAgent
from environment import GridWorld
from training import CheckpointManager, TrainingConfig, TensorBoardLogger, Trainer


def main():
    obstacles = {
        (0, 3),
        (1, 1),
        (1, 3),
        (2, 3),
        (3, 0),
    }

    env = GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=obstacles,
    )

    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    config = TrainingConfig(
        episodes=10_000,
        checkpoint_interval=500,
        log_interval=100,
    )

    checkpoint_manager = CheckpointManager(
        Path("checkpoints/q_learning")
    )

    tensor_logger = TensorBoardLogger(
        Path("runs/q_learning")
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        config=config,
        checkpoint_manager=checkpoint_manager,
        tensor_logger=tensor_logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()
