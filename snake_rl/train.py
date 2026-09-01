from agent import DQNAgent
from environment import SnakeEnvironment
from training import TensorBoardLogger, Trainer


def main():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    agent = DQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        replay_capacity=10_000,
    )

    tensor_logger = TensorBoardLogger(
        "runs/snake_dqn"
    )

    trainer = Trainer(
        environment=environment,
        agent=agent,
        episodes=1_000,
        batch_size=32,
        epsilon_decay=0.001,
        minimum_epsilon=0.1,
        logger=tensor_logger
    )

    trainer.train()
    tensor_logger.close()

if __name__ == "__main__":
    main()
