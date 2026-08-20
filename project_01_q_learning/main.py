import argparse
from pathlib import Path

from agent import QLearningAgent
from environment import GridWorld

from evaluation.evaluate import Evaluator

from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.tensorboard_logger import TensorBoardLogger
from training.train import Trainer
from visualization import GridWorldVisualizer

def create_environment():
    obstacles = {
        (0, 3),
        (1, 1),
        (1, 3),
        (2, 3),
        (3, 0),
    }

    return GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=obstacles,
    )


def create_agent():
    return QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )


def train():
    env = create_environment()
    agent = create_agent()

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


def evaluate():
    env = create_environment()
    agent = create_agent()

    checkpoint_manager = CheckpointManager(
        Path("checkpoints/q_learning")
    )

    checkpoint = checkpoint_manager.load_best(
        agent
    )

    evaluator = Evaluator(
        env=env,
        agent=agent,
    )

    result = evaluator.evaluate()

    print()
    print("=" * 50)
    print("Evaluation")
    print("=" * 50)

    print(f"Checkpoint Episode: {checkpoint['episode']}")
    print(f"Best Reward: {checkpoint['best_reward']:.2f}")
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Total Reward: {result.total_reward:.2f}")

    print()
    print("Learned Path:")

    for index, state in enumerate(result.path):
        if index == 0:
            print(f"START → {state}")
        else:
            print(f"       ↓ {state}")

    print()

    if result.success:
        print("GOAL REACHED 🎯")
    else:
        print("GOAL NOT REACHED ❌")

    visualizer = GridWorldVisualizer(env)
    visualizer.plot_path(result)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["train", "evaluate"],
    )

    args = parser.parse_args()

    if args.command == "train":
        train()

    elif args.command == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()