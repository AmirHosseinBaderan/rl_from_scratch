import argparse
import random
from pathlib import Path

from agent import QLearningAgent
from environment import GridWorld, GridWorldConfig
from evaluation.evaluate import Evaluator
from input import GridInput
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.tensorboard_logger import TensorBoardLogger
from training.train import Trainer
from visualization import GridWorldVisualizer


ROWS = 5
COLS = 5

OBSTACLES = frozenset({
    (0, 3),
    (1, 1),
    (1, 3),
    (2, 3),
    (3, 0),
})

CHECKPOINT_DIR = Path("checkpoints/q_learning")
TENSORBOARD_DIR = Path("runs/q_learning")


def create_environment() -> GridWorld:
    grid_input = GridInput(
        rows=ROWS,
        cols=COLS,
    )

    start, goal = grid_input.get_start_and_goal(
        obstacles=OBSTACLES,
    )

    config = GridWorldConfig(
        rows=ROWS,
        cols=COLS,
        start=start,
        goal=goal,
        obstacles=OBSTACLES,
    )

    return GridWorld(config)


def create_agent() -> QLearningAgent:
    return QLearningAgent(
        rows=ROWS,
        cols=COLS,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )


def train() -> None:
    agent = create_agent()

    config = TrainingConfig(
        episodes=10_000,
        checkpoint_interval=500,
        log_interval=100,
    )

    checkpoint_manager = CheckpointManager(
        CHECKPOINT_DIR
    )

    tensor_logger = TensorBoardLogger(
        TENSORBOARD_DIR
    )

    trainer = Trainer(
        rows=ROWS,
        cols=COLS,
        obstacles=OBSTACLES,
        agent=agent,
        config=config,
        checkpoint_manager=checkpoint_manager,
        tensor_logger=tensor_logger,
    )

    trainer.train()


def evaluate(start: tuple[int, int] | None = None, goal: tuple[int, int] | None = None) -> None:
    agent = create_agent()

    checkpoint_manager = CheckpointManager(
        CHECKPOINT_DIR
    )

    checkpoint = checkpoint_manager.load_best(
        agent
    )

    # Use evaluate approach: pick start/goal from args or random
    valid_positions = []
    for row in range(ROWS):
        for col in range(COLS):
            if (row, col) not in OBSTACLES:
                valid_positions.append((row, col))

    if start is None:
        start = random.choice(valid_positions)

    if goal is None:
        goal = random.choice(valid_positions)
        while goal == start:
            goal = random.choice(valid_positions)

    config = GridWorldConfig(
        rows=ROWS,
        cols=COLS,
        start=start,
        goal=goal,
        obstacles=OBSTACLES,
    )

    env = GridWorld(config)

    evaluator = Evaluator(
        env=env,
        agent=agent,
    )

    result = evaluator.evaluate()

    print()
    print("=" * 50)
    print("Evaluation")
    print("=" * 50)

    print(
        f"Checkpoint Episode: "
        f"{checkpoint['episode']}"
    )

    print(
        f"Best Reward: "
        f"{checkpoint['best_reward']:.2f}"
    )

    print(
        f"Start: "
        f"{env.start}"
    )

    print(
        f"Goal: "
        f"{env.goal}"
    )

    print(
        f"Success: "
        f"{result.success}"
    )

    print(
        f"Steps: "
        f"{result.steps}"
    )

    print(
        f"Total Reward: "
        f"{result.total_reward:.2f}"
    )

    print()
    print("Learned Path:")

    for index, state in enumerate(result.path):
        row, col = state[0], state[1]
        if index == 0:
            print(f"START → ({row}, {col})")
        else:
            print(f"       ↓ ({row}, {col})")

    print()

    if result.success:
        print("GOAL REACHED 🎯")
    else:
        print("GOAL NOT REACHED ❌")

    visualizer = GridWorldVisualizer(env)

    visualizer.plot_path(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-Learning Grid World"
    )

    parser.add_argument(
        "command",
        choices=[
            "train",
            "evaluate",
        ],
    )

    parser.add_argument(
        "--start",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Start position as row col (e.g. --start 0 0)",
    )

    parser.add_argument(
        "--goal",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Goal position as row col (e.g. --goal 4 4)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train()

    elif args.command == "evaluate":
        start = tuple(args.start) if args.start else None
        goal = tuple(args.goal) if args.goal else None
        evaluate(start=start, goal=goal)


if __name__ == "__main__":
    main()