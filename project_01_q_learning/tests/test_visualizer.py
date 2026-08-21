import matplotlib

matplotlib.use("Agg")

from environment import GridWorld, GridWorldConfig
from evaluation import EvaluationResult
from visualization import GridWorldVisualizer


def test_plot_path():
    obstacles = {
        (1, 1),
    }

    config = GridWorldConfig(
        rows=3,
        cols=3,
        start=(0, 0),
        goal=(2, 2),
        obstacles=frozenset(obstacles),
    )

    env = GridWorld(
        config=config,
    )

    result = EvaluationResult(
        success=True,
        total_reward=9.0,
        steps=4,
        path=[
            (0, 0, 2, 2),
            (0, 1, 2, 2),
            (0, 2, 2, 2),
            (1, 2, 2, 2),
            (2, 2, 2, 2),
        ],
    )

    visualizer = GridWorldVisualizer(env)

    visualizer.plot_path(result)
