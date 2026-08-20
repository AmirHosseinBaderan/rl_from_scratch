import matplotlib

matplotlib.use("Agg")

from environment import GridWorld
from evaluation.evaluate import EvaluationResult
from visualization.visualize import GridWorldVisualizer


def test_plot_path():
    obstacles = {
        (1, 1),
    }

    env = GridWorld(
        rows=3,
        cols=3,
        start=(0, 0),
        goal=(2, 2),
        obstacles=obstacles,
    )

    result = EvaluationResult(
        success=True,
        total_reward=9.0,
        steps=4,
        path=[
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ],
    )

    visualizer = GridWorldVisualizer(env)

    visualizer.plot_path(result)
