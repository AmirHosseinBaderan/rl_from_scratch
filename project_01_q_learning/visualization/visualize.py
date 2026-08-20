import matplotlib.pyplot as plt

from environment import GridWorld
from evaluation.evaluate import EvaluationResult


class GridWorldVisualizer:
    def __init__(self, env: GridWorld):
        self.env = env

    def plot_path(
        self,
        result: EvaluationResult,
    ):
        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        self._draw_grid(ax)
        self._draw_obstacles(ax)
        self._draw_start(ax)
        self._draw_goal(ax)
        self._draw_path(ax, result.path)

        ax.set_title(
            f"Learned Path | "
            f"Steps: {result.steps} | "
            f"Reward: {result.total_reward:.2f}"
        )

        plt.tight_layout()
        plt.show()

    def _draw_grid(self, ax):
        ax.set_xlim(0, self.env.cols)
        ax.set_ylim(0, self.env.rows)

        ax.set_xticks(range(self.env.cols + 1))
        ax.set_yticks(range(self.env.rows + 1))

        ax.grid(True)

        ax.set_aspect("equal")

        ax.invert_yaxis()

    def _draw_obstacles(self, ax):
        for row, col in self.env.obstacles:
            ax.add_patch(
                plt.Rectangle(
                    (col, row),
                    1,
                    1,
                )
            )

    def _draw_start(self, ax):
        row, col = self.env.start

        ax.text(
            col + 0.5,
            row + 0.5,
            "S",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

    def _draw_goal(self, ax):
        row, col = self.env.goal

        ax.text(
            col + 0.5,
            row + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

    def _draw_path(self, ax, path):
        if len(path) < 2:
            return

        x = [
            col + 0.5
            for row, col in path
        ]

        y = [
            row + 0.5
            for row, col in path
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
        )