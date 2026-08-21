import matplotlib.pyplot as plt
import numpy as np

from environment import Food, Grid, Snake


class Renderer:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid

        self.figure, self.axes = plt.subplots()

        self.axes.set_xlim(0, grid.width)
        self.axes.set_ylim(0, grid.height)

        self.axes.set_xticks(np.arange(0, grid.width + 1))
        self.axes.set_yticks(np.arange(0, grid.height + 1))

        self.axes.grid(True)

        self.axes.set_aspect("equal")

    def render(self, snake: Snake, food: Food) -> None:
        self.axes.clear()

        self.axes.set_xlim(0, self.grid.width)
        self.axes.set_ylim(0, self.grid.height)

        self.axes.set_xticks(np.arange(0, self.grid.width + 1))
        self.axes.set_yticks(np.arange(0, self.grid.height + 1))

        self.axes.grid(True)
        self.axes.set_aspect("equal")

        snake_positions = np.array(snake.body)

        self.axes.scatter(
            snake_positions[:, 0] + 0.5,
            snake_positions[:, 1] + 0.5,
            s=500,
        )

        self.axes.scatter(
            food.position[0] + 0.5,
            food.position[1] + 0.5,
            s=500,
        )

        plt.pause(0.01)

    def close(self) -> None:
        plt.close(self.figure)
