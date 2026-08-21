import numpy as np

from environment import Grid, Food, Snake
from visualization import Renderer


def main() -> None:
    grid = Grid(
        width=10,
        height=10,
    )

    snake = Snake(
        body=[
            np.array([4, 5]),
            np.array([3, 5]),
            np.array([2, 5]),
        ],
        direction=np.array([1, 0]),
    )

    food = Food(
        position=np.array([7, 5]),
    )

    renderer = Renderer(grid)

    renderer.render(
        snake=snake,
        food=food,
    )

    input("Press Enter to close...")

    renderer.close()


if __name__ == "__main__":
    main()
