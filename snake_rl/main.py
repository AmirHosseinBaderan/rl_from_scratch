import time

from environment import Direction, SnakeEnvironment
from visualization import Renderer


def main() -> None:
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    renderer = Renderer(environment.grid)

    directions = [
        Direction.RIGHT,
        Direction.RIGHT,
        Direction.RIGHT,
        Direction.UP,
        Direction.UP,
        Direction.LEFT,
        Direction.LEFT,
        Direction.DOWN,
        Direction.DOWN,
        Direction.RIGHT,
    ]

    for direction in directions:
        environment.step(direction)

        renderer.render(
            snake=environment.snake,
            food=environment.food,
        )

        time.sleep(0.2)

    input("Press Enter to close...")

    renderer.close()


if __name__ == "__main__":
    main()