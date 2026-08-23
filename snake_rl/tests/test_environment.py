import numpy as np

from environment import Direction, SnakeEnvironment


def test_environment_creation():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    assert environment.grid.width == 10
    assert environment.grid.height == 10

    assert len(environment.snake.body) == 3

    assert np.array_equal(
        environment.food.position,
        np.array([8, 5]),
    )


def test_environment_initializes_snake_in_center():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 5]),
    )

    assert environment.snake.direction == Direction.RIGHT


def test_environment_step_moves_snake():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.RIGHT)

    assert np.array_equal(
        environment.snake.head,
        np.array([6, 5]),
    )


def test_environment_step_changes_direction():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.UP)

    assert environment.snake.direction == Direction.UP

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 6]),
    )


def test_environment_reset_restores_initial_state():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.step(Direction.UP)
    environment.step(Direction.LEFT)

    environment.reset()

    assert np.array_equal(
        environment.snake.head,
        np.array([5, 5]),
    )

    assert environment.snake.direction == Direction.RIGHT

    assert np.array_equal(
        environment.food.position,
        np.array([8, 5]),
    )

def test_collision_when_snake_leaves_right_boundary():
    environment = SnakeEnvironment(
        width=5,
        height=5,
    )

    environment.step(Direction.RIGHT)
    assert not environment.is_collision()

    environment.step(Direction.RIGHT)
    assert not environment.is_collision()

    environment.step(Direction.RIGHT)
    assert environment.is_collision()

def test_collision_when_snake_leaves_left_boundary():
    environment = SnakeEnvironment(
        width=5,
        height=5,
    )

    environment.snake.body = [
        np.array([2, 2]),
        np.array([3, 2]),
        np.array([4, 2]),
    ]

    environment.snake.change_direction(Direction.LEFT)

    environment.step(Direction.LEFT)
    assert not environment.is_collision()

    environment.step(Direction.LEFT)
    assert not environment.is_collision()

    environment.step(Direction.LEFT)
    assert environment.is_collision()

def test_snake_eats_food_and_grows():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.food.position = np.array([6, 5])

    initial_length = len(environment.snake.body)

    environment.step(Direction.RIGHT)

    assert len(environment.snake.body) == initial_length + 1

    assert np.array_equal(
        environment.snake.head,
        np.array([6, 5]),
    )

def test_snake_does_not_grow_without_eating_food():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    initial_length = len(environment.snake.body)

    environment.step(Direction.RIGHT)

    assert len(environment.snake.body) == initial_length

def test_spawn_food_does_not_place_food_on_snake():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.spawn_food()

    assert not any(
        np.array_equal(
            environment.food.position,
            segment,
        )
        for segment in environment.snake.body
    )

def test_food_moves_after_being_eaten():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.food.position = np.array([6, 5])

    environment.step(Direction.RIGHT)

    assert not np.array_equal(
        environment.food.position,
        np.array([6, 5]),
    )

    assert not any(
        np.array_equal(
            environment.food.position,
            segment,
        )
        for segment in environment.snake.body
    )

def test_environment_starts_with_done_false():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    assert environment.done is False

def test_environment_becomes_done_after_wall_collision():
    environment = SnakeEnvironment(
        width=5,
        height=5,
    )

    environment.snake.body = [
        np.array([4, 2]),
        np.array([3, 2]),
        np.array([2, 2]),
    ]

    environment.snake.change_direction(Direction.RIGHT)

    environment.step(Direction.RIGHT)

    assert environment.done is True

def test_environment_becomes_done_after_self_collision():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.snake.body = [
        np.array([5, 5]),
        np.array([5, 4]),
        np.array([4, 4]),
        np.array([4, 5]),
        np.array([5, 6]),
    ]

    environment.snake.direction = Direction.LEFT

    environment.step(Direction.LEFT)

    assert environment.done is True

def test_environment_does_not_move_after_done():
    environment = SnakeEnvironment(
        width=5,
        height=5,
    )

    environment.snake.body = [
        np.array([4, 2]),
        np.array([3, 2]),
        np.array([2, 2]),
    ]

    environment.step(Direction.RIGHT)

    assert environment.done is True

    head_after_collision = environment.snake.head.copy()

    environment.step(Direction.RIGHT)

    assert np.array_equal(
        environment.snake.head,
        head_after_collision,
    )