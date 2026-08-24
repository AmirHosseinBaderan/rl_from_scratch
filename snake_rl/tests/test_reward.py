from environment import Reward, SnakeEnvironment, Direction,Action
import numpy as np

def test_food_reward():
    assert Reward.FOOD == 10


def test_collision_reward():
    assert Reward.COLLISION == -10


def test_step_reward():
    assert Reward.STEP == -1

def test_step_returns_step_reward():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    step_result = environment.step(Action.RIGHT)

    assert step_result.reward == Reward.STEP

def test_step_returns_food_reward_when_food_is_eaten():
    environment = SnakeEnvironment(
        width=10,
        height=10,
    )

    environment.food.position = np.array([6, 5])

    step_result = environment.step(Action.RIGHT)

    assert step_result.reward == Reward.FOOD

def test_step_returns_collision_reward_when_snake_collides():
    environment = SnakeEnvironment(
        width=5,
        height=5,
    )

    environment.snake.body = [
        np.array([4, 2]),
        np.array([3, 2]),
        np.array([2, 2]),
    ]

    step_result = environment.step(Action.RIGHT)

    assert step_result.reward == Reward.COLLISION
    assert environment.done is True