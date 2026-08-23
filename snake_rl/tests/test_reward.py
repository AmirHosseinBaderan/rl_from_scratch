from environment import Reward


def test_food_reward():
    assert Reward.FOOD == 10


def test_collision_reward():
    assert Reward.COLLISION == -10


def test_step_reward():
    assert Reward.STEP == -1