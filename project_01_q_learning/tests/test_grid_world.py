from environment import GridWorld, Action, GridWorldConfig


def create_environment():
    obstacles = {
        (0, 3),
        (1, 1),
        (1, 3),
        (2, 3),
        (3, 0),
    }

    config = GridWorldConfig(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=frozenset(obstacles),
        max_steps=100
    )

    return GridWorld(
        config=config
    )


def test_reset():
    env = create_environment()

    state = env.reset()

    assert state == (0, 0, 4, 4)
    assert env.step_count == 0


def test_valid_movement():
    env = create_environment()

    env.reset()

    state, reward, done = env.step(Action.RIGHT)

    assert state == (0, 1, 4, 4)
    assert reward == -0.1
    assert done is False


def test_obstacle_movement():
    env = create_environment()

    env.reset()

    env.step(Action.RIGHT)

    state, reward, done = env.step(Action.DOWN)

    assert state == (0, 1, 4, 4)
    assert reward == -1.0
    assert done is False


def test_outside_grid():
    env = create_environment()

    env.reset()

    state, reward, done = env.step(Action.UP)

    assert state == (0, 0, 4, 4)
    assert reward == -1.0
    assert done is False


def test_goal():
    env = create_environment()

    env.reset()

    actions = [
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
        Action.DOWN,
        Action.DOWN,
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
    ]

    for action in actions:
        state, reward, done = env.step(action)

    assert state == (4, 4, 4, 4)
    assert reward == 10.0
    assert done is True


def test_valid_actions():
    env = create_environment()

    env.reset()

    actions = env.get_valid_actions()

    assert Action.RIGHT in actions
    assert Action.DOWN in actions
    assert Action.UP not in actions
    assert Action.LEFT not in actions