from environment import Action
from agent import QLearningAgent

import numpy as np



def test_q_table_initialization():
    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    assert agent.q_table.shape == (5, 5, 5, 5, 4)

    assert np.all(agent.q_table == 0)

def test_terminal_q_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
    )

    state = (0, 0, 4, 4)
    action = Action.RIGHT

    agent.update(
        state=state,
        action=action,
        reward=10.0,
        next_state=(0, 1, 4, 4),
        done=True,
    )

    assert agent.q_table[0, 0, 4, 4, Action.RIGHT] == 1.0

def test_non_terminal_q_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.99,
    )

    agent.q_table[0, 1, 4, 4, Action.RIGHT] = 5.0

    agent.update(
        state=(0, 0, 4, 4),
        action=Action.RIGHT,
        reward=-0.1,
        next_state=(0, 1, 4, 4),
        done=False,
    )

    assert np.isclose(
        agent.q_table[0, 0, 4, 4, Action.RIGHT],
        0.485,
    )

def test_choose_action_exploitation():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        epsilon=0.0,
    )

    state = (
        2,
        2,
        4,
        4,
    )

    agent.q_table[
        2,
        2,
        4,
        4,
        Action.RIGHT.value,
    ] = 10.0

    action = agent.choose_action(
        state,
        training=True,
    )

    assert action == Action.RIGHT

def test_different_goals_have_different_q_values():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        epsilon=0.0,
    )

    state_a = (
        2,
        2,
        4,
        4,
    )

    state_b = (
        2,
        2,
        0,
        0,
    )

    agent.q_table[
        2,
        2,
        4,
        4,
        Action.RIGHT.value,
    ] = 10.0

    agent.q_table[
        2,
        2,
        0,
        0,
        Action.LEFT.value,
    ] = 10.0

    assert (
        agent.choose_action(
            state_a,
            training=False,
        )
        == Action.RIGHT
    )

    assert (
        agent.choose_action(
            state_b,
            training=False,
        )
        == Action.LEFT
    )

def test_q_value_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    state = (
        2,
        2,
        4,
        4,
    )

    next_state = (
        2,
        3,
        4,
        4,
    )

    agent.q_table[
        2,
        3,
        4,
        4,
        Action.RIGHT.value,
    ] = 10.0

    agent.update(
        state=state,
        action=Action.RIGHT,
        reward=1.0,
        next_state=next_state,
        done=False,
    )

    expected = (
        0.1
        * (1.0 + 0.9 * 10.0)
    )

    assert np.isclose(
        agent.q_table[
            2,
            2,
            4,
            4,
            Action.RIGHT.value,
        ],
        expected,
    )

def test_terminal_state_update():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    state = (
        3,
        4,
        4,
        4,
    )

    next_state = (
        4,
        4,
        4,
        4,
    )

    agent.update(
        state=state,
        action=Action.DOWN,
        reward=10.0,
        next_state=next_state,
        done=True,
    )

    expected = 1.0

    assert np.isclose(
        agent.q_table[
            3,
            4,
            4,
            4,
            Action.DOWN.value,
        ],
        expected,
    )

def test_epsilon_zero_means_exploitation():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        epsilon=0.0,
    )

    state = (
        0,
        0,
        4,
        4,
    )

    agent.q_table[
        0,
        0,
        4,
        4,
        Action.DOWN.value,
    ] = 5.0

    for _ in range(20):
        action = agent.choose_action(
            state,
            training=True,
        )

        assert action == Action.DOWN

def test_epsilon_decay():
    agent = QLearningAgent(
        rows=5,
        cols=5,
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.1,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.5

    agent.decay_epsilon()

    assert agent.epsilon == 0.25

    agent.decay_epsilon()
    agent.decay_epsilon()
    agent.decay_epsilon()

    assert agent.epsilon >= 0.1