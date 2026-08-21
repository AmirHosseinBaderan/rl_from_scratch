from agent import QLearningAgent
from environment import GridWorld, GridWorldConfig, Action
from evaluation.evaluate import Evaluator


def test_evaluation_reaches_goal():
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
    )

    env = GridWorld(
        config=config,
    )

    agent = QLearningAgent(
        rows=5,
        cols=5,
    )

    # Artificially define a known policy
    agent.q_table[0, 0, 4, 4, Action.RIGHT] = 1
    agent.q_table[0, 1, 4, 4, Action.RIGHT] = 1
    agent.q_table[0, 2, 4, 4, Action.DOWN] = 1
    agent.q_table[1, 2, 4, 4, Action.DOWN] = 1
    agent.q_table[2, 2, 4, 4, Action.DOWN] = 1
    agent.q_table[3, 2, 4, 4, Action.RIGHT] = 1
    agent.q_table[3, 3, 4, 4, Action.DOWN] = 1
    agent.q_table[4, 3, 4, 4, Action.RIGHT] = 1

    evaluator = Evaluator(
        env=env,
        agent=agent,
    )

    result = evaluator.evaluate()

    assert result.success is True
    assert result.path[-1] == (4, 4, 4, 4)