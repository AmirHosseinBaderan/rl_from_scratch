from dataclasses import dataclass

from agent import DQNAgent, Experience
from environment import SnakeEnvironment


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    steps: int


def run_episode(
    environment: SnakeEnvironment,
    agent: DQNAgent,
    batch_size: int,
) -> EpisodeResult:

    state = environment.reset()

    total_reward = 0.0
    steps = 0

    while not environment.done:
        action = agent.select_action(state)

        result = environment.step(action)

        experience = Experience(
            state=state.values,
            action=action,
            reward=float(result.reward),
            next_state=result.state.values,
            done=result.done,
        )

        agent.replay_buffer.add(experience)

        if len(agent.replay_buffer) >= batch_size:
            agent.learn(batch_size)

        total_reward += float(result.reward)
        steps += 1

        state = result.state

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
    )