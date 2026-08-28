from dataclasses import dataclass

from agent import DQNAgent
from agent.experience import Experience
from environment import Action, SnakeEnvironment


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    steps: int
    losses: list[float]


def run_episode(
    environment: SnakeEnvironment,
    agent: DQNAgent,
    batch_size: int,
) -> EpisodeResult:
    state = environment.reset()

    total_reward = 0.0
    steps = 0
    losses: list[float] = []

    while not environment.done:
        action: Action = agent.select_action(state)

        result = environment.step(action)

        total_reward += result.reward
        steps += 1

        experience = Experience(
            state=state.values.copy(),
            action=action,
            reward=result.reward,
            next_state=result.state.values.copy(),
            done=result.done,
        )

        agent.replay_buffer.add(experience)

        if len(agent.replay_buffer) >= batch_size:
            loss = agent.learn(batch_size)
            losses.append(loss)

        state = result.state

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        losses=losses,
    )
