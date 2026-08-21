from dataclasses import dataclass

from agent import QLearningAgent
from environment import GridWorld


@dataclass
class EvaluationResult:
    success: bool
    total_reward: float
    steps: int
    path: list[tuple[int, int]]


class Evaluator:
    def __init__(
        self,
        env: GridWorld,
        agent: QLearningAgent,
    ):
        self.env = env
        self.agent = agent

    def evaluate(self) -> EvaluationResult:
        state = self.env.reset()

        self.agent.epsilon = 0.0

        total_reward = 0.0
        steps = 0

        path = [state]

        done = False

        while not done:
            action = self.agent.get_greedy_action(state)

            next_state, reward, done = self.env.step(action)

            state = next_state

            path.append(state)

            total_reward += reward
            steps += 1

        row, col = state[0], state[1]

        return EvaluationResult(
            success=(row, col) == self.env.goal,
            total_reward=total_reward,
            steps=steps,
            path=path,
        )