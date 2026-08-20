from pathlib import Path

from project_01_q_learning.agent import QLearningAgent
from project_01_q_learning.environment import GridWorld
from project_01_q_learning.training import (
    CheckpointManager,
    TrainingConfig,
    TensorBoardLogger
)


class Trainer:
    def __init__(
            self,
            env: GridWorld,
            agent: QLearningAgent,
            config: TrainingConfig,
            checkpoint_manager: CheckpointManager,
            logger: TensorBoardLogger,
    ):
        self.env = env
        self.agent = agent
        self.config = config

        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

        self.start_episode = 1
        self.best_reward = float("-inf")

        self.successful_episodes = 0

    def train(self):
        for episode in range(
                self.start_episode,
                self.config.episodes + 1,
        ):
            state = self.env.reset()

            total_reward = 0.0
            episode_length = 0

            done = False

            while not done:
                action = self.agent.choose_action(state)

                next_state, reward, done = self.env.step(
                    action
                )

                self.agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                state = next_state

                total_reward += reward
                episode_length += 1

            success = state == self.env.goal

            if success:
                self.successful_episodes += 1

            success_rate = (
                    self.successful_episodes / episode
            )

            self.agent.decay_epsilon()

            self.logger.log_episode(
                episode=episode,
                reward=total_reward,
                length=episode_length,
                epsilon=self.agent.epsilon,
                success_rate=success_rate,
            )

            if total_reward > self.best_reward:
                self.best_reward = total_reward

                self.checkpoint_manager.save_best(
                    agent=self.agent,
                    episode=episode,
                    best_reward=self.best_reward,
                )

            if (
                    episode
                    % self.config.checkpoint_interval
                    == 0
            ):
                self.checkpoint_manager.save_latest(
                    agent=self.agent,
                    episode=episode,
                    best_reward=self.best_reward,
                )

            if (
                    episode
                    % self.config.log_interval
                    == 0
            ):
                print(
                    f"Episode {episode:5d} | "
                    f"Reward: {total_reward:7.2f} | "
                    f"Length: {episode_length:3d} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Success Rate: {success_rate:.2%}"
                )

        self.logger.close()
