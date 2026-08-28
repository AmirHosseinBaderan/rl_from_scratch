import torch

from environment import Action, SnakeState

from .replay_buffer import ReplayBuffer
from .action_selector import EpsilonGreedySelector
from .batch import create_batch
from .q_network import QNetwork
from .q_values import select_action_q_values
from .targets import compute_bellman_target
from .loss import compute_loss
from .network_optimizer import NetworkOptimizer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        replay_capacity: int,
    ) -> None:
        self.gamma = gamma

        self.network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        )

        self.selector = EpsilonGreedySelector(
            epsilon=epsilon,
            action_size=action_size,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
        )

        self.optimizer = NetworkOptimizer(
            network=self.network,
            learning_rate=learning_rate,
        )

    def select_action(
            self,
            state: SnakeState,
    ) -> Action:
        state_tensor = torch.as_tensor(
            state.values,
            dtype=torch.float32,
        )

        with torch.no_grad():
            q_values = self.network(
                state_tensor
            )

        return self.selector.select(q_values)

    def learn(
        self,
        batch_size: int,
    ) -> float:
        experiences = self.replay_buffer.sample(
            batch_size
        )

        batch = create_batch(experiences)

        current_q_values = self.network(
            batch.states
        )

        predicted_q_values = select_action_q_values(
            q_values=current_q_values,
            actions=batch.actions,
        )

        with torch.no_grad():
            next_q_values = self.network(
                batch.next_states
            )

            target_q_values = compute_bellman_target(
                rewards=batch.rewards,
                next_q_values=next_q_values,
                dones=batch.dones,
                gamma=self.gamma,
            )

        loss = compute_loss(
            predicted_q_values=predicted_q_values,
            target_q_values=target_q_values,
        )

        self.optimizer.step(loss)

        return loss.item()