import numpy as np

from environment import Action


class QLearningAgent:
    def __init__(
        self,
        rows: int,
        cols: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.rows = rows
        self.cols = cols

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.num_actions = len(Action)

        self.q_table = np.zeros(
            (
                rows,
                cols,
                rows,
                cols,
                self.num_actions,
            ),
            dtype=np.float32,
        )

    def choose_action(
        self,
        state: tuple[int, int, int, int],
        training: bool = True,
    ) -> Action:
        if training and np.random.random() < self.epsilon:
            return Action(
                np.random.randint(
                    self.num_actions
                )
            )

        row, col, goal_row, goal_col = state

        q_values = self.q_table[
            row,
            col,
            goal_row,
            goal_col,
        ]

        return Action(
            int(np.argmax(q_values))
        )

    def update(
        self,
        state: tuple[int, int, int, int],
        action: Action,
        reward: float,
        next_state: tuple[int, int, int, int],
        done: bool,
    ) -> None:
        row, col, goal_row, goal_col = state

        next_row, next_col, next_goal_row, next_goal_col = (
            next_state
        )

        current_q = self.q_table[
            row,
            col,
            goal_row,
            goal_col,
            action.value,
        ]

        if done:
            target = reward
        else:
            next_q = self.q_table[
                next_row,
                next_col,
                next_goal_row,
                next_goal_col,
            ]

            target = (
                reward
                + self.discount_factor
                * np.max(next_q)
            )

        self.q_table[
            row,
            col,
            goal_row,
            goal_col,
            action.value,
        ] += (
            self.learning_rate
            * (target - current_q)
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min,
        )

    def get_greedy_action(self, state):
        row, col, goal_row, goal_col = state

        return Action(
            int(np.argmax(
                self.q_table[row, col, goal_row, goal_col]
            ))
        )
