import random
import numpy as np

from project_01_q_learning.environment import Action


class QLearningAgent:
    def __init__(self,
                 rows,
                 cols,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01
                 ):
        self.rows = rows
        self.cols = cols

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros(
            (rows, cols, len(Action)),
            dtype=np.float32
        )

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(Action))

        row, col = state
        return Action(
            np.argmax(
                self.q_table[row, col]
            )
        )

    def update(
            self,
            state,
            action,
            reward,
            next_state,
            done
    ):
        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[
            row,
            col,
            action
        ]

        if done:
            target = reward

        else:
            next_max_q = np.max(
                self.q_table[next_row, next_col]
            )

            target = (
                    reward
                    + self.discount_factor * next_max_q
            )

        new_q = (
                current_q
                + self.learning_rate * (
                        target - current_q
                )
        )

        self.q_table[
            row,
            col,
            action
        ] = new_q

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
