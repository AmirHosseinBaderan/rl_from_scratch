from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    def __init__(
        self,
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=None,
        max_steps=100,
    ):
        self.rows = rows
        self.cols = cols

        self.start = start
        self.goal = goal

        self.obstacles = set(obstacles or [])

        self.max_steps = max_steps

        self.agent_position = self.start
        self.step_count = 0

    def reset(self):
        self.agent_position = self.start
        self.step_count = 0

        return self.agent_position

    def step(self, action):
        self.step_count += 1

        current_row, current_col = self.agent_position

        next_row = current_row
        next_col = current_col

        if action == Action.UP:
            next_row -= 1

        elif action == Action.DOWN:
            next_row += 1

        elif action == Action.LEFT:
            next_col -= 1

        elif action == Action.RIGHT:
            next_col += 1

        else:
            raise ValueError(f"Invalid action: {action}")

        next_position = (next_row, next_col)

        # Outside the grid
        if not self._is_inside(next_position):
            reward = -1.0
            next_position = self.agent_position

        # Obstacle
        elif next_position in self.obstacles:
            reward = -1.0
            next_position = self.agent_position

        # Goal
        elif next_position == self.goal:
            reward = 10.0

        # Normal movement
        else:
            reward = -0.1

        self.agent_position = next_position

        done = (
            self.agent_position == self.goal
            or self.step_count >= self.max_steps
        )

        return self.agent_position, reward, done

    def get_valid_actions(self):
        valid_actions = []

        for action in Action:
            next_position = self._get_next_position(
                self.agent_position,
                action,
            )

            if self._is_inside(next_position) and \
                    next_position not in self.obstacles:
                valid_actions.append(action)

        return valid_actions

    def _get_next_position(self, position, action):
        row, col = position

        if action == Action.UP:
            row -= 1

        elif action == Action.DOWN:
            row += 1

        elif action == Action.LEFT:
            col -= 1

        elif action == Action.RIGHT:
            col += 1

        return row, col

    def _is_inside(self, position):
        row, col = position

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )