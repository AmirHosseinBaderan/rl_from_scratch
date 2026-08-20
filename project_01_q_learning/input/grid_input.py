from environment import GridWorldConfig


class GridInput:
    def __init__(
        self,
        config: GridWorldConfig,
    ):
        self.config = config

    def get_position(
        self,
        name: str,
    ) -> tuple[int, int]:

        while True:
            try:
                value = input(
                    f"Enter {name} position (row col): "
                ).strip()

                row, col = map(
                    int,
                    value.split(),
                )

                if not self._is_valid_position(
                    row,
                    col,
                ):
                    print(
                        f"Invalid position. "
                        f"Row: 0-{self.config.rows - 1}, "
                        f"Col: 0-{self.config.cols - 1}"
                    )
                    continue

                return row, col

            except ValueError:
                print(
                    "Invalid input. "
                    "Use: row col"
                )

    def get_start_and_goal(
        self,
        obstacles: set[tuple[int, int]],
    ) -> GridWorldConfig:

        while True:
            start = self.get_position("start")
            goal = self.get_position("goal")

            if start == goal:
                print(
                    "Start and goal cannot be the same."
                )
                continue

            if start in obstacles:
                print(
                    "Start cannot be placed "
                    "on an obstacle."
                )
                continue

            if goal in obstacles:
                print(
                    "Goal cannot be placed "
                    "on an obstacle."
                )
                continue

            return GridWorldConfig(
                rows=self.config.rows,
                cols=self.config.cols,
                start=start,
                goal=goal,
                obstacles=frozenset(obstacles),
                max_steps=self.config.max_steps,
            )

    def _is_valid_position(
        self,
        row: int,
        col: int,
    ) -> bool:
        return (
            0 <= row < self.config.rows
            and
            0 <= col < self.config.cols
        )
