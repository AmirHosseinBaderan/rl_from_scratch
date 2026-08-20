class GridInput:
    def __init__(
        self,
        rows: int,
        cols: int,
    ):
        self.rows = rows
        self.cols = cols

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
                        f"Row: 0-{self.rows - 1}, "
                        f"Col: 0-{self.cols - 1}"
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
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
    ]:

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

            return start, goal

    def _is_valid_position(
        self,
        row: int,
        col: int,
    ) -> bool:
        return (
            0 <= row < self.rows
            and
            0 <= col < self.cols
        )