"""3D visualization renderer for the drone environment."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

from environment import EnvConfig, Obstacle, Target


class DroneRenderer:
    """3D visualization renderer for the drone environment.

    This renderer is completely decoupled from the environment logic.
    It takes the necessary data as parameters and renders it.
    """

    def __init__(self, config: EnvConfig) -> None:
        """Initialize the renderer.

        Args:
            config: Environment configuration for world bounds.
        """
        self.config = config
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[Axes3D] = None

    def setup_figure(self) -> Tuple[plt.Figure, Axes3D]:
        """Set up the 3D figure and axes.

        Returns:
            Tuple of (figure, axes).
        """
        self._fig = plt.figure(figsize=(10, 8))
        self._ax = self._fig.add_subplot(111, projection="3d")
        return self._fig, self._ax

    def render(
        self,
        drone_position: np.ndarray,
        target: Target,
        obstacles: List[Obstacle],
        trajectory: np.ndarray,
        title: str = "Drone Navigation",
    ) -> plt.Figure:
        """Render the current state of the environment.

        Args:
            drone_position: Current drone position.
            target: Target object.
            obstacles: List of obstacles.
            trajectory: Array of shape (num_steps, 3) with trajectory.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        if self._fig is None or self._ax is None:
            self.setup_figure()

        ax = self._ax
        ax.clear()

        # Set world bounds
        x_min, x_max, y_min, y_max, z_min, z_max = self.config.world_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        # Draw world boundaries as a wireframe box
        self._draw_world_bounds(ax, x_min, x_max, y_min, y_max, z_min, z_max)

        # Draw obstacles
        for obstacle in obstacles:
            self._draw_obstacle(ax, obstacle)

        # Draw target
        self._draw_target(ax, target)

        # Draw trajectory
        if len(trajectory) > 0:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "b-",
                linewidth=1.5,
                alpha=0.7,
                label="Trajectory",
            )

        # Draw drone
        ax.scatter(
            drone_position[0],
            drone_position[1],
            drone_position[2],
            c="red",
            s=100,
            marker="^",
            label="Drone",
        )

        ax.legend()
        plt.tight_layout()

        return self._fig

    def _draw_world_bounds(
        self,
        ax: Axes3D,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ) -> None:
        """Draw the world boundaries as a wireframe box."""
        # Define the 8 corners of the box
        corners = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )

        # Define edges
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical
        ]

        for i, j in edges:
            ax.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                "k-",
                linewidth=0.5,
                alpha=0.3,
            )

    def _draw_obstacle(self, ax: Axes3D, obstacle: Obstacle) -> None:
        """Draw an obstacle as a wireframe box."""
        min_bounds, max_bounds = obstacle.get_bounds()

        # Define corners
        corners = np.array(
            [
                [min_bounds[0], min_bounds[1], min_bounds[2]],
                [max_bounds[0], min_bounds[1], min_bounds[2]],
                [max_bounds[0], max_bounds[1], min_bounds[2]],
                [min_bounds[0], max_bounds[1], min_bounds[2]],
                [min_bounds[0], min_bounds[1], max_bounds[2]],
                [max_bounds[0], min_bounds[1], max_bounds[2]],
                [max_bounds[0], max_bounds[1], max_bounds[2]],
                [min_bounds[0], max_bounds[1], max_bounds[2]],
            ]
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for i, j in edges:
            ax.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                "r-",
                linewidth=1.0,
                alpha=0.6,
            )

    def _draw_target(self, ax: Axes3D, target: Target) -> None:
        """Draw the target as a green sphere."""
        ax.scatter(
            target.position[0],
            target.position[1],
            target.position[2],
            c="green",
            s=200,
            marker="o",
            label="Target",
        )

    def show(self) -> None:
        """Display the figure."""
        if self._fig is not None:
            plt.show()

    def save(self, filepath: str) -> None:
        """Save the figure to a file.

        Args:
            filepath: Path to save the figure.
        """
        if self._fig is not None:
            self._fig.savefig(filepath, dpi=150, bbox_inches="tight")

    def close(self) -> None:
        """Close the figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
