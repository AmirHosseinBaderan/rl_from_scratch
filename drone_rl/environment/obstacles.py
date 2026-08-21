"""Obstacle management for the drone environment."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import EnvConfig, Obstacle


class ObstacleManager:
    """Manages obstacle generation and collision detection."""

    def __init__(self, config: EnvConfig) -> None:
        """Initialize the obstacle manager.

        Args:
            config: Environment configuration.
        """
        self.config = config
        self.obstacles: List[Obstacle] = []

    def generate(
        self,
        rng: np.random.Generator,
        drone_start: np.ndarray,
        target_pos: np.ndarray,
    ) -> List[Obstacle]:
        """Generate obstacles for the episode.

        Obstacles are placed randomly but avoid the drone start position
        and the target position.

        Args:
            rng: NumPy random generator for reproducibility.
            drone_start: Starting position of the drone.
            target_pos: Target position.

        Returns:
            List of generated obstacles.
        """
        if not self.config.obstacle_randomize:
            # Use fixed obstacles (deterministic based on seed)
            return self._generate_fixed(rng, drone_start, target_pos)

        obstacles = []
        x_min, x_max, y_min, y_max, z_min, z_max = self.config.world_bounds

        for _ in range(self.config.num_obstacles):
            for attempt in range(100):  # Try to place obstacle without overlap
                # Random position
                pos = rng.uniform(
                    low=[x_min, y_min, z_min],
                    high=[x_max, y_max, z_max],
                )

                # Random size
                size = rng.uniform(
                    low=self.config.obstacle_min_size,
                    high=self.config.obstacle_max_size,
                    size=3,
                )

                obstacle = Obstacle(position=pos, size=size)

                # Check distance from drone start and target
                dist_to_start = np.linalg.norm(pos - drone_start)
                dist_to_target = np.linalg.norm(pos - target_pos)

                min_dist = self.config.target_reached_threshold * 3
                if dist_to_start < min_dist or dist_to_target < min_dist:
                    continue

                # Check overlap with existing obstacles
                overlap = False
                for existing in obstacles:
                    if self._obstacles_overlap(obstacle, existing):
                        overlap = True
                        break

                if not overlap:
                    obstacles.append(obstacle)
                    break

        self.obstacles = obstacles
        return obstacles

    def _generate_fixed(
        self,
        rng: np.random.Generator,
        drone_start: np.ndarray,
        target_pos: np.ndarray,
    ) -> List[Obstacle]:
        """Generate fixed obstacles using the random generator state.

        Args:
            rng: NumPy random generator.
            drone_start: Starting position of the drone.
            target_pos: Target position.

        Returns:
            List of generated obstacles.
        """
        # Use the same generation logic but with a fixed seed
        return self.generate(rng, drone_start, target_pos)

    def _obstacles_overlap(self, a: Obstacle, b: Obstacle) -> bool:
        """Check if two obstacles overlap."""
        min_a, max_a = a.get_bounds()
        min_b, max_b = b.get_bounds()

        # Check if boxes overlap on all axes
        overlap_x = min_a[0] <= max_b[0] and max_a[0] >= min_b[0]
        overlap_y = min_a[1] <= max_b[1] and max_a[1] >= min_b[1]
        overlap_z = min_a[2] <= max_b[2] and max_a[2] >= min_b[2]

        return overlap_x and overlap_y and overlap_z

    def check_collision(
        self, position: np.ndarray, drone_radius: float = 0.3
    ) -> Tuple[bool, Optional[Obstacle]]:
        """Check if the drone collides with any obstacle.

        Args:
            position: Drone position.
            drone_radius: Radius of the drone for collision detection.

        Returns:
            Tuple of (collision_detected, colliding_obstacle).
        """
        for obstacle in self.obstacles:
            if obstacle.intersects_sphere(position, drone_radius):
                return True, obstacle
        return False, None

    def get_obstacles(self) -> List[Obstacle]:
        """Return the current list of obstacles."""
        return self.obstacles
