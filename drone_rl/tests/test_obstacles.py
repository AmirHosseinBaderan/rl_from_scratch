"""Tests for obstacle management."""

from __future__ import annotations

import numpy as np
import pytest

from environment import EnvConfig, Obstacle, ObstacleManager


class TestObstacleManager:
    """Test suite for ObstacleManager."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = EnvConfig(
            world_bounds=(-10, 10, -10, 10, 0, 10),
            num_obstacles=3,
            obstacle_min_size=0.5,
            obstacle_max_size=1.0,
        )
        self.manager = ObstacleManager(self.config)

    def test_obstacle_generation_works(self) -> None:
        """Test that obstacle generation works."""
        rng = np.random.default_rng(42)
        drone_start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        target_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        obstacles = self.manager.generate(rng, drone_start, target_pos)

        assert isinstance(obstacles, list)
        assert len(obstacles) <= self.config.num_obstacles

    def test_obstacle_generation_deterministic(self) -> None:
        """Test that obstacle generation is deterministic with seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        drone_start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        target_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        obstacles1 = self.manager.generate(rng1, drone_start, target_pos)
        obstacles2 = self.manager.generate(rng2, drone_start, target_pos)

        assert len(obstacles1) == len(obstacles2)
        for o1, o2 in zip(obstacles1, obstacles2):
            assert np.allclose(o1.position, o2.position)
            assert np.allclose(o1.size, o2.size)

    def test_collision_detection_works(self) -> None:
        """Test that collision detection works."""
        # Add obstacle to manager
        obstacle = Obstacle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.manager.obstacles = [obstacle]

        # Point inside obstacle
        collision, _ = self.manager.check_collision(
            np.array([5.0, 5.0, 5.0], dtype=np.float32)
        )
        assert collision is True

        # Point outside obstacle
        collision, _ = self.manager.check_collision(
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        assert collision is False

    def test_collision_with_sphere(self) -> None:
        """Test collision detection with sphere (drone radius)."""
        # Add obstacle to manager
        obstacle = Obstacle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.manager.obstacles = [obstacle]

        # Point just outside obstacle but within drone radius
        collision, _ = self.manager.check_collision(
            np.array([5.5, 5.0, 5.0], dtype=np.float32), drone_radius=0.6
        )
        assert collision is True

        # Point far from obstacle
        collision, _ = self.manager.check_collision(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), drone_radius=0.3
        )
        assert collision is False

    def test_obstacle_bounds(self) -> None:
        """Test obstacle bounds calculation."""
        obstacle = Obstacle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            size=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        )

        min_bounds, max_bounds = obstacle.get_bounds()

        assert np.allclose(min_bounds, [4.0, 3.0, 4.5])
        assert np.allclose(max_bounds, [6.0, 7.0, 5.5])

    def test_obstacle_contains_point(self) -> None:
        """Test point containment in obstacle."""
        obstacle = Obstacle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        assert obstacle.contains_point(np.array([5.0, 5.0, 5.0])) is True
        assert obstacle.contains_point(np.array([4.5, 5.0, 5.0])) is True
        assert obstacle.contains_point(np.array([0.0, 0.0, 0.0])) is False

    def test_obstacle_intersects_sphere(self) -> None:
        """Test sphere-box intersection."""
        obstacle = Obstacle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Sphere center inside box
        assert obstacle.intersects_sphere(np.array([5.0, 5.0, 5.0]), 0.1)

        # Sphere center outside but overlapping
        assert obstacle.intersects_sphere(np.array([5.5, 5.0, 5.0]), 0.6)

        # Sphere far away
        assert not obstacle.intersects_sphere(np.array([0.0, 0.0, 0.0]), 0.3)

    def test_no_obstacles_overlap(self) -> None:
        """Test that generated obstacles don't overlap."""
        rng = np.random.default_rng(42)
        drone_start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        target_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        obstacles = self.manager.generate(rng, drone_start, target_pos)

        # Check no overlaps
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                assert not self.manager._obstacles_overlap(obstacles[i], obstacles[j])
