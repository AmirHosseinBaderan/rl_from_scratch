"""Configuration and data structures for the drone environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    """Configuration for the drone environment.

    All parameters are configurable to allow easy experimentation.
    """

    # World boundaries
    world_bounds: Tuple[float, float, float, float, float, float] = (
        -10.0, 10.0,  # x_min, x_max
        -10.0, 10.0,  # y_min, y_max
        0.0, 10.0,    # z_min, z_max
    )

    # Physics parameters
    dt: float = 0.1
    max_velocity: float = 5.0
    max_acceleration: float = 2.0
    drag_coefficient: float = 0.1
    throttle_force: float = 3.0

    # Orientation parameters
    max_roll_rate: float = 1.0
    max_pitch_rate: float = 1.0
    max_yaw_rate: float = 0.5

    # Episode parameters
    max_steps: int = 500
    target_reached_threshold: float = 0.5

    # Reward parameters
    reward_distance_scale: float = 1.0
    reward_target_reached: float = 100.0
    reward_collision: float = -100.0
    reward_out_of_bounds: float = -50.0
    reward_step_penalty: float = -0.1

    # Obstacle parameters
    num_obstacles: int = 5
    obstacle_min_size: float = 0.5
    obstacle_max_size: float = 2.0
    obstacle_randomize: bool = True

    # Target parameters
    target_randomize: bool = True
    target_min_distance: float = 3.0
    target_max_distance: float = 8.0

    # Random seed
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.max_velocity <= 0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.target_reached_threshold <= 0:
            raise ValueError(
                f"target_reached_threshold must be positive, "
                f"got {self.target_reached_threshold}"
            )


@dataclass
class DroneState:
    """State of the drone at a given timestep."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # orientation: [roll, pitch, yaw] in radians

    def copy(self) -> DroneState:
        """Return a deep copy of the state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
        )

    def to_array(self) -> np.ndarray:
        """Convert state to a flat numpy array."""
        return np.concatenate([self.position, self.velocity, self.orientation]).astype(
            np.float32
        )


@dataclass
class Target:
    """Target position in the world."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def distance_to(self, position: np.ndarray) -> float:
        """Calculate Euclidean distance to a position."""
        return float(np.linalg.norm(self.position - position))

    def direction_to(self, position: np.ndarray) -> np.ndarray:
        """Calculate unit direction vector from position to target."""
        diff = self.position - position
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return (diff / norm).astype(np.float32)


@dataclass
class Obstacle:
    """Axis-aligned box obstacle."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    size: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    # size: [half_width_x, half_width_y, half_width_z]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_bounds, max_bounds) of the obstacle."""
        min_bounds = self.position - self.size
        max_bounds = self.position + self.size
        return min_bounds, max_bounds

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the obstacle."""
        min_bounds, max_bounds = self.get_bounds()
        return bool(np.all(point >= min_bounds) and np.all(point <= max_bounds))

    def intersects_sphere(
        self, center: np.ndarray, radius: float
    ) -> bool:
        """Check if a sphere intersects this box obstacle."""
        min_bounds, max_bounds = self.get_bounds()
        # Find closest point on box to sphere center
        closest = np.clip(center, min_bounds, max_bounds)
        distance = np.linalg.norm(center - closest)
        return distance < radius
