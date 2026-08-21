"""Main drone environment class implementing the RL interface."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import EnvConfig, DroneState, Obstacle, Target
from .physics import PhysicsEngine
from .obstacles import ObstacleManager
from .reward import RewardCalculator


class DroneEnv:
    """Drone Navigation Environment.

    A lightweight 3D drone navigation environment for reinforcement learning.

    The environment follows the standard RL interface:
        state = env.reset()
        next_state, reward, done, info = env.step(action)

    State space (16-dimensional):
        [drone_x, drone_y, drone_z,
         velocity_x, velocity_y, velocity_z,
         roll, pitch, yaw,
         target_x, target_y, target_z,
         distance_to_target,
         direction_x, direction_y, direction_z]

    Action space (4-dimensional continuous):
        [throttle, pitch, roll, yaw]
        - throttle ∈ [0, 1]: vertical thrust
        - pitch ∈ [-1, 1]: forward/backward tilt
        - roll ∈ [-1, 1]: left/right tilt
        - yaw ∈ [-1, 1]: rotation around vertical axis

    Termination conditions:
        - target_reached: drone is within threshold of target
        - collision: drone hits an obstacle
        - out_of_bounds: drone leaves the world
        - timeout: maximum steps reached
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        """Initialize the drone environment.

        Args:
            config: Environment configuration. If None, uses default config.
        """
        self.config = config or EnvConfig()
        self._validate_config()

        # Initialize components
        self.physics = PhysicsEngine(self.config)
        self.obstacle_manager = ObstacleManager(self.config)
        self.reward_calculator = RewardCalculator(self.config)

        # State variables
        self._drone_state: Optional[DroneState] = None
        self._target: Optional[Target] = None
        self._step_count: int = 0
        self._rng: Optional[np.random.Generator] = None
        self._trajectory: list = []

        # Action space bounds
        self.action_low = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.config.dt}")
        if self.config.max_velocity <= 0:
            raise ValueError(
                f"max_velocity must be positive, got {self.config.max_velocity}"
            )
        if self.config.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.config.max_steps}")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Initial state as a numpy array of shape (16,).
        """
        # Set up random generator
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)
        else:
            self._rng = np.random.default_rng()

        # Reset step counter
        self._step_count = 0
        self._trajectory = []

        # Reset reward calculator
        self.reward_calculator.reset(0.0)

        # Initialize drone state
        self._drone_state = self._initialize_drone()

        # Generate target
        self._target = self._generate_target()

        # Generate obstacles
        self.obstacle_manager.generate(
            self._rng, self._drone_state.position, self._target.position
        )

        # Record initial position in trajectory
        self._trajectory.append(self._drone_state.position.copy())

        # Calculate initial distance and reset reward calculator
        initial_distance = self._target.distance_to(self._drone_state.position)
        self.reward_calculator.reset(initial_distance)

        # Return initial state
        return self._get_state()

    def _initialize_drone(self) -> DroneState:
        """Initialize the drone at a random start position.

        Returns:
            Initial drone state.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.config.world_bounds

        # Random position within bounds, avoiding edges
        margin = 1.0
        position = self._rng.uniform(
            low=[x_min + margin, y_min + margin, z_min + margin],
            high=[x_max - margin, y_max - margin, z_max - margin],
        )

        # Zero initial velocity and orientation
        velocity = np.zeros(3, dtype=np.float32)
        orientation = np.zeros(3, dtype=np.float32)

        return DroneState(
            position=position.astype(np.float32),
            velocity=velocity,
            orientation=orientation,
        )

    def _generate_target(self) -> Target:
        """Generate a target position.

        Returns:
            Target object.
        """
        if not self.config.target_randomize:
            # Fixed target position
            return Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))

        x_min, x_max, y_min, y_max, z_min, z_max = self.config.world_bounds

        # Random direction and distance
        direction = self._rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)

        distance = self._rng.uniform(
            self.config.target_min_distance,
            self.config.target_max_distance,
        )

        # Target position relative to drone start
        target_pos = self._drone_state.position + direction * distance

        # Clamp to world bounds
        target_pos = np.clip(
            target_pos,
            [x_min + 0.5, y_min + 0.5, z_min + 0.5],
            [x_max - 0.5, y_max - 0.5, z_max - 0.5],
        )

        return Target(position=target_pos.astype(np.float32))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action array of shape (4,).
                [throttle, pitch, roll, yaw]

        Returns:
            Tuple of (next_state, reward, done, info).
        """
        if self._drone_state is None or self._target is None:
            raise RuntimeError(
                "Environment not reset. Call env.reset() before env.step()."
            )

        # Validate and clamp action
        action = np.clip(action, self.action_low, self.action_high).astype(np.float32)

        # Apply action to get new state
        new_state = self.physics.apply_action(self._drone_state, action)

        # Check bounds
        in_bounds, bound_info = self.physics.check_bounds(new_state.position)
        out_of_bounds = not in_bounds

        # Check collision
        collision, colliding_obstacle = self.obstacle_manager.check_collision(
            new_state.position
        )

        # Check target reached
        distance_to_target = self._target.distance_to(new_state.position)
        target_reached = distance_to_target < self.config.target_reached_threshold

        # Calculate reward
        reward, reward_breakdown = self.reward_calculator.calculate(
            state=new_state,
            target=self._target,
            collision=collision,
            out_of_bounds=out_of_bounds,
            target_reached=target_reached,
        )

        # Determine termination
        done = False
        termination_reason = ""

        if target_reached:
            done = True
            termination_reason = "target_reached"
        elif collision:
            done = True
            termination_reason = "collision"
        elif out_of_bounds:
            done = True
            termination_reason = "out_of_bounds"
        elif self._step_count >= self.config.max_steps - 1:
            done = True
            termination_reason = "timeout"

        # Update state
        self._drone_state = new_state
        self._step_count += 1
        self._trajectory.append(new_state.position.copy())

        # Build info dict
        info = {
            "termination_reason": termination_reason,
            "step": self._step_count,
            "distance_to_target": distance_to_target,
            "reward_breakdown": reward_breakdown,
            "position": new_state.position.copy(),
            "velocity": new_state.velocity.copy(),
            "orientation": new_state.orientation.copy(),
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "target_reached": target_reached,
        }

        if collision and colliding_obstacle is not None:
            info["colliding_obstacle"] = {
                "position": colliding_obstacle.position.copy(),
                "size": colliding_obstacle.size.copy(),
            }

        if out_of_bounds:
            info["bound_info"] = bound_info

        # Get next state
        next_state = self._get_state()

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Build the state vector from current drone state and target.

        Returns:
            State vector of shape (16,).
        """
        if self._drone_state is None or self._target is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        drone = self._drone_state
        target = self._target

        # Derived values
        distance = target.distance_to(drone.position)
        direction = target.direction_to(drone.position)

        state = np.concatenate(
            [
                drone.position,      # 3: x, y, z
                drone.velocity,      # 3: vx, vy, vz
                drone.orientation,   # 3: roll, pitch, yaw
                target.position,     # 3: target_x, target_y, target_z
                [distance],          # 1: distance_to_target
                direction,           # 3: direction_x, direction_y, direction_z
            ]
        ).astype(np.float32)

        return state

    def get_state(self) -> np.ndarray:
        """Return the current state without stepping.

        Returns:
            Current state vector.
        """
        return self._get_state()

    def render(self) -> None:
        """Render the environment (placeholder for external renderer)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    @property
    def state_dim(self) -> int:
        """Dimension of the state space."""
        return 16

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return 4

    @property
    def action_space(self) -> Dict[str, Any]:
        """Description of the action space."""
        return {
            "type": "continuous",
            "shape": (4,),
            "low": self.action_low,
            "high": self.action_high,
            "dtype": np.float32,
            "names": ["throttle", "pitch", "roll", "yaw"],
            "ranges": {
                "throttle": "[0, 1]",
                "pitch": "[-1, 1]",
                "roll": "[-1, 1]",
                "yaw": "[-1, 1]",
            },
        }

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Description of the observation space."""
        return {
            "type": "continuous",
            "shape": (16,),
            "dtype": np.float32,
            "low": -np.inf,
            "high": np.inf,
        }

    def get_trajectory(self) -> np.ndarray:
        """Return the trajectory of the current episode.

        Returns:
            Array of shape (num_steps, 3) containing positions.
        """
        return np.array(self._trajectory, dtype=np.float32)

    def get_obstacles(self) -> list:
        """Return the current obstacles."""
        return self.obstacle_manager.get_obstacles()

    def get_target_position(self) -> np.ndarray:
        """Return the current target position."""
        if self._target is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._target.position.copy()

    def get_drone_state(self) -> Optional[DroneState]:
        """Return the current drone state."""
        return self._drone_state
