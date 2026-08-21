"""Physics engine for the drone simulation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import EnvConfig, DroneState


class PhysicsEngine:
    """Simplified physics engine for drone motion.

    Uses a deterministic, discrete-time model with:
    - Thrust-based vertical motion
    - Drag-based damping
    - Orientation-based horizontal motion
    """

    def __init__(self, config: EnvConfig) -> None:
        """Initialize the physics engine.

        Args:
            config: Environment configuration.
        """
        self.config = config

    def apply_action(
        self, state: DroneState, action: np.ndarray
    ) -> DroneState:
        """Apply an action to the drone state.

        Action format: [throttle, pitch, roll, yaw]
        - throttle ∈ [0, 1]: vertical thrust
        - pitch ∈ [-1, 1]: forward/backward tilt
        - roll ∈ [-1, 1]: left/right tilt
        - yaw ∈ [-1, 1]: rotation around vertical axis

        Args:
            state: Current drone state.
            action: Action array of shape (4,).

        Returns:
            New drone state after applying action.
        """
        throttle, pitch, roll, yaw = action

        # Clamp action values
        throttle = float(np.clip(throttle, 0.0, 1.0))
        pitch = float(np.clip(pitch, -1.0, 1.0))
        roll = float(np.clip(roll, -1.0, 1.0))
        yaw = float(np.clip(yaw, -1.0, 1.0))

        dt = self.config.dt

        # --- Update orientation ---
        # Yaw rotation (around z-axis)
        new_yaw = state.orientation[2] + yaw * self.config.max_yaw_rate * dt

        # Pitch and roll affect orientation but we keep it simple
        # In a real drone, pitch/roll would change the thrust direction
        new_pitch = state.orientation[1] + pitch * self.config.max_pitch_rate * dt
        new_roll = state.orientation[0] + roll * self.config.max_roll_rate * dt

        # Clamp pitch and roll to reasonable angles
        new_pitch = float(np.clip(new_pitch, -np.pi / 4, np.pi / 4))
        new_roll = float(np.clip(new_roll, -np.pi / 4, np.pi / 4))

        new_orientation = np.array([new_roll, new_pitch, new_yaw], dtype=np.float32)

        # --- Compute thrust direction ---
        # Thrust is primarily vertical, but tilted by pitch and roll
        # This is a simplified model
        thrust_direction = np.array(
            [
                np.sin(new_roll) * np.cos(new_pitch),
                np.sin(new_pitch),
                np.cos(new_roll) * np.cos(new_pitch),
            ],
            dtype=np.float32,
        )

        # --- Compute acceleration ---
        # Thrust acceleration
        thrust_acc = thrust_direction * throttle * self.config.throttle_force

        # Gravity
        gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        # Drag (proportional to velocity)
        drag = -self.config.drag_coefficient * state.velocity

        # Total acceleration
        acceleration = thrust_acc + gravity + drag

        # Clamp acceleration
        acc_magnitude = np.linalg.norm(acceleration)
        if acc_magnitude > self.config.max_acceleration:
            acceleration = (
                acceleration / acc_magnitude * self.config.max_acceleration
            )

        # --- Update velocity ---
        new_velocity = state.velocity + acceleration * dt

        # Clamp velocity
        vel_magnitude = np.linalg.norm(new_velocity)
        if vel_magnitude > self.config.max_velocity:
            new_velocity = (
                new_velocity / vel_magnitude * self.config.max_velocity
            )

        # --- Update position ---
        new_position = state.position + new_velocity * dt

        return DroneState(
            position=new_position.astype(np.float32),
            velocity=new_velocity.astype(np.float32),
            orientation=new_orientation.astype(np.float32),
        )

    def check_bounds(
        self, position: np.ndarray
    ) -> Tuple[bool, str]:
        """Check if position is within world bounds.

        Args:
            position: Drone position.

        Returns:
            Tuple of (is_in_bounds, boundary_violation_info).
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.config.world_bounds

        if position[0] < x_min:
            return False, f"x_min ({position[0]:.2f} < {x_min})"
        if position[0] > x_max:
            return False, f"x_max ({position[0]:.2f} > {x_max})"
        if position[1] < y_min:
            return False, f"y_min ({position[1]:.2f} < {y_min})"
        if position[1] > y_max:
            return False, f"y_max ({position[1]:.2f} > {y_max})"
        if position[2] < z_min:
            return False, f"z_min ({position[2]:.2f} < {z_min})"
        if position[2] > z_max:
            return False, f"z_max ({position[2]:.2f} > {z_max})"

        return True, ""
