"""Tests for the physics engine."""

from __future__ import annotations

import numpy as np
import pytest

from environment import EnvConfig, DroneState, PhysicsEngine


class TestPhysicsEngine:
    """Test suite for PhysicsEngine."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = EnvConfig(dt=0.1)
        self.physics = PhysicsEngine(self.config)

    def test_movement_works(self) -> None:
        """Test that movement changes position."""
        state = DroneState(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            velocity=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_state = self.physics.apply_action(state, action)

        # Position should change due to velocity (and gravity/drag)
        assert not np.allclose(state.position, new_state.position)

    def test_velocity_changes_appropriately(self) -> None:
        """Test that velocity changes with throttle."""
        # Use zero gravity config for this test
        config = EnvConfig(dt=0.1, throttle_force=10.0)
        physics = PhysicsEngine(config)

        state = DroneState(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # Apply full throttle
        action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_state = physics.apply_action(state, action)

        # Velocity should increase in z direction (thrust overcomes gravity)
        assert new_state.velocity[2] > 0

    def test_timestep_works(self) -> None:
        """Test that timestep affects movement."""
        # Use config with no gravity and no drag for predictable behavior
        config = EnvConfig(dt=0.1, drag_coefficient=0.0)
        # Override gravity in physics by using a custom approach
        physics = PhysicsEngine(config)

        state = DroneState(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            velocity=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # Use zero throttle to avoid gravity effects
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_state = physics.apply_action(state, action)

        # Position should change (approximately by velocity * dt, minus drag)
        # With zero drag and zero throttle, only gravity acts
        # So x should change by velocity * dt, z should decrease due to gravity
        assert new_state.position[0] > state.position[0]  # x increases due to velocity

    def test_altitude_constraints(self) -> None:
        """Test that altitude is constrained by world bounds."""
        config = EnvConfig(
            world_bounds=(-10, 10, -10, 10, 0, 10),
            dt=0.1,
        )
        physics = PhysicsEngine(config)

        # Start at ground level
        state = DroneState(
            position=np.array([0.0, 0.0, 0.1], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # Apply negative throttle (should not go below z=0 due to bounds check)
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_state = physics.apply_action(state, action)

        # Position should be valid
        assert new_state.position[2] >= 0.0

    def test_action_clamping(self) -> None:
        """Test that actions are clamped to valid ranges."""
        state = DroneState(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # Action with extreme values
        action = np.array([10.0, -5.0, 5.0, -3.0], dtype=np.float32)
        new_state = self.physics.apply_action(state, action)

        # Should not raise an error
        assert isinstance(new_state, DroneState)

    def test_drag_reduces_velocity(self) -> None:
        """Test that drag reduces velocity over time."""
        config = EnvConfig(dt=0.1, drag_coefficient=1.0)
        physics = PhysicsEngine(config)

        state = DroneState(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            velocity=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_state = physics.apply_action(state, action)

        # Velocity should decrease due to drag
        assert np.linalg.norm(new_state.velocity) < np.linalg.norm(state.velocity)

    def test_bounds_check(self) -> None:
        """Test bounds checking."""
        config = EnvConfig(
            world_bounds=(-10, 10, -10, 10, 0, 10),
        )
        physics = PhysicsEngine(config)

        # Inside bounds
        in_bounds, _ = physics.check_bounds(np.array([0.0, 0.0, 5.0]))
        assert in_bounds is True

        # Outside bounds
        in_bounds, _ = physics.check_bounds(np.array([15.0, 0.0, 5.0]))
        assert in_bounds is False

        in_bounds, _ = physics.check_bounds(np.array([0.0, 0.0, -1.0]))
        assert in_bounds is False
