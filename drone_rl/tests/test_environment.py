"""Tests for the main drone environment."""

from __future__ import annotations

import numpy as np
import pytest

from environment import DroneEnv, EnvConfig


class TestDroneEnv:
    """Test suite for DroneEnv."""

    def test_reset_returns_valid_state(self) -> None:
        """Test that reset returns a valid state array."""
        env = DroneEnv()
        state = env.reset(seed=42)

        assert isinstance(state, np.ndarray)
        assert state.shape == (16,)
        assert state.dtype == np.float32

    def test_state_shape_is_correct(self) -> None:
        """Test that state has the expected shape."""
        env = DroneEnv()
        state = env.reset(seed=42)

        # State should be 16-dimensional
        expected_shape = (16,)
        assert state.shape == expected_shape

    def test_action_shape_is_correct(self) -> None:
        """Test that action space has the expected shape."""
        env = DroneEnv()

        assert env.action_dim == 4
        assert env.action_space["shape"] == (4,)

    def test_action_clipping_works(self) -> None:
        """Test that actions are properly clipped."""
        env = DroneEnv()
        env.reset(seed=42)

        # Action with values outside bounds
        action = np.array([2.0, -3.0, 5.0, -2.0], dtype=np.float32)

        next_state, reward, done, info = env.step(action)

        # Should not raise an error
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_changes_drone_state(self) -> None:
        """Test that step changes the drone state."""
        env = DroneEnv()
        state1 = env.reset(seed=42)

        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        state2, reward, done, info = env.step(action)

        # States should be different
        assert not np.allclose(state1, state2)

    def test_reward_is_numeric(self) -> None:
        """Test that reward is a numeric value."""
        env = DroneEnv()
        env.reset(seed=42)

        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _ = env.step(action)

        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)

    def test_done_is_boolean(self) -> None:
        """Test that done is a boolean."""
        env = DroneEnv()
        env.reset(seed=42)

        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, done, _ = env.step(action)

        assert isinstance(done, bool)

    def test_info_contains_termination_info(self) -> None:
        """Test that info dict contains termination information."""
        env = DroneEnv()
        env.reset(seed=42)

        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, done, info = env.step(action)

        assert "termination_reason" in info
        assert "step" in info
        assert "distance_to_target" in info

    def test_deterministic_with_seed(self) -> None:
        """Test that environment is deterministic with same seed."""
        env1 = DroneEnv()
        state1 = env1.reset(seed=42)
        action = np.array([0.5, 0.1, 0.1, 0.1], dtype=np.float32)
        _, reward1, done1, info1 = env1.step(action)

        env2 = DroneEnv()
        state2 = env2.reset(seed=42)
        _, reward2, done2, info2 = env2.step(action)

        assert np.allclose(state1, state2)
        assert np.isclose(reward1, reward2)
        assert done1 == done2

    def test_target_reached_termination(self) -> None:
        """Test that episode terminates when drone is at target position."""
        config = EnvConfig(
            world_bounds=(-5, 5, -5, 5, 0, 5),
            target_reached_threshold=0.5,
            max_steps=10,
            num_obstacles=0,
        )
        env = DroneEnv(config=config)

        # Reset to generate target
        env.reset(seed=42)

        # Get the actual target position and place drone there
        target_pos = env.get_target_position()
        env._drone_state.position = target_pos.copy()

        # Take a step - should terminate because drone is at target
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, done, info = env.step(action)

        assert done is True
        assert info["termination_reason"] == "target_reached"

    def test_timeout_termination(self) -> None:
        """Test that episode terminates on timeout."""
        config = EnvConfig(max_steps=10)
        env = DroneEnv(config=config)
        env.reset(seed=42)

        for _ in range(15):
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            _, _, done, info = env.step(action)
            if done:
                assert info["termination_reason"] == "timeout"
                return

        pytest.fail("Episode should have terminated due to timeout")

    def test_step_without_reset_raises_error(self) -> None:
        """Test that step without reset raises an error."""
        env = DroneEnv()

        with pytest.raises(RuntimeError):
            env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    def test_get_trajectory(self) -> None:
        """Test that trajectory is recorded correctly."""
        env = DroneEnv()
        env.reset(seed=42)

        for _ in range(5):
            action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
            _, _, done, _ = env.step(action)
            if done:
                break

        trajectory = env.get_trajectory()
        assert trajectory.shape[1] == 3  # 3D positions
        assert trajectory.shape[0] >= 1  # At least initial position
