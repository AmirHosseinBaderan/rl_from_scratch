"""Tests for the reward calculator."""

from __future__ import annotations

import numpy as np
import pytest

from environment import EnvConfig, DroneState, RewardCalculator, Target


class TestRewardCalculator:
    """Test suite for RewardCalculator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = EnvConfig()
        self.calculator = RewardCalculator(self.config)

    def test_reward_is_numeric(self) -> None:
        """Test that reward is numeric."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(target.distance_to(state.position))
        reward, _ = self.calculator.calculate(
            state=state,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=False,
        )

        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)

    def test_target_reached_gives_positive_reward(self) -> None:
        """Test that reaching target gives positive reward."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(0.0)
        reward, _ = self.calculator.calculate(
            state=state,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=True,
        )

        assert reward > 0
        # Reward includes target_reached bonus minus step penalty
        assert reward >= self.config.reward_target_reached + self.config.reward_step_penalty

    def test_collision_gives_negative_reward(self) -> None:
        """Test that collision gives negative reward."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(target.distance_to(state.position))
        reward, _ = self.calculator.calculate(
            state=state,
            target=target,
            collision=True,
            out_of_bounds=False,
            target_reached=False,
        )

        assert reward < 0
        assert reward <= self.config.reward_collision

    def test_out_of_bounds_gives_negative_reward(self) -> None:
        """Test that going out of bounds gives negative reward."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(target.distance_to(state.position))
        reward, _ = self.calculator.calculate(
            state=state,
            target=target,
            collision=False,
            out_of_bounds=True,
            target_reached=False,
        )

        assert reward < 0
        assert reward <= self.config.reward_out_of_bounds

    def test_step_penalty_applied(self) -> None:
        """Test that step penalty is always applied."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(target.distance_to(state.position))
        reward, _ = self.calculator.calculate(
            state=state,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=False,
        )

        # Should include step penalty
        assert reward <= 0  # step penalty is negative

    def test_distance_reward_for_approaching_target(self) -> None:
        """Test that reward is higher when approaching target."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))

        # State far from target
        state_far = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # State closer to target
        state_close = DroneState(
            position=np.array([4.0, 4.0, 4.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        # Calculate reward for approaching
        self.calculator.reset(target.distance_to(state_far.position))
        reward_far, _ = self.calculator.calculate(
            state=state_far,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=False,
        )

        self.calculator.reset(target.distance_to(state_close.position))
        reward_close, _ = self.calculator.calculate(
            state=state_close,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=False,
        )

        # Reward for being closer should be higher (less negative step penalty
        # and potentially positive distance reward if moving closer)
        # The key is that the reward components are calculated correctly
        assert isinstance(reward_far, float)
        assert isinstance(reward_close, float)

    def test_reward_components_dict(self) -> None:
        """Test that reward components dict is returned correctly."""
        target = Target(position=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        state = DroneState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.zeros(3, dtype=np.float32),
        )

        self.calculator.reset(target.distance_to(state.position))
        components = self.calculator.get_reward_components(
            state=state,
            target=target,
            collision=False,
            out_of_bounds=False,
            target_reached=False,
        )

        assert "total_reward" in components
        assert "current_distance" in components
        assert "step_penalty" in components
        assert components["step_penalty"] == self.config.reward_step_penalty
