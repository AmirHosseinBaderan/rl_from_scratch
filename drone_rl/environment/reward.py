"""Reward calculation for the drone environment."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import EnvConfig, DroneState, Target


class RewardCalculator:
    """Calculates rewards for drone navigation.

    Reward design principles:
    1. Positive reward for reducing distance to target
    2. Large positive reward for reaching target
    3. Strong negative reward for collision
    4. Negative reward for leaving bounds
    5. Small step penalty to encourage efficiency
    """

    def __init__(self, config: EnvConfig) -> None:
        """Initialize the reward calculator.

        Args:
            config: Environment configuration.
        """
        self.config = config
        self._previous_distance: Optional[float] = None

    def reset(self, initial_distance: float) -> None:
        """Reset the reward calculator for a new episode.

        Args:
            initial_distance: Initial distance to target.
        """
        self._previous_distance = initial_distance

    def calculate(
        self,
        state: DroneState,
        target: Target,
        collision: bool,
        out_of_bounds: bool,
        target_reached: bool,
    ) -> Tuple[float, str]:
        """Calculate the reward for the current step.

        Args:
            state: Current drone state.
            target: Target object.
            collision: Whether the drone collided with an obstacle.
            out_of_bounds: Whether the drone left the world bounds.
            target_reached: Whether the drone reached the target.

        Returns:
            Tuple of (reward, reward_breakdown_string).
        """
        reward = 0.0
        breakdown_parts = []

        current_distance = target.distance_to(state.position)

        # 1. Distance-based reward (shaping)
        if self._previous_distance is not None:
            distance_delta = self._previous_distance - current_distance
            distance_reward = distance_delta * self.config.reward_distance_scale
            reward += distance_reward
            breakdown_parts.append(f"dist={distance_reward:.3f}")

        self._previous_distance = current_distance

        # 2. Target reached reward
        if target_reached:
            reward += self.config.reward_target_reached
            breakdown_parts.append(f"target={self.config.reward_target_reached:.1f}")

        # 3. Collision penalty
        if collision:
            reward += self.config.reward_collision
            breakdown_parts.append(f"collision={self.config.reward_collision:.1f}")

        # 4. Out of bounds penalty
        if out_of_bounds:
            reward += self.config.reward_out_of_bounds
            breakdown_parts.append(f"oob={self.config.reward_out_of_bounds:.1f}")

        # 5. Step penalty (encourages efficiency)
        reward += self.config.reward_step_penalty
        breakdown_parts.append(f"step={self.config.reward_step_penalty:.3f}")

        breakdown = " | ".join(breakdown_parts)
        return reward, breakdown

    def get_reward_components(
        self,
        state: DroneState,
        target: Target,
        collision: bool,
        out_of_bounds: bool,
        target_reached: bool,
    ) -> dict:
        """Return detailed reward components for debugging.

        Args:
            state: Current drone state.
            target: Target object.
            collision: Whether the drone collided.
            out_of_bounds: Whether the drone left bounds.
            target_reached: Whether the target was reached.

        Returns:
            Dictionary of reward components.
        """
        current_distance = target.distance_to(state.position)
        distance_delta = (
            (self._previous_distance - current_distance)
            if self._previous_distance is not None
            else 0.0
        )

        return {
            "distance_delta": distance_delta,
            "distance_reward": distance_delta * self.config.reward_distance_scale,
            "target_reached_reward": (
                self.config.reward_target_reached if target_reached else 0.0
            ),
            "collision_penalty": self.config.reward_collision if collision else 0.0,
            "out_of_bounds_penalty": (
                self.config.reward_out_of_bounds if out_of_bounds else 0.0
            ),
            "step_penalty": self.config.reward_step_penalty,
            "total_reward": (
                distance_delta * self.config.reward_distance_scale
                + (self.config.reward_target_reached if target_reached else 0.0)
                + (self.config.reward_collision if collision else 0.0)
                + (self.config.reward_out_of_bounds if out_of_bounds else 0.0)
                + self.config.reward_step_penalty
            ),
            "current_distance": current_distance,
        }
