"""Main application for the Drone RL Environment.

This script provides a simple interactive visualization of the drone
navigating in the 3D environment using matplotlib.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from environment import DroneEnv, EnvConfig, Target
from visualization import DroneRenderer


def run_simulation(
    seed: int = 42,
    max_steps: int = 300,
    render: bool = True,
) -> None:
    """Run a simple drone simulation with visualization.

    Args:
        seed: Random seed for reproducibility.
        max_steps: Maximum number of steps to run.
        render: Whether to render the environment in real-time.
    """
    # Create environment
    config = EnvConfig(
        world_bounds=(-10, 10, -10, 10, 0, 10),
        max_steps=max_steps,
        num_obstacles=5,
        target_min_distance=3.0,
        target_max_distance=8.0,
        seed=seed,
    )

    env = DroneEnv(config=config)
    renderer = DroneRenderer(config)

    # Reset environment
    state = env.reset(seed=seed)
    target_pos = env.get_target_position()

    print("=" * 60)
    print("DRONE NAVIGATION SIMULATION")
    print("=" * 60)
    print(f"World bounds: X[-10, 10] Y[-10, 10] Z[0, 10]")
    print(f"Target position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
    print(f"Max steps: {max_steps}")
    print("=" * 60)

    # Setup figure
    if render:
        fig, ax = renderer.setup_figure()
        plt.ion()  # Interactive mode

    total_reward = 0.0
    step = 0
    done = False

    # Simple heuristic: move toward target with random exploration
    rng = np.random.default_rng(seed)

    while not done and step < max_steps:
        drone_pos = env.get_drone_state().position

        # Simple heuristic: mostly move toward target, sometimes explore
        direction = target_pos - drone_pos
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            direction = direction / distance

        # Mix of goal-directed and exploratory actions
        if rng.random() < 0.8:
            # Goal-directed action
            throttle = min(1.0, 0.5 + 0.5 / (distance + 0.5))
            action = np.array(
                [
                    throttle,
                    np.clip(direction[0] * 0.5, -1, 1),
                    np.clip(direction[1] * 0.5, -1, 1),
                    np.clip(direction[2] * 0.3, -1, 1),
                ],
                dtype=np.float32,
            )
        else:
            # Random exploratory action
            action = rng.uniform(
                low=env.action_low,
                high=env.action_high,
            ).astype(np.float32)

        # Step environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Print status
        drone_pos = env.get_drone_state().position
        status = (
            f"Step {step:3d} | "
            f"Pos: ({drone_pos[0]:+6.2f}, {drone_pos[1]:+6.2f}, {drone_pos[2]:+6.2f}) | "
            f"Dist: {info['distance_to_target']:6.2f} | "
            f"Reward: {reward:+7.2f} | "
            f"Total: {total_reward:+8.2f}"
        )

        if done:
            status += f" | TERMINATED: {info['termination_reason']}"

        print(status)

        # Render
        if render:
            target = Target(position=target_pos)
            renderer.render(
                drone_position=drone_pos,
                target=target,
                obstacles=env.get_obstacles(),
                trajectory=env.get_trajectory(),
                title=f"Drone Navigation - Step {step} | Reward: {total_reward:.1f}",
            )
            plt.pause(0.05)

    # Final summary
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")

    if done:
        print(f"Termination reason: {info['termination_reason']}")
        print(f"Final distance to target: {info['distance_to_target']:.2f}")
    else:
        print("Termination: max steps reached")

    # Final render
    if render:
        target = Target(position=target_pos)
        renderer.render(
            drone_position=env.get_drone_state().position,
            target=target,
            obstacles=env.get_obstacles(),
            trajectory=env.get_trajectory(),
            title=f"Final State - Step {step} | Reward: {total_reward:.1f}",
        )
        plt.ioff()
        renderer.show()
    else:
        # Console-based summary
        print("\nFinal drone position:")
        print(f"  X: {env.get_drone_state().position[0]:.2f}")
        print(f"  Y: {env.get_drone_state().position[1]:.2f}")
        print(f"  Z: {env.get_drone_state().position[2]:.2f}")
        print(f"\nTarget position:")
        print(f"  X: {target_pos[0]:.2f}")
        print(f"  Y: {target_pos[1]:.2f}")
        print(f"  Z: {target_pos[2]:.2f}")
        print(f"\nDistance to target: {info['distance_to_target']:.2f}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Drone Navigation Simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum steps (default: 300)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable matplotlib rendering (console only)",
    )

    args = parser.parse_args()

    run_simulation(
        seed=args.seed,
        max_steps=args.steps,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
