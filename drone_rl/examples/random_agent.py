"""Random agent demo for the drone environment.

This script demonstrates how to interact with the DroneEnv using a random agent.
It serves as an integration test and shows the standard RL interaction pattern:

    state = env.reset()
    while not done:
        action = random_action()
        next_state, reward, done, info = env.step(action)

This is NOT a learning agent - it just takes random actions to verify
that the environment interface works correctly.
"""

from __future__ import annotations

import argparse

import numpy as np

from environment import DroneEnv, EnvConfig, Target
from visualization import DroneRenderer


def run_random_agent(
    seed: int = 42,
    max_episodes: int = 1,
    render: bool = True,
    save_plot: bool = False,
) -> None:
    """Run a random agent in the drone environment.

    Args:
        seed: Random seed for reproducibility.
        max_episodes: Number of episodes to run.
        render: Whether to render the environment.
        save_plot: Whether to save the final plot to a file.
    """
    # Create environment with a smaller world for easier visualization
    config = EnvConfig(
        world_bounds=(-10, 10, -10, 10, 0, 10),
        max_steps=200,
        num_obstacles=5,
        target_min_distance=3.0,
        target_max_distance=8.0,
        seed=seed,
    )

    env = DroneEnv(config=config)
    renderer = DroneRenderer(config)

    for episode in range(max_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{max_episodes}")
        print(f"{'='*50}")

        state = env.reset(seed=seed + episode)
        print(f"Initial state shape: {state.shape}")
        print(f"Initial state: {state}")

        total_reward = 0.0
        step = 0
        done = False

        while not done:
            # Generate random action
            action = np.random.uniform(
                low=env.action_low,
                high=env.action_high,
            ).astype(np.float32)

            # Step the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # Print progress
            if step % 20 == 0 or done:
                drone_pos = env.get_drone_state().position
                print(
                    f"Step {step:3d} | "
                    f"Pos: ({drone_pos[0]:+.2f}, "
                    f"{drone_pos[1]:+.2f}, "
                    f"{drone_pos[2]:+.2f}) | "
                    f"Dist: {info['distance_to_target']:.2f} | "
                    f"Reward: {reward:+.2f} | "
                    f"Total: {total_reward:+.2f} | "
                    f"Done: {done}"
                )

            # Render
            if render and step % 5 == 0:
                target = Target(position=env.get_target_position())
                renderer.render(
                    drone_position=env.get_drone_state().position,
                    target=target,
                    obstacles=env.get_obstacles(),
                    trajectory=env.get_trajectory(),
                    title=f"Episode {episode + 1} - Step {step}",
                )
                import matplotlib.pyplot

                matplotlib.pyplot.pause(0.01)

        # Print episode summary
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Termination reason: {info['termination_reason']}")
        print(f"  Final distance to target: {info['distance_to_target']:.2f}")

        # Final render
        if render:
            target = Target(position=env.get_target_position())
            renderer.render(
                drone_position=env.get_drone_state().position,
                target=target,
                obstacles=env.get_obstacles(),
                trajectory=env.get_trajectory(),
                title=f"Episode {episode + 1} - Finished ({info['termination_reason']})",
            )

            if save_plot:
                renderer.save(f"episode_{episode + 1}_final.png")
                print(f"  Saved plot to episode_{episode + 1}_final.png")

            if max_episodes == 1:
                renderer.show()
            else:
                import matplotlib.pyplot

                matplotlib.pyplot.pause(1.0)

        renderer.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Random agent demo for the drone environment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save final plot to file",
    )

    args = parser.parse_args()

    run_random_agent(
        seed=args.seed,
        max_episodes=args.episodes,
        render=not args.no_render,
        save_plot=args.save_plot,
    )


if __name__ == "__main__":
    main()
