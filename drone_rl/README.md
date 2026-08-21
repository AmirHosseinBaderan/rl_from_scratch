# Drone RL Environment

A lightweight custom Drone Navigation Simulator built for reinforcement learning education. This environment provides a clean, algorithm-independent interface for training RL agents (DQN, PPO, etc.) to navigate a drone in a 3D world.

## Project Goal

Build a simplified drone navigation environment that allows an external RL agent to:

1. Reset the environment
2. Receive the current state
3. Send an action/command
4. Receive: next state, reward, done, info
5. Repeat until episode termination

The environment is completely independent from the RL algorithm.

## Environment Architecture

```
drone_rl/
├── environment/
│   ├── __init__.py
│   ├── config.py        # Data structures and configuration
│   ├── physics.py       # Simplified motion model
│   ├── obstacles.py     # Obstacle generation and collision
│   ├── reward.py        # Reward calculation
│   └── drone_env.py     # Main environment class (RL interface)
│
├── visualization/
│   ├── __init__.py
│   └── renderer.py      # 3D matplotlib visualization
│
├── tests/
│   ├── test_environment.py
│   ├── test_physics.py
│   ├── test_rewards.py
│   └── test_obstacles.py
│
├── examples/
│   └── random_agent.py  # Random agent integration test
│
├── requirements.txt
└── README.md
```

## State Space

The environment returns a 16-dimensional state vector:

```
[
    drone_x,           # 0: Drone X position
    drone_y,           # 1: Drone Y position
    drone_z,           # 2: Drone Z position (altitude)
    velocity_x,        # 3: Velocity in X
    velocity_y,        # 4: Velocity in Y
    velocity_z,        # 5: Velocity in Z
    roll,              # 6: Roll angle (radians)
    pitch,             # 7: Pitch angle (radians)
    yaw,               # 8: Yaw angle (radians)
    target_x,          # 9: Target X position
    target_y,          # 10: Target Y position
    target_z,          # 11: Target Z position
    distance_to_target,# 12: Euclidean distance to target
    direction_x,       # 13: Unit direction X toward target
    direction_y,       # 14: Unit direction Y toward target
    direction_z        # 15: Unit direction Z toward target
]
```

## Action Space

The environment accepts a 4-dimensional continuous action:

```
[throttle, pitch, roll, yaw]

- throttle ∈ [0, 1]:    Vertical thrust (0 = no thrust, 1 = max thrust)
- pitch ∈ [-1, 1]:      Forward/backward tilt
- roll ∈ [-1, 1]:       Left/right tilt
- yaw ∈ [-1, 1]:        Rotation around vertical axis
```

## Reward Design

The reward function encourages:

1. **Distance reduction**: Positive reward proportional to distance reduction
2. **Target reached**: +100 reward
3. **Collision**: -100 penalty
4. **Out of bounds**: -50 penalty
5. **Step penalty**: -0.1 per step (encourages efficiency)

Mathematically:

```
reward = (previous_distance - current_distance) * distance_scale
       + target_reached * 100
       + collision * (-100)
       + out_of_bounds * (-50)
       + (-0.1)
```

## Termination Conditions

An episode terminates when:

- `target_reached`: Drone is within `target_reached_threshold` of target
- `collision`: Drone hits an obstacle
- `out_of_bounds`: Drone leaves the world boundaries
- `timeout`: Maximum number of steps reached

## Physics Simplification

The environment uses a simplified deterministic motion model:

- Fixed timestep: `dt = 0.1`
- Thrust-based vertical motion
- Drag-based damping
- Orientation-based horizontal motion
- No realistic aerodynamics
- No physics engine dependency

## World Boundaries

Default 3D Cartesian world:

```
X ∈ [-10, 10]
Y ∈ [-10, 10]
Z ∈ [0, 10]
```

## How to Run

### Install Dependencies

```bash
cd drone_rl
pip install -r requirements.txt
```

### Run Random Agent Demo

```bash
python -m drone_rl.examples.random_agent
```

With options:

```bash
python -m drone_rl.examples.random_agent --seed 42 --episodes 3 --save-plot
```

### Run Tests

```bash
cd drone_rl
pytest tests/ -v
```

## How a Future DQN/PPO Agent Will Connect

The environment exposes a clean RL interface:

```python
from drone_rl.environment import DroneEnv

# Create environment
env = DroneEnv()

# Reset
state = env.reset(seed=42)

# Interaction loop
done = False
while not done:
    # Agent chooses action (DQN, PPO, etc.)
    action = agent.choose_action(state)

    # Environment step
    next_state, reward, done, info = env.step(action)

    # Agent learns
    agent.learn(state, action, reward, next_state, done)

    state = next_state
```

### DQN Compatibility (Discrete Actions)

DQN requires discrete actions. An external action adapter can be used:

```python
class DiscreteActionAdapter:
    """Maps discrete actions to continuous drone actions."""

    ACTIONS = {
        0: [0.5, 0.0, 0.0, 0.0],   # Hover
        1: [0.5, 1.0, 0.0, 0.0],   # Forward
        2: [0.5, -1.0, 0.0, 0.0],  # Backward
        3: [0.5, 0.0, 1.0, 0.0],   # Right
        4: [0.5, 0.0, -1.0, 0.0],  # Left
        5: [1.0, 0.0, 0.0, 0.0],   # Ascend
        6: [0.0, 0.0, 0.0, 0.0],   # Descend
    }

    def __init__(self, num_actions: int = 7):
        self.num_actions = num_actions

    def to_continuous(self, discrete_action: int) -> np.ndarray:
        return np.array(self.ACTIONS[discrete_action], dtype=np.float32)
```

This adapter is **completely separate** from the environment.

### PPO Compatibility (Continuous Actions)

PPO works natively with continuous actions:

```python
from drone_rl.environment import DroneEnv

env = DroneEnv()
state = env.reset()

# PPO outputs continuous actions directly
action, log_prob, value = policy(state)
next_state, reward, done, info = env.step(action)
```

## Why the Environment is Algorithm-Independent

The environment:

1. **Does not know** whether the agent is DQN, PPO, Q-Learning, or another algorithm
2. **Only exposes** the standard RL interface: `reset()`, `step(action)`
3. **Has no dependencies** on any RL framework
4. **Uses standard** NumPy arrays for states and actions
5. **Is fully configurable** through `EnvConfig`

This design allows you to:
- Swap RL algorithms without changing the environment
- Test different algorithms on the same task
- Focus on algorithm development rather than environment debugging

## Configuration

All simulation parameters are configurable through `EnvConfig`:

```python
from drone_rl.environment import EnvConfig

config = EnvConfig(
    world_bounds=(-10, 10, -10, 10, 0, 10),
    dt=0.1,
    max_velocity=5.0,
    max_steps=500,
    num_obstacles=5,
    target_randomize=True,
    obstacle_randomize=True,
    seed=42,
)

env = DroneEnv(config=config)
```

## Example State

```python
env = DroneEnv()
state = env.reset(seed=42)
print(state)
# array([ 2.345,  3.456,  5.678,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
#         0.   ,  7.891,  8.123,  6.234,  6.789,  0.678,  0.567,  0.456],
#       dtype=float32)
```

## Example Action

```python
# Hover with slight forward pitch
action = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float32)
```

## Example Step Result

```python
env = DroneEnv()
state = env.reset(seed=42)
action = np.array([0.5, 0.1, 0.0, 0.0], dtype=np.float32)
next_state, reward, done, info = env.step(action)

print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")
```

## Test Results

Run tests with:

```bash
pytest tests/ -v
```

Tests cover:
- Environment reset and step
- State and action shapes
- Action clipping
- Physics simulation
- Reward calculation
- Obstacle generation and collision
- Termination conditions
- Deterministic behavior with seeds
