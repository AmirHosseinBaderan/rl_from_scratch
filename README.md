# rl_from_scratch

A collection of reinforcement learning projects implemented from scratch.

## Projects

### [Project 01: Q-Learning Grid World](./project_01_q_learning.md)

A complete implementation of the Q-Learning algorithm on a custom Grid World environment.

**Features:**
- Tabular Q-Learning with epsilon-greedy exploration
- 5×5 Grid World with obstacles, start, and goal positions
- Checkpointing (save/load best and latest Q-tables)
- TensorBoard logging for training metrics
- Policy evaluation and path visualization
- Comprehensive unit tests

**Quick Start:**

```bash
cd project_01_q_learning
pip install -r requirements.txt

# Train the agent
python main.py train

# Evaluate the trained agent
python main.py evaluate

# View training metrics
tensorboard --logdir runs/q_learning
```

For detailed documentation, see [project_01_q_learning.md](./project_01_q_learning.md).
