# 2D Tennis Simulator

A simulator where AI agents compete and learn from each other in a simplified 2D tennis environment. Watch as basic tactics like "returning to home position" emerge naturally through reinforcement learning.

**[日本語版 README はこちら](README_ja.md)**

## Features

- Real-time AI vs AI matches with visualization
- Policy Gradient (REINFORCE) learning agents
- Live reward graphs and debug overlays
- Agent save/load functionality
- Multiple agent types (rule-based and learning)
- Gymnasium-compatible environment for RL research

## Installation

```bash
# Clone the repository
git clone https://github.com/sashi7446/2DTennisSimulator.git
cd 2DTennisSimulator

# Install dependencies
pip install pygame numpy

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

## Quick Start

```bash
# Watch AI agents compete
python main.py

# Train a learning agent vs rule-based AI (with debug display)
python main.py --agent-a neural --agent-b chase --debug

# List available agent types
python main.py --mode list
```

## Agent Types

| Type | Description |
|------|-------------|
| `chase` | Simple ball-chasing AI (default) |
| `smart` | Improved chase with better positioning |
| `random` | Random actions (baseline comparison) |
| `neural` | Policy Gradient neural network agent |
| `transformer` | Advanced Transformer-based model |
| `baseliner` | Defensive baseline strategy |
| `positional` | Position-aware tactical agent |

## Usage

### Visual Mode (Watch Games)

```bash
# Basic match (chase vs chase)
python main.py

# Specify agents
python main.py --agent-a neural --agent-b smart

# Debug mode (shows reward graphs, state info)
python main.py --agent-a neural --agent-b chase --debug

# Adjust ball speed and game parameters
python main.py --speed 7.0
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `D` | Toggle debug display |
| `S` | Save agents (when --save-dir specified) |
| `R` | Reset game |
| `1-4` | Speed control (1x/2x/4x/Max) |
| `SPACE` | Pause game |
| `N` | Step forward (when paused) |
| `ESC` | Quit |

### Headless Training Mode

Fast training without graphics:

```bash
# Train for 100 episodes
python main.py --mode headless --agent-a neural --agent-b chase --episodes 100

# Train and save agents periodically
python main.py --mode headless --agent-a neural --agent-b chase --episodes 500 --save-dir saved_agents
```

### Benchmark Mode

Train headless, then visualize the trained agents:

```bash
python main.py --mode benchmark --agent-a neural --agent-b smart --episodes 300
```

### Save and Load Agents

```bash
# Train and save
python main.py --agent-a neural --agent-b chase --save-dir my_agents --debug

# Load saved agents
python main.py --agent-a my_agents/agent_a_neural --agent-b chase

# Match between saved agents
python main.py --agent-a saved/champion_v1 --agent-b saved/champion_v2
```

## Gymnasium Integration

Use as a standard Gymnasium environment for RL research:

```python
from env import TennisEnv, SinglePlayerTennisEnv
import numpy as np

# Two-player environment
env = TennisEnv(render_mode="human")
obs, info = env.reset()

# Action format: (movement_direction 0-16, hit_angle)
action = (0, np.array([45.0]))
obs, reward, terminated, truncated, info = env.step(action)

# Single-player environment (opponent is AI-controlled)
env = SinglePlayerTennisEnv(opponent_policy="chase")
obs, info = env.reset()

# Action format for single player: numpy array with [movement, hit_angle]
action = np.array([8, 45.0])  # movement 8 = stay, 45 degree hit angle
obs, reward, terminated, truncated, info = env.step(action)
```

## Game Rules

### Field
- Rectangular playing field enclosed by walls
- Two "in areas" in the center (3:2 ratio with gap between)

### In-Flag System
- Ball passes through in-area → In-flag ON
- Player hits ball → In-flag resets to OFF

### Scoring
When ball reaches a wall:
- In-flag ON → Hitter **scores** a point
- In-flag OFF → Hitter **loses** a point (out)

### Hit Conditions
A player can only hit when:
1. Ball is within reach distance
2. In-flag is ON

## Configuration

Adjust game parameters in `config.py`:

```python
Config(
    # Field dimensions
    field_width=800,
    field_height=400,

    # In-area settings
    area_width=150,
    area_height=100,
    area_gap=100,

    # Ball physics
    ball_speed=5.0,
    serve_angle_range=15.0,

    # Player attributes
    player_speed=3.0,
    reach_distance=30.0,

    # Reward shaping
    reward_point_win=1.0,
    reward_point_lose=-1.0,
    reward_rally=0.1,
)
```

## Project Structure

```
├── main.py          # CLI entry point
├── config.py        # Game configuration
├── field.py         # Field and in-areas
├── ball.py          # Ball physics and in-flag
├── player.py        # Player movement and hitting
├── game.py          # Core game logic
├── renderer.py      # Pygame rendering (with debug overlays)
├── env.py           # Gymnasium environments
├── debug.py         # Debug logging and validation
├── agents/          # Agent implementations
│   ├── __init__.py
│   ├── base.py      # Base class (save/load)
│   ├── chase.py     # ChaseAgent, SmartChaseAgent
│   ├── random_agent.py  # RandomAgent
│   ├── neural.py    # NeuralAgent (Policy Gradient)
│   ├── transformer.py   # TransformerAgent
│   ├── baseliner.py # BaselinerAgent
│   └── positional.py    # PositionalAgent
└── tests/           # Unit tests (200+ tests)
```

## Running Tests

```bash
# Run all tests
python run_tests.py

# Or using unittest directly
python -m unittest discover tests/ -v
```

## Creating Custom Agents

Extend the `Agent` base class to create your own AI:

```python
from agents.base import Agent

class MyAgent(Agent):
    def act(self, observation):
        # observation is a dict with game state
        # Return (movement_direction, hit_angle)
        # movement: 0-15 (22.5° increments), 16 (stay)
        # hit_angle: 0-360 degrees (float)
        return 16, 0.0

    def learn(self, reward, done):
        # Called after each step with reward
        pass
```

Register in `agents/__init__.py` and add to `create_agent()` in `main.py`.

## Debug Mode Features

Enable with `--debug` flag:

- Ball position, velocity, and in-flag state
- Player positions and states
- Reward graphs (4 types):
  - Player A cumulative rewards (per episode)
  - Player A 5-episode moving average
  - Player B cumulative rewards (per episode)
  - Player B 5-episode moving average
- Trajectory prediction overlay
- Grid overlay for positioning

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
