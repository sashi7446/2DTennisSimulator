# 2D Tennis Simulator

A simulator where AI agents compete and learn from each other in a simplified 2D tennis environment. Watch as basic tactics like "returning to home position" emerge naturally through reinforcement learning.

**[日本語版 README はこちら](README.md)**

**▶ Watch matches (works on phones): https://sashi7446.github.io/2DTennisSimulator/**

## Features

- Real-time AI vs AI matches with visualization
- Play-by-play commentary, rally counter and longest-rally records
- Synthesised hit sounds and impact effects (no audio assets)
- Replay viewer playable on a phone via GitHub Pages, no server
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

# Mute sound (already silent when no audio device is available)
python main.py --mute
```

### Reading the Screen

| Element | Meaning |
|---------|---------|
| Red circle (left) / blue circle (right) | Player A / Player B |
| Faint circle around a player | Reach. The ball must enter it to be returnable |
| Bright yellow ball | In. Hitting a wall now **scores** for whoever hit it |
| Dim yellow ball | Has not passed through an in-area. Hitting a wall **loses** the point (out) |
| Expanding ring | A return happened on that frame |
| Log at the bottom left | Play-by-play, newest first, fading with age |
| `Rally: 3 (best 12)` | Current rally length and the longest so far this session |
| `Game Over! Player B wins! (in)` | Result and why it ended (`in` = winner, `out` = error) |

Hit sounds are pitched by ball speed, and the point sting rises for `in` and
falls for `out`.

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

## Watch on a Phone (Replay Viewer)

**https://sashi7446.github.io/2DTennisSimulator/**

Recorded matches play back in a static HTML page. No server, no pygame, no
Python on the viewing device.

| Control | Action |
|---------|--------|
| Tap the court | Play / pause |
| ⏮ / ⏭ | Previous / next point |
| 1.0x | Playback speed (1.0x → 2.0x → 0.5x) |

The header shows the settings the match was recorded with, and a
🔥 LONGEST RALLY badge marks the longest rally in the recording.

### Recording from a phone (GitHub Actions)

Open **Actions → Record Replay → Run workflow** in the GitHub app or browser,
fill in the matchup, and the page updates about a minute later.

| Input | Meaning |
|-------|---------|
| `agent_a` / `agent_b` | Agent name, or a path to a saved agent |
| `points` | Number of points to record |
| `ball_speed` | Ball speed (blank = 15.0). Lower means longer rallies |
| `player_speed` | Player speed (blank = 4.0). **Raising it makes anticipation pointless** |
| `reach` | Reach distance (blank = 30.0) |

Agent names are free text, not a dropdown: a new agent becomes selectable as
soon as `create_agent()` knows about it, with no change to the workflow.

### Recording locally

```bash
# Writes docs/replay.js
python record_replay.py --agent-a smart --agent-b baseliner --points 20

# Longer rallies - lower the ball speed only, see the warning below
python record_replay.py --agent-a smart --agent-b smart --speed 8
```

Open `docs/index.html` directly (it works over `file://` too), or commit
`docs/replay.js` to publish. Pages is served from **Settings → Pages →
Source: main / docs**.

A replay stores only positions and hit events - a few KB per point for quick
points, around 20 KB for long rallies.

**Do not raise `player_speed`.** A player speed of 4 against a ball speed of 15
is what forces agents to anticipate instead of react, which is the behaviour
this project exists to observe. At player speed 8 chasing the ball is enough,
and `chase`'s win rate against `smart` climbs from 3% to 18%. Ball speed 8
combined with player speed 8 means points never end at all.

To lengthen rallies, lower the ball speed only and leave player speed at 4.0:
`smart vs smart` averages 3.1 exchanges at ball 15, 5.0 at ball 12 and 7.5 at
ball 8.

Two `baseliner` agents never finish a point even at default settings.
Recording stops after 1800 frames per point and warns when that happens.

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
    area_width=200,
    area_height=300,
    area_gap=250,

    # Ball physics
    ball_speed=15.0,
    serve_angle_range=15.0,

    # Player attributes
    player_speed=4.0,
    reach_distance=30.0,

    # Rewards (see Reward Triggers below)
    reward_point_win=1.0,
    reward_point_lose=-1.0,
    reward_rally=0.1,
    reward_in_area=0.05,
    reward_step=0.0,
)
```

## Reward Triggers

Rewards for reinforcement learning agents are triggered by 5 different events:

| Trigger | Parameter | Default | When |
|---------|-----------|---------|------|
| Hit | `reward_rally` | `0.1` | When a player hits the ball |
| In Area | `reward_in_area` | `0.05` | When hit ball passes through in-area |
| Point Win | `reward_point_win` | `1.0` | When a point is won |
| Point Loss | `reward_point_lose` | `-1.0` | When a point is lost |
| Time Step | `reward_step` | `0.0` | Every step (applied to both players) |

### Reward Design Examples

```python
# Default: Sparse rewards (focus on point win/loss)
Config()

# Encourage faster gameplay
Config(reward_step=-0.001)

# Emphasize rallying
Config(reward_rally=0.5, reward_in_area=0.2)

# Pure sparse rewards (points only)
Config(reward_rally=0.0, reward_in_area=0.0, reward_step=0.0)
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
├── audio.py         # Synthesised sound effects (numpy waveforms)
├── record_replay.py # Records matches into docs/replay.js
├── docs/            # Phone-friendly replay viewer (GitHub Pages)
│   ├── index.html   # Self-contained player, no dependencies
│   └── replay.js    # Recorded match data (generated)
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
Once `create_agent()` knows the name, the **Record Replay** workflow accepts it
too - no workflow edit needed.

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
