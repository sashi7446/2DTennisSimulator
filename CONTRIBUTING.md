# Contributing to 2D Tennis Simulator

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sashi7446/2DTennisSimulator.git
cd 2DTennisSimulator
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Development dependencies (includes testing and linting tools)
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
# Run tests
python run_tests.py

# Run the simulator
python main.py --mode list
```

## Code Style

This project uses automated tools to maintain consistent code quality:

### Formatting with Black

```bash
# Check formatting
black --check .

# Auto-format code
black .
```

### Linting with Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking with MyPy

```bash
mypy .
```

### Pre-commit Checklist

Before submitting a PR, ensure:

1. All tests pass: `python run_tests.py`
2. Code is formatted: `black .`
3. No linting errors: `ruff check .`
4. Type hints are valid: `mypy .`

## Running Tests

### All Tests

```bash
python run_tests.py
```

### Specific Test Files

```bash
python -m unittest tests.test_game -v
python -m unittest tests.test_agents -v
python -m unittest tests.test_cli -v
```

### Test Categories

- `test_game.py` - Core game logic
- `test_ball.py` - Ball physics and in-flag system
- `test_player.py` - Player movement and hitting
- `test_field.py` - Field and in-area logic
- `test_agents.py` - Agent implementations
- `test_env.py` - Gymnasium environment
- `test_cli.py` - CLI integration

## Project Structure

```
2DTennisSimulator/
├── main.py              # CLI entry point
├── config.py            # Configuration management
├── game.py              # Core game logic
├── ball.py              # Ball physics
├── player.py            # Player mechanics
├── field.py             # Field definitions
├── env.py               # Gymnasium environments
├── renderer.py          # Visualization
├── agents/              # Agent implementations
│   ├── __init__.py      # Public API
│   ├── base.py          # Base class
│   ├── chase.py         # Rule-based agents
│   ├── neural.py        # Learning agents
│   └── ...
├── tests/               # Test suite
├── pyproject.toml       # Tool configuration
└── requirements*.txt    # Dependencies
```

## Making Changes

### Adding a New Agent

1. Create `agents/my_agent.py`:

```python
from agents.base import Agent, AgentConfig

class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.config = AgentConfig(
            name="MyAgent",
            agent_type="my_agent",
            description="Description of my agent"
        )

    def act(self, observation: dict) -> tuple[int, float]:
        # Implement decision logic
        # Return (movement_direction, hit_angle)
        return 16, 0.0  # Stay still, hit straight

    def learn(self, reward: float, done: bool) -> None:
        # Implement learning logic (optional)
        pass
```

2. Register in `agents/__init__.py`:

```python
from agents.my_agent import MyAgent

__all__ = [..., "MyAgent"]
```

3. Add to `main.py` in `create_agent()`:

```python
elif agent_type == "my_agent":
    agent = MyAgent()
```

4. Add tests in `tests/test_agents.py`

### Modifying Game Mechanics

1. Update the relevant module (`game.py`, `ball.py`, `player.py`, `field.py`)
2. Update corresponding tests
3. Update documentation if behavior changes
4. Verify all tests pass

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clear, focused commits
- Follow code style guidelines
- Add/update tests as needed

### 3. Test Your Changes

```bash
# Run all checks
black .
ruff check .
mypy .
python run_tests.py
```

### 4. Submit PR

- Use a clear, descriptive title
- Describe what changes were made and why
- Reference any related issues
- Ensure CI passes

### PR Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
- [ ] All tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing performed

## Related Issues
Fixes #123
```

## Reporting Issues

### Bug Reports

Please include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

### Feature Requests

Please include:
- Clear description of the feature
- Use case / motivation
- Possible implementation approach (optional)

## Code Review Guidelines

When reviewing PRs, consider:

1. **Correctness** - Does the code work as intended?
2. **Tests** - Are there adequate tests?
3. **Style** - Does it follow project conventions?
4. **Performance** - Any obvious performance issues?
5. **Documentation** - Are changes documented?

## Questions?

Feel free to open an issue for questions or discussions about contributing.

Thank you for helping improve 2D Tennis Simulator!
