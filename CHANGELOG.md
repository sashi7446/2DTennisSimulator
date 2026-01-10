# Changelog

All notable changes to the 2D Tennis Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Progress bar display using `tqdm` for headless training mode
- Enhanced help text for CLI with detailed examples and descriptions
- This CHANGELOG.md file to track project changes
- CLI integration tests (`tests/test_cli.py`) with 27 tests covering headless mode, agent options, and error handling
- English README.md for international users
- Japanese README moved to README_ja.md
- CONTRIBUTING.md with development setup, coding guidelines, and PR process

### Changed
- Improved CLI help output with usage examples and detailed option descriptions
- Progress feedback now shows win rates during training
- Test suite expanded from 176 to 203 tests

### Fixed
- N/A

## [0.2.0] - 2026-01-10

### Added
- TASK_LIST.md with prioritized improvement suggestions and implementation guides
- GitHub Actions CI workflow for automated testing
- Benchmark mode: train headless then automatically launch debug visualization
- TransformerAgent: advanced Transformer-based learning agent
- Agent implementation cheat sheet in README.md

### Changed
- Revised CODE_REVIEW.md: updated status and corrected inaccurate reviews
- Renamed `in_flag` to `is_in` for better tennis terminology alignment
- Refactored `Config.to_dict()` to use `asdict()` instead of manual enumeration
- Updated court size configuration handling in agents

### Fixed
- Easy code review issues addressed
- Point scoring tests fixed by adjusting player B initial position
- Dictionary key access errors in neural.py now properly raise exceptions

## [0.1.0] - Initial Release

### Added
- Core game engine (Game, Ball, Player, Field)
- Multiple agent types: ChaseAgent, SmartChaseAgent, RandomAgent
- NeuralAgent with Policy Gradient (REINFORCE) learning
- Visual mode with pygame rendering
- Headless training mode for fast learning
- Debug mode with state overlay and reward graphs
- Agent save/load functionality
- Gymnasium environment wrapper (TennisEnv, SinglePlayerTennisEnv)
- Comprehensive test suite (96+ tests)
- Documentation (README.md, ARCHITECTURE.md, CODE_REVIEW.md)

### Features
- AI vs AI real-time matches
- Configurable game parameters
- In-area system for point scoring
- Reward system for reinforcement learning
- Debug visualization with graphs and overlays

---

## Version History Summary

| Version | Date | Key Features |
|---------|------|--------------|
| 0.2.0 | 2026-01-10 | Task list, CI/CD, Benchmark mode, Transformer agent |
| 0.1.0 | - | Initial release with core features |

---

*For detailed commit history, see `git log` or GitHub commit history.*
