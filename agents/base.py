"""Base Agent class for 2D Tennis Simulator."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "unnamed"
    agent_type: str = "base"
    version: str = "1.0"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(**data)

    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AgentConfig":
        """Load config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class Agent(ABC):
    """
    Abstract base class for all agents.

    An agent receives observations and returns actions.
    Agents can be saved/loaded for reuse.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.player_id: Optional[int] = None  # 0 for A, 1 for B

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """
        Choose an action based on the observation.

        Args:
            observation: Game state from game.get_observation()

        Returns:
            Tuple of (movement_direction, hit_angle)
            - movement_direction: 0-15 for directions, 16 for stay
            - hit_angle: 0-360 degrees for hit direction
        """
        pass

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass

    def learn(self, reward: float, done: bool) -> None:
        """
        Update agent based on reward (for learning agents).

        Args:
            reward: Reward from last action
            done: Whether episode is finished
        """
        pass

    def set_player_id(self, player_id: int) -> None:
        """Set which player this agent controls (0=A, 1=B)."""
        self.player_id = player_id

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for display."""
        return {
            "name": self.config.name,
            "type": self.config.agent_type,
            "version": self.config.version,
            "description": self.config.description,
        }

    def save(self, directory: str) -> str:
        """
        Save agent to directory.

        Args:
            directory: Directory to save agent files

        Returns:
            Path to the saved agent directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        self.config.save(str(config_path))

        # Save weights (override in subclasses)
        self._save_weights(path)

        return str(path)

    def _save_weights(self, directory: Path) -> None:
        """Save agent-specific weights. Override in subclasses."""
        pass

    @classmethod
    def load(cls, directory: str) -> "Agent":
        """
        Load agent from directory.

        Args:
            directory: Directory containing agent files

        Returns:
            Loaded agent instance
        """
        path = Path(directory)
        config_path = path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")

        config = AgentConfig.load(str(config_path))

        # Create agent instance
        agent = cls(config)

        # Load weights (override in subclasses)
        agent._load_weights(path)

        return agent

    def _load_weights(self, directory: Path) -> None:
        """Load agent-specific weights. Override in subclasses."""
        pass


def get_agent_class(agent_type: str) -> type:
    """Get agent class by type name."""
    from agents.chase import ChaseAgent
    from agents.random_agent import RandomAgent

    agent_classes = {
        "chase": ChaseAgent,
        "random": RandomAgent,
    }

    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Available: {list(agent_classes.keys())}")

    return agent_classes[agent_type]


def load_agent(directory: str) -> Agent:
    """
    Load any agent from directory by reading its type from config.

    Args:
        directory: Directory containing agent files

    Returns:
        Loaded agent instance
    """
    config_path = Path(directory) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")

    config = AgentConfig.load(str(config_path))
    agent_class = get_agent_class(config.agent_type)

    return agent_class.load(directory)
