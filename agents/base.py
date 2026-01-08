"""Base Agent class for 2D Tennis Simulator."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class AgentConfig:
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
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AgentConfig":
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))


class Agent(ABC):
    """Abstract base: receives observations, returns (movement, hit_angle) actions."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.player_id: Optional[int] = None

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Return (movement_direction 0-16, hit_angle 0-360)."""
        pass

    def reset(self) -> None:
        pass

    def learn(self, reward: float, done: bool) -> None:
        pass

    def set_player_id(self, player_id: int) -> None:
        self.player_id = player_id

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.config.name, "type": self.config.agent_type,
                "version": self.config.version, "description": self.config.description}

    def save(self, directory: str) -> str:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(str(path / "config.json"))
        self._save_weights(path)
        return str(path)

    def _save_weights(self, directory: Path) -> None:
        pass

    @classmethod
    def load(cls, directory: str) -> "Agent":
        path = Path(directory)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")
        agent = cls(AgentConfig.load(str(config_path)))
        agent._load_weights(path)
        return agent

    def _load_weights(self, directory: Path) -> None:
        pass


def get_agent_class(agent_type: str) -> type:
    from agents.chase import ChaseAgent, SmartChaseAgent
    from agents.random_agent import RandomAgent
    from agents.positional import PositionalAgent
    classes = {
        "chase": ChaseAgent,
        "smart_chase": SmartChaseAgent,
        "random": RandomAgent,
        "positional": PositionalAgent,
    }
    try:
        from agents.neural import NeuralAgent
        classes["neural"] = NeuralAgent
    except ImportError:
        pass
    if agent_type not in classes:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(classes.keys())}")
    return classes[agent_type]


def load_agent(directory: str) -> Agent:
    config_path = Path(directory) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")
    config = AgentConfig.load(str(config_path))
    return get_agent_class(config.agent_type).load(directory)
