"""Random Agent - Baseline random AI."""

import random
from typing import Any, Dict, Optional, Tuple

from agents.base import Agent, AgentConfig


class RandomAgent(Agent):
    """
    Random agent that makes random decisions.

    Useful as a baseline for comparison.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="RandomBot",
                agent_type="random",
                version="1.0",
                description="Completely random agent for baseline comparison",
                parameters={},
            )
        super().__init__(config)

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Make random decisions."""
        move_direction = random.randint(0, 16)  # 0-15 directions + 16 stay
        hit_angle = random.uniform(0, 360)
        return (move_direction, hit_angle)

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        info = super().get_info()
        info["strategy"] = "Completely random actions"
        return info
