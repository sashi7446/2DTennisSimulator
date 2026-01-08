"""Baseliner Agent - Defensive baseline player."""

import math
from typing import Dict, Any, Tuple, Optional

from agents.base import Agent, AgentConfig


def _get_my_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    """Get player's own position from observation."""
    prefix = "player_a" if player_id == 0 else "player_b"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]


def _angle_to_direction(dx: float, dy: float) -> int:
    """Convert delta to discrete 16-direction (or 16 for stay)."""
    if abs(dx) < 1 and abs(dy) < 1:
        return 16  # Stay
    angle = math.degrees(math.atan2(dy, dx))
    return int((angle % 360) / 22.5) % 16


class BaselinerAgent(Agent):
    """
    Defensive baseliner agent.

    Stays at the back of the court and returns to home position after hitting.
    Prioritizes consistency over aggression.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig(
            name="Baseliner",
            agent_type="baseliner",
            version="1.0",
            description="Defensive baseliner - stays back and returns to home position",
            parameters={
                "home_x_ratio": 0.1,  # How far back (0.1 = 10% from baseline)
            }
        ))
        self.field_width = 800.0
        self.field_height = 400.0

    def set_field_dimensions(self, width: float, height: float) -> None:
        self.field_width = width
        self.field_height = height

    def _get_home_position(self) -> Tuple[float, float]:
        """Get the home position for this player."""
        home_x_ratio = self.config.parameters.get("home_x_ratio", 0.1)
        if self.player_id == 0:
            # Player A: left side, stay near baseline
            home_x = self.field_width * home_x_ratio
        else:
            # Player B: right side, stay near baseline
            home_x = self.field_width * (1.0 - home_x_ratio)
        home_y = self.field_height / 2  # Center vertically
        return home_x, home_y

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        my_x, my_y = _get_my_pos(observation, self.player_id)
        ball_y = observation["ball_y"]

        home_x, _ = self._get_home_position()

        # Only move in X to return to home position, Y to match ball
        dx = home_x - my_x  # Always try to stay at home X position
        dy = ball_y - my_y  # Match ball's Y position

        move_dir = _angle_to_direction(dx, dy)

        # Hit angle: just hit it back to opponent's side
        hit_angle = 0.0 if self.player_id == 0 else 180.0

        return (move_dir, hit_angle)
