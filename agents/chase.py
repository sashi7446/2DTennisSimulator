"""Chase Agent - Simple ball-chasing AI."""

import math
from typing import Any, Dict, Optional, Tuple

from agents.base import Agent, AgentConfig


def _get_my_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    """Get player's own position from observation."""
    prefix = "player_a" if player_id == 0 else "player_b"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]


def _angle_to_direction(dx: float, dy: float) -> int:
    """Convert delta to discrete 16-direction (or 16 for stay)."""
    if abs(dx) < 1 and abs(dy) < 1:
        return 16
    angle = math.degrees(math.atan2(dy, dx))
    return int((angle % 360) / 22.5) % 16


class ChaseAgent(Agent):
    """Simple agent that chases the ball and hits toward opponent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="ChaseBot",
                agent_type="chase",
                version="1.0",
                description="Ball-chasing AI",
                parameters={"aggression": 1.0},
            )
        )
        self.field_width, self.field_height = 800.0, 400.0

    def set_field_dimensions(self, width: float, height: float) -> None:
        self.field_width, self.field_height = width, height

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        my_x, my_y = _get_my_pos(observation, self.player_id)
        dx = observation["ball_x"] - my_x
        dy = observation["ball_y"] - my_y
        return (_angle_to_direction(dx, dy), 0.0 if self.player_id == 0 else 180.0)


class SmartChaseAgent(ChaseAgent):
    """Chase agent with positioning awareness and ball prediction."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="SmartChaseBot",
                agent_type="chase",
                version="2.0",
                description="Chase AI with positioning",
                parameters={"home_return_threshold": 200, "hit_angle_variance": 15},
            )
        )

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        import random

        my_x, my_y = _get_my_pos(observation, self.player_id)
        home_x = self.field_width * (0.2 if self.player_id == 0 else 0.8)
        home_y = self.field_height / 2

        ball_x, ball_y = observation["ball_x"], observation["ball_y"]
        ball_vx, ball_vy = observation["ball_vx"], observation["ball_vy"]

        # Predict ball position
        pred_x = max(0, min(ball_x + ball_vx * 10, self.field_width))
        pred_y = max(0, min(ball_y + ball_vy * 10, self.field_height))

        ball_dist = math.hypot(ball_x - my_x, ball_y - my_y)
        threshold = self.config.parameters.get("home_return_threshold", 200)

        if ball_dist < threshold:
            target_x, target_y = pred_x, pred_y
        else:
            target_x, target_y = home_x, (home_y + ball_y) / 2

        move_dir = _angle_to_direction(target_x - my_x, target_y - my_y)
        base_angle = 0.0 if self.player_id == 0 else 180.0
        variance = self.config.parameters.get("hit_angle_variance", 15)

        return (move_dir, base_angle + random.uniform(-variance, variance))
