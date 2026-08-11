"""Intercept Agent - moves to where the ball will be, not where it is."""

import math
import random
from typing import Any, Dict, Optional, Tuple

from agents.base import Agent, AgentConfig

# Court geometry, mirrored from Config. Agents must not import Config
# (ARCHITECTURE.md), so these are defaults that set_field_dimensions and
# set_physics can correct.
DEFAULT_FIELD = (800.0, 400.0)
DEFAULT_AREA = (200.0, 300.0, 250.0)  # width, height, gap
DEFAULT_PLAYER_SPEED = 4.0
DEFAULT_REACH = 30.0
DEFAULT_BALL_RADIUS = 5.0

LOOKAHEAD_FRAMES = 300
HIT_ANGLE_VARIANCE = 15.0


def _to_direction(dx: float, dy: float) -> int:
    """Convert a delta into one of 16 headings, or 16 to stand still."""
    if abs(dx) < 1 and abs(dy) < 1:
        return 16
    return int((math.degrees(math.atan2(dy, dx)) % 360) / 22.5) % 16


class InterceptAgent(Agent):
    """Predicts where the ball becomes returnable and gets there in time.

    ChaseAgent walks towards the ball's current position, which loses balls
    that were reachable had the player started towards the interception
    point instead. This agent projects the ball forward and picks the
    earliest point it can actually arrive at.

    Between rallies it recovers to deep centre on its own half, the most
    defensive spot in tennis, and never crosses the net - the ball is only
    returnable inside its own in-area, so there is nothing to chase over
    there.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="Interceptor",
                agent_type="intercept",
                version="1.0",
                description="Predictive interception with defensive recovery",
                parameters={"reach_margin": 0.9},
            )
        )
        self.field_width, self.field_height = DEFAULT_FIELD
        self.area_width, self.area_height, self.area_gap = DEFAULT_AREA
        self.player_speed = DEFAULT_PLAYER_SPEED
        self.reach_distance = DEFAULT_REACH
        self.ball_radius = DEFAULT_BALL_RADIUS

    def set_field_dimensions(self, width: float, height: float) -> None:
        self.field_width, self.field_height = float(width), float(height)

    def set_physics(
        self,
        player_speed: float,
        reach_distance: float,
        area: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Correct the movement assumptions when the config is not the default."""
        self.player_speed = float(player_speed)
        self.reach_distance = float(reach_distance)
        if area is not None:
            self.area_width, self.area_height, self.area_gap = (float(v) for v in area)

    def _my_area(self) -> Tuple[float, float]:
        """Top-left of the in-area on this player's side."""
        cx, cy = self.field_width / 2, self.field_height / 2
        y = cy - self.area_height / 2
        if self.player_id == 0:
            return (cx - self.area_gap / 2 - self.area_width, y)
        return (cx + self.area_gap / 2, y)

    def _defensive_home(self) -> Tuple[float, float]:
        """Deep centre behind the in-area - furthest point still covering both wings."""
        ax, _ = self._my_area()
        if self.player_id == 0:
            return (ax / 2, self.field_height / 2)
        area_right = ax + self.area_width
        return (area_right + (self.field_width - area_right) / 2, self.field_height / 2)

    def _interception(
        self, mx: float, my: float, obs: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        """Earliest projected ball position this player can reach in time."""
        x, y = obs["ball_x"], obs["ball_y"]
        vx, vy = obs["ball_vx"], obs["ball_vy"]
        returnable = bool(obs["ball_is_in"])

        ax, ay = self._my_area()
        margin = self.config.parameters.get("reach_margin", 0.9)
        r = self.ball_radius

        for step in range(1, LOOKAHEAD_FRAMES + 1):
            x += vx
            y += vy
            # Any wall ends the point, so there is nothing to chase past one
            if x - r <= 0 or x + r >= self.field_width:
                return None
            if y - r <= 0 or y + r >= self.field_height:
                return None
            if not returnable and ax <= x <= ax + self.area_width:
                if ay <= y <= ay + self.area_height:
                    returnable = True
            if not returnable:
                continue
            budget = self.player_speed * step + self.reach_distance * margin
            if math.hypot(x - mx, y - my) <= budget:
                return (x, y)
        return None

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        prefix = "player_a" if self.player_id == 0 else "player_b"
        mx, my = observation[f"{prefix}_x"], observation[f"{prefix}_y"]

        target = self._interception(mx, my, observation) or self._defensive_home()

        # Staying on your own half: crossing gains nothing, since the ball
        # only becomes returnable inside this player's in-area
        centre = self.field_width / 2
        tx = min(target[0], centre) if self.player_id == 0 else max(target[0], centre)

        base_angle = 0.0 if self.player_id == 0 else 180.0
        variance = random.uniform(-HIT_ANGLE_VARIANCE, HIT_ANGLE_VARIANCE)
        return (_to_direction(tx - mx, target[1] - my), base_angle + variance)
