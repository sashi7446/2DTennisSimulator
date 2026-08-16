"""Solver Agent - searches the hit angle that the opponent cannot reach."""

import math
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentConfig
from agents.intercept import InterceptAgent, _to_direction

DEFAULT_BALL_SPEED = 15.0

# Finer than a player's reach at the far side of the court, so no smaller
# step can find a shot this one misses.
ANGLE_STEP = 2.0

# The first wall exits the loop long before this; it only bounds a ball that
# somehow never reaches one.
SHOT_LOOKAHEAD_FRAMES = 200

# The search only pays off within a hit of the ball, and it runs every frame.
HIT_SEARCH_SLACK = 6.0

WINNER_SCORE = float("inf")


class SolverAgent(InterceptAgent):
    """Chooses where to hit by simulating every candidate shot.

    InterceptAgent decides how to reach the ball and then hits it straight
    back, which keeps a rally alive but never ends one. The reaching logic is
    inherited unchanged; only the hit angle is searched.

    The opponent is assumed to run straight at the ball at full speed from
    the moment of the hit, so a shot judged unreachable here is unreachable
    for any opponent, not just the ones in this repo.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="Solver",
                agent_type="solver",
                version="1.0",
                description="One-ply shot search against the opponent's reachable area",
                parameters={"reach_margin": 0.9, "angle_step": ANGLE_STEP},
            )
        )
        self.ball_speed = DEFAULT_BALL_SPEED

    def set_ball_speed(self, speed: float) -> None:
        """Correct the shot speed when the config is not the default."""
        self.ball_speed = float(speed)

    def _opponent_area(self) -> Tuple[float, float]:
        """Top-left of the in-area a shot from this player must pass through."""
        cx, cy = self.field_width / 2, self.field_height / 2
        y = cy - self.area_height / 2
        if self.player_id == 0:
            return (cx + self.area_gap / 2, y)
        return (cx - self.area_gap / 2 - self.area_width, y)

    def _candidate_angles(self) -> List[float]:
        step = float(self.config.parameters.get("angle_step", ANGLE_STEP))
        count = max(1, int(round(360.0 / step)))
        return [i * step for i in range(count)]

    def _score_shot(
        self, angle: float, bx: float, by: float, ox: float, oy: float
    ) -> Optional[float]:
        """Slack the shot leaves the opponent, or None if it lands out.

        Lower is better, and WINNER_SCORE means they never come within range.
        """
        rad = math.radians(angle)
        vx, vy = math.cos(rad) * self.ball_speed, math.sin(rad) * self.ball_speed
        ax, ay = self._opponent_area()
        r = self.ball_radius
        x, y = bx, by
        is_in = False
        best_slack = WINNER_SCORE

        for step in range(1, SHOT_LOOKAHEAD_FRAMES + 1):
            x += vx
            y += vy

            if not is_in and ax <= x <= ax + self.area_width:
                if ay <= y <= ay + self.area_height:
                    is_in = True

            if x - r <= 0 or x + r >= self.field_width or y - r <= 0 or y + r >= self.field_height:
                # The wall ends the point either way: for a valid ball in my
                # favour, otherwise as my own shot going out
                return best_slack if is_in else None

            if not is_in:
                continue

            # One frame of travel more than elapsed, because the opponent
            # also moves in the frame the hit itself happens
            budget = self.player_speed * (step + 1) + self.reach_distance
            slack = budget - math.hypot(x - ox, y - oy)
            if slack >= 0:
                best_slack = min(best_slack, slack)

        return best_slack if is_in else None

    def _best_hit_angle(self, obs: Dict[str, Any], mx: float, my: float) -> Optional[float]:
        """Search the candidate angles, or None when no shot is imminent."""
        bx, by = obs["ball_x"], obs["ball_y"]
        if math.hypot(bx - mx, by - my) > self.reach_distance + HIT_SEARCH_SLACK:
            return None

        opp_prefix = "player_b" if self.player_id == 0 else "player_a"
        ox, oy = obs[f"{opp_prefix}_x"], obs[f"{opp_prefix}_y"]

        best_angle, best_score = None, None
        for angle in self._candidate_angles():
            score = self._score_shot(angle, bx, by, ox, oy)
            if score is None:
                continue
            if score == WINNER_SCORE:
                return angle
            if best_score is None or score < best_score:
                best_angle, best_score = angle, score
        return best_angle

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        prefix = "player_a" if self.player_id == 0 else "player_b"
        mx, my = observation[f"{prefix}_x"], observation[f"{prefix}_y"]

        target = self._interception(mx, my, observation) or self._defensive_home()

        # As in InterceptAgent: the ball is only returnable on this player's
        # own half, so crossing the centre gains nothing
        centre = self.field_width / 2
        tx = min(target[0], centre) if self.player_id == 0 else max(target[0], centre)
        movement = _to_direction(tx - mx, target[1] - my)

        angle = self._best_hit_angle(observation, mx, my)
        if angle is None:
            angle = 0.0 if self.player_id == 0 else 180.0
        return (movement, angle)
