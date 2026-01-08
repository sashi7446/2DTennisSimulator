"""Positional Agent - Tactical positioning and shot placement AI."""

import math
import random
from typing import Dict, Any, Tuple, Optional

from agents.base import Agent, AgentConfig


def _get_my_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    """Get player's own position from observation."""
    prefix = "player_a" if player_id == 0 else "player_b"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]


def _get_opponent_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    """Get opponent's position from observation."""
    prefix = "player_b" if player_id == 0 else "player_a"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]


def _angle_to_direction(dx: float, dy: float) -> int:
    """Convert delta to discrete 16-direction (or 16 for stay)."""
    if abs(dx) < 1 and abs(dy) < 1:
        return 16
    angle = math.degrees(math.atan2(dy, dx))
    return int((angle % 360) / 22.5) % 16


def _normalize_angle(angle: float) -> float:
    """Normalize angle to 0-360 range."""
    return angle % 360


class PositionalAgent(Agent):
    """
    Tactical agent that focuses on:
    1. Strategic court positioning (defensive zones)
    2. Shot placement based on opponent position
    3. Court coverage optimization
    4. Adaptive defensive/offensive stance
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig(
            name="PositionalBot",
            agent_type="positional",
            version="1.0",
            description="Tactical positioning AI with adaptive shot placement",
            parameters={
                "defensive_depth": 0.25,      # How far back to position (0.0 = net, 0.5 = center)
                "court_coverage": 0.7,        # How aggressively to cover court width (0.0-1.0)
                "prediction_time": 15,        # Frames to predict ball trajectory
                "shot_variance": 20,          # Angle variance for shot placement (degrees)
                "aggressive_threshold": 0.6,  # Distance ratio to switch to aggressive shots
                "center_bias": 0.3,           # Bias towards center positioning (0.0-1.0)
            }
        ))
        self.field_width = 800.0
        self.field_height = 400.0

    def set_field_dimensions(self, width: float, height: float) -> None:
        self.field_width = width
        self.field_height = height

    def _calculate_defensive_position(
        self,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
    ) -> Tuple[float, float]:
        """Calculate optimal defensive position based on ball state."""
        params = self.config.parameters

        # Base defensive line position
        if self.player_id == 0:
            base_x = self.field_width * params.get("defensive_depth", 0.25)
        else:
            base_x = self.field_width * (1.0 - params.get("defensive_depth", 0.25))

        # Predict ball future position
        pred_time = params.get("prediction_time", 15)
        pred_x = ball_x + ball_vx * pred_time
        pred_y = ball_y + ball_vy * pred_time

        # Clamp prediction to field bounds
        pred_x = max(0, min(pred_x, self.field_width))
        pred_y = max(0, min(pred_y, self.field_height))

        # Calculate target Y based on predicted ball position and court coverage
        coverage = params.get("court_coverage", 0.7)
        center_bias = params.get("center_bias", 0.3)

        center_y = self.field_height / 2
        target_y = center_y + (pred_y - center_y) * coverage

        # Apply center bias
        target_y = target_y * (1 - center_bias) + center_y * center_bias

        # Ensure we stay within reasonable bounds
        margin = self.field_height * 0.1
        target_y = max(margin, min(target_y, self.field_height - margin))

        return base_x, target_y

    def _calculate_shot_angle(
        self,
        my_x: float,
        my_y: float,
        opp_x: float,
        opp_y: float,
        ball_x: float,
        ball_y: float,
    ) -> float:
        """Calculate optimal shot angle to exploit opponent's position."""
        params = self.config.parameters

        # Determine if we're in aggressive range
        my_field_pos = my_x / self.field_width
        aggressive_threshold = params.get("aggressive_threshold", 0.6)

        # Base angle towards opponent's side
        base_angle = 0.0 if self.player_id == 0 else 180.0

        # Calculate opponent's position relative to center
        opp_y_offset = opp_y - (self.field_height / 2)

        # Determine shot strategy
        is_aggressive = (
            (self.player_id == 0 and my_field_pos < aggressive_threshold) or
            (self.player_id == 1 and my_field_pos > (1 - aggressive_threshold))
        )

        if is_aggressive:
            # Aggressive: aim away from opponent
            if abs(opp_y_offset) > self.field_height * 0.15:
                # Opponent is off-center, aim to opposite side
                target_angle_offset = -30 if opp_y_offset > 0 else 30
            else:
                # Opponent is centered, aim to corners
                target_angle_offset = random.choice([-35, 35])
        else:
            # Defensive: aim deep with slight angle
            target_angle_offset = random.uniform(-15, 15)

        # Add variance for unpredictability
        variance = params.get("shot_variance", 20)
        angle_variation = random.uniform(-variance, variance)

        final_angle = base_angle + target_angle_offset + angle_variation
        return _normalize_angle(final_angle)

    def _should_intercept_ball(
        self,
        my_x: float,
        my_y: float,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
    ) -> bool:
        """Determine if ball is coming towards our side and needs interception."""
        # Check if ball is moving towards our side
        if self.player_id == 0:
            return ball_vx < 0 and ball_x < self.field_width * 0.5
        else:
            return ball_vx > 0 and ball_x > self.field_width * 0.5

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Execute tactical positioning and shot selection."""
        # Extract observation data
        my_x, my_y = _get_my_pos(observation, self.player_id)
        opp_x, opp_y = _get_opponent_pos(observation, self.player_id)

        ball_x = observation["ball_x"]
        ball_y = observation["ball_y"]
        ball_vx = observation["ball_vx"]
        ball_vy = observation["ball_vy"]
        ball_in_flag = observation.get("ball_in_flag", True)

        # Calculate optimal defensive position
        target_x, target_y = self._calculate_defensive_position(
            ball_x, ball_y, ball_vx, ball_vy
        )

        # If ball needs interception, adjust target to ball's predicted position
        if ball_in_flag and self._should_intercept_ball(
            my_x, my_y, ball_x, ball_y, ball_vx, ball_vy
        ):
            pred_time = self.config.parameters.get("prediction_time", 15)
            intercept_x = ball_x + ball_vx * pred_time
            intercept_y = ball_y + ball_vy * pred_time

            # Clamp to field bounds
            intercept_x = max(0, min(intercept_x, self.field_width))
            intercept_y = max(0, min(intercept_y, self.field_height))

            # Blend defensive position with intercept position
            dist_to_ball = math.hypot(ball_x - my_x, ball_y - my_y)
            intercept_urgency = max(0, min(1, (200 - dist_to_ball) / 200))

            target_x = target_x * (1 - intercept_urgency) + intercept_x * intercept_urgency
            target_y = target_y * (1 - intercept_urgency) + intercept_y * intercept_urgency

        # Calculate movement direction
        dx = target_x - my_x
        dy = target_y - my_y
        move_direction = _angle_to_direction(dx, dy)

        # Calculate shot angle
        shot_angle = self._calculate_shot_angle(
            my_x, my_y, opp_x, opp_y, ball_x, ball_y
        )

        return move_direction, shot_angle
