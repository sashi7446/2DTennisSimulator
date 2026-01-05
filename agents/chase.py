"""Chase Agent - Simple ball-chasing AI."""

import math
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from agents.base import Agent, AgentConfig


class ChaseAgent(Agent):
    """
    Simple rule-based agent that chases the ball.

    Strategy:
    - Always move toward the ball
    - Hit toward the opponent's side
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ChaseBot",
                agent_type="chase",
                version="1.0",
                description="Simple ball-chasing AI that always moves toward the ball",
                parameters={
                    "aggression": 1.0,  # How directly to chase (1.0 = straight line)
                },
            )
        super().__init__(config)

        # Field dimensions (set when agent is used)
        self.field_width: float = 800
        self.field_height: float = 400

    def set_field_dimensions(self, width: float, height: float) -> None:
        """Set field dimensions for hit angle calculation."""
        self.field_width = width
        self.field_height = height

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Move toward ball, hit toward opponent."""
        # Get positions based on player_id
        if self.player_id == 0:
            my_x = observation["player_a_x"]
            my_y = observation["player_a_y"]
        else:
            my_x = observation["player_b_x"]
            my_y = observation["player_b_y"]

        ball_x = observation["ball_x"]
        ball_y = observation["ball_y"]

        # Calculate direction to ball
        dx = ball_x - my_x
        dy = ball_y - my_y

        # Convert to angle
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        # Discretize to 16 directions
        move_direction = int(angle / 22.5) % 16

        # Hit toward opponent's side
        if self.player_id == 0:
            # Player A hits right
            hit_angle = 0.0
        else:
            # Player B hits left
            hit_angle = 180.0

        return (move_direction, hit_angle)

    def get_info(self) -> Dict[str, Any]:
        """Get agent info with additional details."""
        info = super().get_info()
        info["strategy"] = "Chase ball directly, hit toward opponent"
        return info


class SmartChaseAgent(ChaseAgent):
    """
    Improved chase agent with better positioning.

    Improvements over basic ChaseAgent:
    - Returns to home position when ball is far
    - Varies hit angles slightly
    - Considers ball velocity for interception
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="SmartChaseBot",
                agent_type="chase",
                version="2.0",
                description="Improved chase AI with positioning awareness",
                parameters={
                    "home_return_threshold": 200,  # Distance to return home
                    "hit_angle_variance": 15,  # Random variance in hit angle
                },
            )
        super().__init__(config)

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Smart chase with positioning."""
        import random

        # Get positions based on player_id
        if self.player_id == 0:
            my_x = observation["player_a_x"]
            my_y = observation["player_a_y"]
            home_x = self.field_width * 0.2  # Left side home
        else:
            my_x = observation["player_b_x"]
            my_y = observation["player_b_y"]
            home_x = self.field_width * 0.8  # Right side home

        home_y = self.field_height / 2
        ball_x = observation["ball_x"]
        ball_y = observation["ball_y"]
        ball_vx = observation["ball_vx"]
        ball_vy = observation["ball_vy"]

        # Predict ball position (simple prediction)
        predict_frames = 10
        predicted_x = ball_x + ball_vx * predict_frames
        predicted_y = ball_y + ball_vy * predict_frames

        # Clamp to field
        predicted_x = max(0, min(predicted_x, self.field_width))
        predicted_y = max(0, min(predicted_y, self.field_height))

        # Calculate distance to ball
        ball_dist = math.sqrt((ball_x - my_x) ** 2 + (ball_y - my_y) ** 2)

        # Decision: chase ball or return home
        threshold = self.config.parameters.get("home_return_threshold", 200)

        if ball_dist < threshold:
            # Chase the predicted ball position
            target_x = predicted_x
            target_y = predicted_y
        else:
            # Ball is far, move toward home position
            # But bias toward ball's y position
            target_x = home_x
            target_y = (home_y + ball_y) / 2

        # Calculate direction to target
        dx = target_x - my_x
        dy = target_y - my_y

        if abs(dx) < 1 and abs(dy) < 1:
            move_direction = 16  # Stay
        else:
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            move_direction = int(angle / 22.5) % 16

        # Hit with some variance
        variance = self.config.parameters.get("hit_angle_variance", 15)
        if self.player_id == 0:
            base_angle = 0.0
        else:
            base_angle = 180.0

        hit_angle = base_angle + random.uniform(-variance, variance)

        return (move_direction, hit_angle)

    def get_info(self) -> Dict[str, Any]:
        """Get agent info with additional details."""
        info = super().get_info()
        info["strategy"] = "Smart positioning with ball prediction"
        return info
