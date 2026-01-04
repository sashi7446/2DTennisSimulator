"""Player logic for 2D Tennis Simulator."""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

from config import Config
from field import Field
from ball import Ball


# 16 discrete movement directions (22.5 degrees apart)
NUM_DIRECTIONS = 16
DIRECTION_ANGLES = [i * (360 / NUM_DIRECTIONS) for i in range(NUM_DIRECTIONS)]
# Add a "stay" action (no movement)
NUM_MOVEMENT_ACTIONS = NUM_DIRECTIONS + 1  # 16 directions + 1 stay


def direction_to_angle(direction: int) -> Optional[float]:
    """
    Convert a discrete direction to an angle in degrees.

    Args:
        direction: 0-15 for movement directions, 16 for stay

    Returns:
        Angle in degrees (0 = right, 90 = down), or None for stay
    """
    if direction >= NUM_DIRECTIONS:
        return None  # Stay in place
    return DIRECTION_ANGLES[direction]


@dataclass
class Player:
    """
    A tennis player with position and movement capabilities.

    Players can move in 16 discrete directions and hit the ball
    when it's within reach and has a valid in-flag.
    """

    player_id: int  # 0 for player A, 1 for player B
    x: float
    y: float
    speed: float
    radius: float
    reach_distance: float

    def move(self, direction: int, field: Field) -> None:
        """
        Move the player in a discrete direction.

        Args:
            direction: 0-15 for directions, 16 for stay
            field: The field (for boundary clamping)
        """
        angle = direction_to_angle(direction)
        if angle is None:
            return  # Stay in place

        angle_rad = math.radians(angle)
        dx = math.cos(angle_rad) * self.speed
        dy = math.sin(angle_rad) * self.speed

        new_x = self.x + dx
        new_y = self.y + dy

        # Clamp to field boundaries
        self.x, self.y = field.clamp_position(new_x, new_y, self.radius)

    def can_hit(self, ball: Ball) -> bool:
        """
        Check if the player can hit the ball.

        The player can hit if:
        1. Ball is within reach distance
        2. Ball's in-flag is ON

        Args:
            ball: The ball to check

        Returns:
            True if player can hit the ball
        """
        if not ball.in_flag:
            return False

        distance = ball.distance_to(self.x, self.y)
        return distance <= self.reach_distance

    def hit_ball(self, ball: Ball, angle_degrees: float, speed: float) -> bool:
        """
        Attempt to hit the ball.

        Args:
            ball: The ball to hit
            angle_degrees: Direction to hit the ball
            speed: Speed of the hit

        Returns:
            True if hit was successful, False otherwise
        """
        if not self.can_hit(ball):
            return False

        ball.hit(self.player_id, angle_degrees, speed)
        return True

    def distance_to_ball(self, ball: Ball) -> float:
        """Calculate distance to the ball."""
        return ball.distance_to(self.x, self.y)

    @property
    def position(self) -> Tuple[float, float]:
        """Get player position as tuple."""
        return (self.x, self.y)

    def reset(self, x: float, y: float) -> None:
        """Reset player to a position."""
        self.x = x
        self.y = y


def create_players(field: Field, config: Config) -> Tuple[Player, Player]:
    """
    Create both players at their starting positions.

    Args:
        field: The field
        config: Game configuration

    Returns:
        Tuple of (player_a, player_b)
    """
    (pos_a, pos_b) = field.get_player_start_positions()

    player_a = Player(
        player_id=0,
        x=pos_a[0],
        y=pos_a[1],
        speed=config.player_speed,
        radius=config.player_radius,
        reach_distance=config.reach_distance,
    )

    player_b = Player(
        player_id=1,
        x=pos_b[0],
        y=pos_b[1],
        speed=config.player_speed,
        radius=config.player_radius,
        reach_distance=config.reach_distance,
    )

    return (player_a, player_b)
