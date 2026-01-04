"""Ball logic for 2D Tennis Simulator."""

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple

from config import Config
from field import Field


@dataclass
class Ball:
    """
    The tennis ball with position, velocity, and in-flag state.

    The in-flag tracks whether the ball has passed through an in-area
    since the last hit. This determines whether a shot is valid (in)
    or invalid (out).
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    in_flag: bool = False
    last_hit_by: Optional[int] = None  # 0 for player A, 1 for player B, None for serve
    radius: float = 5.0

    def update(self, field: Field) -> Optional[str]:
        """
        Update ball position and check for in-area/wall collisions.

        Args:
            field: The field to check collisions against

        Returns:
            None if no wall collision, or the wall name if collision occurred
        """
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Check if ball enters an in-area (sets in_flag to True)
        if not self.in_flag and field.is_in_area(self.x, self.y):
            self.in_flag = True

        # Check for wall collision
        wall = field.check_wall_collision(self.x, self.y, self.radius)
        return wall

    def set_velocity_from_angle(self, angle_degrees: float, speed: float) -> None:
        """
        Set ball velocity from an angle and speed.

        Args:
            angle_degrees: Direction in degrees (0 = right, 90 = down)
            speed: Speed of the ball
        """
        angle_rad = math.radians(angle_degrees)
        self.vx = math.cos(angle_rad) * speed
        self.vy = math.sin(angle_rad) * speed

    def get_angle(self) -> float:
        """Get the current direction of the ball in degrees."""
        return math.degrees(math.atan2(self.vy, self.vx))

    def get_speed(self) -> float:
        """Get the current speed of the ball."""
        return math.sqrt(self.vx**2 + self.vy**2)

    def hit(self, player_id: int, angle_degrees: float, speed: float) -> None:
        """
        Handle a player hitting the ball.

        Args:
            player_id: 0 for player A, 1 for player B
            angle_degrees: Direction to hit the ball
            speed: Speed of the hit
        """
        self.set_velocity_from_angle(angle_degrees, speed)
        self.in_flag = False  # Reset in-flag on hit
        self.last_hit_by = player_id

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance from ball to a point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    @property
    def position(self) -> Tuple[float, float]:
        """Get ball position as tuple."""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get ball velocity as tuple."""
        return (self.vx, self.vy)

    def reset(self, x: float, y: float) -> None:
        """Reset ball to a position with zero velocity."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.in_flag = False
        self.last_hit_by = None


def create_serve_ball(
    field: Field, config: Config, direction: Optional[int] = None
) -> Ball:
    """
    Create a ball for serving from the center.

    Args:
        field: The field
        config: Game configuration
        direction: -1 for left, 1 for right, None for random

    Returns:
        Ball positioned at center with serve velocity
    """
    import random

    center_x, center_y = field.center

    # Random direction if not specified
    if direction is None:
        direction = random.choice([-1, 1])

    # Random angle within serve_angle_range
    angle_offset = random.uniform(
        -config.serve_angle_range, config.serve_angle_range
    )

    # Base angle: 0 for right, 180 for left
    base_angle = 0 if direction > 0 else 180
    serve_angle = base_angle + angle_offset

    ball = Ball(x=center_x, y=center_y, radius=config.ball_radius)
    ball.set_velocity_from_angle(serve_angle, config.ball_speed)

    return ball
