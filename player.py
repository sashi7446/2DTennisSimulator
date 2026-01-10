"""Player logic for 2D Tennis Simulator."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ball import Ball
from config import Config
from field import Field

NUM_DIRECTIONS = 16
DIRECTION_ANGLES = [i * 22.5 for i in range(NUM_DIRECTIONS)]
NUM_MOVEMENT_ACTIONS = NUM_DIRECTIONS + 1


def direction_to_angle(direction: int) -> Optional[float]:
    """Convert direction (0-15) to angle; 16+ returns None (stay)."""
    return DIRECTION_ANGLES[direction] if direction < NUM_DIRECTIONS else None


@dataclass
class Player:
    """Tennis player with movement and hit capabilities."""

    player_id: int
    x: float
    y: float
    speed: float
    radius: float
    reach_distance: float

    def move(self, direction: int, field: Field) -> None:
        angle = direction_to_angle(direction)
        if angle is None:
            return
        rad = math.radians(angle)
        self.x, self.y = field.clamp_position(
            self.x + math.cos(rad) * self.speed, self.y + math.sin(rad) * self.speed, self.radius
        )

    def can_hit(self, ball: Ball) -> bool:
        return ball.is_in and ball.distance_to(self.x, self.y) <= self.reach_distance

    def hit_ball(self, ball: Ball, angle_degrees: float, speed: float) -> bool:
        if not self.can_hit(ball):
            return False
        ball.hit(self.player_id, angle_degrees, speed)
        return True

    def distance_to_ball(self, ball: Ball) -> float:
        return ball.distance_to(self.x, self.y)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def reset(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def create_players(field: Field, config: Config) -> Tuple[Player, Player]:
    pos_a, pos_b = field.get_player_start_positions()

    def make_player(pid: int, pos: Tuple[float, float]) -> Player:
        return Player(
            pid, pos[0], pos[1], config.player_speed, config.player_radius, config.reach_distance
        )

    return (make_player(0, pos_a), make_player(1, pos_b))
