"""Field and in-area logic for 2D Tennis Simulator."""

from dataclasses import dataclass
from typing import Optional, Tuple

from config import Config


@dataclass
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class Field:
    """Tennis field with walls and in-areas (A left, B right)."""

    def __init__(self, config: Config):
        self.config = config
        self.bounds = Rectangle(0, 0, config.field_width, config.field_height)
        area_a_pos, area_b_pos = config.get_area_positions()
        self.area_a = Rectangle(*area_a_pos, config.area_width, config.area_height)
        self.area_b = Rectangle(*area_b_pos, config.area_width, config.area_height)

    @property
    def center(self) -> Tuple[float, float]:
        return self.bounds.center

    def is_in_area(self, x: float, y: float) -> bool:
        return self.area_a.contains_point(x, y) or self.area_b.contains_point(x, y)

    def is_in_area_a(self, x: float, y: float) -> bool:
        return self.area_a.contains_point(x, y)

    def is_in_area_b(self, x: float, y: float) -> bool:
        return self.area_b.contains_point(x, y)

    def check_wall_collision(self, x: float, y: float, radius: float = 0) -> Optional[str]:
        """Return wall name ('left', 'right', 'top', 'bottom') or None."""
        if x - radius <= 0:
            return "left"
        if x + radius >= self.config.field_width:
            return "right"
        if y - radius <= 0:
            return "top"
        if y + radius >= self.config.field_height:
            return "bottom"
        return None

    def clamp_position(self, x: float, y: float, radius: float = 0) -> Tuple[float, float]:
        return (
            max(radius, min(x, self.config.field_width - radius)),
            max(radius, min(y, self.config.field_height - radius)),
        )

    def get_player_start_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns ((player_a_x, y), (player_b_x, y))."""
        cy = self.config.field_height / 2
        area_b_right = self.area_b.x + self.area_b.width
        return (
            (self.area_a.x / 2, cy),
            (area_b_right + (self.config.field_width - area_b_right) / 2, cy),
        )
