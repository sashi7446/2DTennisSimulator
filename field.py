"""Field and in-area logic for 2D Tennis Simulator."""

from dataclasses import dataclass
from typing import Tuple, Optional

from config import Config


@dataclass
class Rectangle:
    """A rectangle defined by top-left corner and dimensions."""

    x: float
    y: float
    width: float
    height: float

    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is inside the rectangle."""
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center of the rectangle."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounds as (left, top, right, bottom)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class Field:
    """
    The tennis field with walls and in-areas.

    The field is a rectangular space surrounded by walls.
    Inside are two in-areas (Area A on the left, Area B on the right)
    separated by a gap.
    """

    def __init__(self, config: Config):
        self.config = config

        # Field boundaries (the walls)
        self.bounds = Rectangle(0, 0, config.field_width, config.field_height)

        # Calculate in-area positions
        (area_a_pos, area_b_pos) = config.get_area_positions()

        # Create in-areas
        self.area_a = Rectangle(
            area_a_pos[0], area_a_pos[1], config.area_width, config.area_height
        )
        self.area_b = Rectangle(
            area_b_pos[0], area_b_pos[1], config.area_width, config.area_height
        )

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center of the field."""
        return self.bounds.center

    def is_in_area(self, x: float, y: float) -> bool:
        """Check if a point is inside either in-area."""
        return self.area_a.contains_point(x, y) or self.area_b.contains_point(x, y)

    def is_in_area_a(self, x: float, y: float) -> bool:
        """Check if a point is inside Area A."""
        return self.area_a.contains_point(x, y)

    def is_in_area_b(self, x: float, y: float) -> bool:
        """Check if a point is inside Area B."""
        return self.area_b.contains_point(x, y)

    def check_wall_collision(
        self, x: float, y: float, radius: float = 0
    ) -> Optional[str]:
        """
        Check if a point (with optional radius) has hit any wall.

        Returns:
            None if no collision, or a string indicating which wall:
            'left', 'right', 'top', 'bottom'
        """
        if x - radius <= 0:
            return "left"
        if x + radius >= self.config.field_width:
            return "right"
        if y - radius <= 0:
            return "top"
        if y + radius >= self.config.field_height:
            return "bottom"
        return None

    def clamp_position(
        self, x: float, y: float, radius: float = 0
    ) -> Tuple[float, float]:
        """
        Clamp a position to stay within the field walls.

        Args:
            x, y: Position to clamp
            radius: Optional radius to account for object size

        Returns:
            Clamped (x, y) position
        """
        clamped_x = max(radius, min(x, self.config.field_width - radius))
        clamped_y = max(radius, min(y, self.config.field_height - radius))
        return (clamped_x, clamped_y)

    def get_player_start_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get starting positions for both players.

        Player A starts left of Area A, Player B starts right of Area B.

        Returns:
            ((player_a_x, player_a_y), (player_b_x, player_b_y))
        """
        center_y = self.config.field_height / 2

        # Player A starts to the left of Area A
        player_a_x = self.area_a.x / 2
        player_a_y = center_y

        # Player B starts to the right of Area B
        player_b_x = self.area_b.x + self.area_b.width + (
            self.config.field_width - (self.area_b.x + self.area_b.width)
        ) / 2
        player_b_y = center_y

        return ((player_a_x, player_a_y), (player_b_x, player_b_y))
