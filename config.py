"""Configuration for 2D Tennis Simulator."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class Config:
    """Configuration parameters for the tennis simulator."""

    # Field dimensions
    field_width: int = 800
    field_height: int = 400

    # In-area dimensions (1:2 aspect ratio)
    area_width: int = 200
    area_height: int = 300
    area_gap: int = 250  # Gap between Area A and Area B

    # Ball properties
    ball_speed: float = 15.0
    ball_radius: int = 5
    serve_angle_range: float = 15.0  # ±15 degrees from horizontal

    # Player properties
    player_speed: float = 4.0
    player_radius: int = 15
    reach_distance: float = 30.0

    # Rewards (sparse reward for 1-point episodes)
    reward_point_win: float = 1.0
    reward_point_lose: float = -1.0
    reward_rally: float = 0.1

    # Game settings
    max_steps_per_point: int = 1000  # Prevent infinite rallies
    # Note: Game engine always ends after 1 point (1 point = 1 episode)
    # Multi-episode tracking is handled by the training loop, not the game engine

    # Display settings
    fps: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def get_area_positions(self) -> tuple:
        """
        Calculate positions of Area A and Area B.

        Returns:
            Tuple of ((area_a_x, area_a_y), (area_b_x, area_b_y))
            where x, y are the top-left corners of each area.
        """
        center_x = self.field_width / 2
        center_y = self.field_height / 2

        # Area A is on the left side
        area_a_x = center_x - self.area_gap / 2 - self.area_width
        area_a_y = center_y - self.area_height / 2

        # Area B is on the right side
        area_b_x = center_x + self.area_gap / 2
        area_b_y = center_y - self.area_height / 2

        return ((area_a_x, area_a_y), (area_b_x, area_b_y))


# Default configuration instance
DEFAULT_CONFIG = Config()
