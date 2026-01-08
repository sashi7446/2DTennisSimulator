"""Configuration for 2D Tennis Simulator."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Config:
    """Configuration parameters for the tennis simulator."""

    # Field dimensions
    field_width: int = 800
    field_height: int = 400

    # In-area dimensions (3:4 aspect ratio)
    area_width: int = 150
    area_height: int = 200
    area_gap: int = 150  # Gap between Area A and Area B

    # Ball properties
    ball_speed: float = 5.0
    ball_radius: int = 5
    serve_angle_range: float = 15.0  # ±15 degrees from horizontal

    # Player properties
    player_speed: float = 3.0
    player_radius: int = 15
    reach_distance: float = 60.0

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
        return {
            "field_width": self.field_width,
            "field_height": self.field_height,
            "area_width": self.area_width,
            "area_height": self.area_height,
            "area_gap": self.area_gap,
            "ball_speed": self.ball_speed,
            "ball_radius": self.ball_radius,
            "serve_angle_range": self.serve_angle_range,
            "player_speed": self.player_speed,
            "player_radius": self.player_radius,
            "reach_distance": self.reach_distance,
            "reward_point_win": self.reward_point_win,
            "reward_point_lose": self.reward_point_lose,
            "reward_rally": self.reward_rally,
            "max_steps_per_point": self.max_steps_per_point,
            "fps": self.fps,
        }

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
