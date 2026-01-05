"""Game logic for 2D Tennis Simulator."""

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Tuple, Optional, Dict, Any, List

from config import Config
from field import Field
from ball import Ball, create_serve_ball
from player import Player, create_players, NUM_MOVEMENT_ACTIONS


def clamp_hit_angle(angle: float, player_id: int):
    """
    Smoothly map hit angle (0-360°) to a ±45° range.

    Mapping:
    - 0° -> +45°
    - 180° -> 0°
    - 360° -> -45°

    Args:
        angle: Raw hit angle in degrees
        player_id: 0 for player A, 1 for player B

    Returns:
        Mapped angle in degrees
    """
    import math
    # Map 0-360 to +45 to -45 using stretched cosine
    offset = 45.0 * math.cos(math.radians(angle / 2.0))

    if player_id == 0:  # Player A faces right (0°)
        return offset
    else:               # Player B faces left (180°)
        return 180.0 + offset


class GameState(Enum):
    """Current state of the game."""

    SERVING = "serving"  # Ball about to be served
    PLAYING = "playing"  # Ball in play
    POINT_OVER = "point_over"  # Point just ended
    GAME_OVER = "game_over"  # Game finished


@dataclass
class PointResult:
    """Result of a completed point."""

    winner: int  # 0 for player A, 1 for player B
    reason: str  # 'in' (valid shot), 'out' (invalid shot)
    last_hit_by: Optional[int]  # Who hit the ball last


@dataclass
class StepResult:
    """Result of a single game step."""

    rewards: Tuple[float, float]  # Rewards for (player_a, player_b)
    done: bool  # Whether the game is over
    point_result: Optional[PointResult]  # Result if point ended
    hit_occurred: Tuple[bool, bool]  # Whether each player hit this step


class Game:
    """
    Main game class that manages the tennis simulation.

    Handles:
    - Game state management
    - Player actions (movement and hitting)
    - Point scoring and rewards
    - Serve logic
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.field = Field(self.config)
        self.player_a, self.player_b = create_players(self.field, self.config)
        self.ball: Optional[Ball] = None

        # Game state
        self.state = GameState.SERVING
        self.scores = [0, 0]  # [player_a_score, player_b_score]
        self.steps_this_point = 0
        self.total_steps = 0
        self.rally_count = 0  # Number of hits in current rally

        # Start the first point
        self._serve()

    def _serve(self) -> None:
        """Start a new point with a serve from center."""
        self.ball = create_serve_ball(self.field, self.config)
        self.state = GameState.PLAYING
        self.steps_this_point = 0
        self.rally_count = 0

        # Reset players to starting positions
        (pos_a, pos_b) = self.field.get_player_start_positions()
        self.player_a.reset(pos_a[0], pos_a[1])
        self.player_b.reset(pos_b[0], pos_b[1])

    def _end_point(self, winner: int, reason: str) -> PointResult:
        """
        End the current point.

        Args:
            winner: 0 for player A, 1 for player B
            reason: 'in' or 'out'

        Returns:
            PointResult with details
        """
        self.scores[winner] += 1
        result = PointResult(
            winner=winner,
            reason=reason,
            last_hit_by=self.ball.last_hit_by if self.ball else None,
        )

        # Game always ends after 1 point (1 point = 1 episode)
        self.state = GameState.GAME_OVER

        return result

    def step(
        self,
        action_a: Tuple[int, float],
        action_b: Tuple[int, float],
    ) -> StepResult:
        """
        Advance the game by one step.

        Args:
            action_a: (movement_direction, hit_angle) for player A
                     movement_direction: 0-16 (16 = stay)
                     hit_angle: angle in degrees for hitting
            action_b: (movement_direction, hit_angle) for player B

        Returns:
            StepResult with rewards and game state
        """
        if self.state == GameState.GAME_OVER:
            return StepResult(
                rewards=(0.0, 0.0),
                done=True,
                point_result=None,
                hit_occurred=(False, False),
            )

        if self.state == GameState.POINT_OVER:
            # Start a new point
            self._serve()

        self.steps_this_point += 1
        self.total_steps += 1

        # Unpack actions
        move_a, hit_angle_a = action_a
        move_b, hit_angle_b = action_b

        # Move players
        self.player_a.move(move_a, self.field)
        self.player_b.move(move_b, self.field)

        # Check for hits (automatic when in range with valid in-flag)
        hit_a = False
        hit_b = False
        rewards = [0.0, 0.0]

        if self.ball:
            # Player A tries to hit
            if self.player_a.can_hit(self.ball):
                # Clamp angle to front 90° (±45° from facing direction)
                clamped_angle_a = clamp_hit_angle(hit_angle_a, 0)
                self.player_a.hit_ball(
                    self.ball, clamped_angle_a, self.config.ball_speed
                )
                hit_a = True
                self.rally_count += 1
                rewards[0] += self.config.reward_rally

            # Player B tries to hit
            elif self.player_b.can_hit(self.ball):
                # Clamp angle to front 90° (±45° from facing direction)
                clamped_angle_b = clamp_hit_angle(hit_angle_b, 1)
                self.player_b.hit_ball(
                    self.ball, clamped_angle_b, self.config.ball_speed
                )
                hit_b = True
                self.rally_count += 1
                rewards[1] += self.config.reward_rally

            # Update ball position
            wall = self.ball.update(self.field)

            # Check for point end
            point_result = None
            if wall is not None:
                # Ball hit a wall - determine winner
                if self.ball.in_flag:
                    # Valid shot - hitter wins
                    if self.ball.last_hit_by is not None:
                        winner = self.ball.last_hit_by
                        point_result = self._end_point(winner, "in")
                        rewards[winner] += self.config.reward_point_win
                        rewards[1 - winner] += self.config.reward_point_lose
                    else:
                        # Serve that wasn't returned - goes to whoever didn't receive
                        # Determine which side the ball went to
                        if wall == "left":
                            winner = 1  # Player B wins (ball went to A's side)
                        else:
                            winner = 0  # Player A wins (ball went to B's side)
                        point_result = self._end_point(winner, "in")
                        rewards[winner] += self.config.reward_point_win
                        rewards[1 - winner] += self.config.reward_point_lose
                else:
                    # Invalid shot (out) - hitter loses
                    if self.ball.last_hit_by is not None:
                        loser = self.ball.last_hit_by
                        winner = 1 - loser
                        point_result = self._end_point(winner, "out")
                        rewards[winner] += self.config.reward_point_win
                        rewards[loser] += self.config.reward_point_lose
                    else:
                        # Serve went out without entering in-area
                        # This shouldn't happen with proper serve angles
                        # but handle it anyway - just restart
                        self._serve()

            # Check for timeout
            if self.steps_this_point >= self.config.max_steps_per_point:
                # Timeout - no winner, just restart
                self.state = GameState.POINT_OVER

        return StepResult(
            rewards=(rewards[0], rewards[1]),
            done=self.state == GameState.GAME_OVER,
            point_result=point_result if 'point_result' in dir() and point_result else None,
            hit_occurred=(hit_a, hit_b),
        )

    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current game observation for AI agents.

        Returns:
            Dictionary containing game state information
        """
        ball_x, ball_y = self.ball.position if self.ball else (0, 0)
        ball_vx, ball_vy = self.ball.velocity if self.ball else (0, 0)
        ball_in_flag = self.ball.in_flag if self.ball else False

        return {
            # Ball state
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_vx": ball_vx,
            "ball_vy": ball_vy,
            "ball_in_flag": ball_in_flag,
            # Player A state
            "player_a_x": self.player_a.x,
            "player_a_y": self.player_a.y,
            # Player B state
            "player_b_x": self.player_b.x,
            "player_b_y": self.player_b.y,
            # Game state
            "score_a": self.scores[0],
            "score_b": self.scores[1],
            "rally_count": self.rally_count,
            # Field info (static but useful for normalization)
            "field_width": self.config.field_width,
            "field_height": self.config.field_height,
        }

    def reset(self) -> Dict[str, Any]:
        """
        Reset the game to initial state.

        Returns:
            Initial observation
        """
        self.scores = [0, 0]
        self.total_steps = 0
        self.state = GameState.SERVING
        self._serve()
        return self.get_observation()

    @property
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.state == GameState.GAME_OVER

    @property
    def winner(self) -> Optional[int]:
        """Get the winner (0 or 1) if game is over, None otherwise."""
        if not self.is_game_over:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1
