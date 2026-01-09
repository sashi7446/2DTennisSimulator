"""Game logic for 2D Tennis Simulator."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict, Any

from config import Config
from field import Field
from ball import Ball, create_serve_ball
from player import Player, create_players, NUM_MOVEMENT_ACTIONS


class GameState(Enum):
    SERVING = "serving"
    PLAYING = "playing"
    POINT_OVER = "point_over"
    GAME_OVER = "game_over"


@dataclass
class PointResult:
    winner: int
    reason: str
    last_hit_by: Optional[int]


@dataclass
class StepResult:
    rewards: Tuple[float, float]
    done: bool
    point_result: Optional[PointResult]
    hit_occurred: Tuple[bool, bool]


class Game:
    """Main game class managing tennis simulation state and logic."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.field = Field(self.config)
        self.player_a, self.player_b = create_players(self.field, self.config)
        self.ball: Optional[Ball] = None
        self.state = GameState.SERVING
        self.scores = [0, 0]
        self.steps_this_point = 0
        self.total_steps = 0
        self.rally_count = 0
        self._serve()

    def _serve(self) -> None:
        """Start a new point."""
        self.ball = create_serve_ball(self.field, self.config)
        self.state = GameState.PLAYING
        self.steps_this_point = 0
        self.rally_count = 0
        pos_a, pos_b = self.field.get_player_start_positions()
        self.player_a.reset(*pos_a)
        self.player_b.reset(*pos_b)

    def _end_point(self, winner: int, reason: str) -> PointResult:
        """End point and return result."""
        self.scores[winner] += 1
        self.state = GameState.GAME_OVER
        return PointResult(winner, reason, self.ball.last_hit_by if self.ball else None)

    def _award_point(self, winner: int, reason: str, rewards: list) -> PointResult:
        """Award point to winner and update rewards."""
        result = self._end_point(winner, reason)
        rewards[winner] += self.config.reward_point_win
        rewards[1 - winner] += self.config.reward_point_lose
        return result

    def _handle_wall_collision(self, wall: str, rewards: list) -> Optional[PointResult]:
        """Handle wall collision and determine point result."""
        if self.ball.is_in:
            # Valid shot - hitter or serve winner
            if self.ball.last_hit_by is not None:
                return self._award_point(self.ball.last_hit_by, "in", rewards)
            else:
                winner = 1 if wall == "left" else 0
                return self._award_point(winner, "in", rewards)
        elif self.ball.last_hit_by is not None:
            # Out - hitter loses
            return self._award_point(1 - self.ball.last_hit_by, "out", rewards)
        else:
            self._serve()
            return None

    def step(self, action_a: Tuple[int, float], action_b: Tuple[int, float]) -> StepResult:
        """Advance game by one step. Returns StepResult with rewards and state."""
        if self.state == GameState.GAME_OVER:
            return StepResult((0.0, 0.0), True, None, (False, False))

        if self.state == GameState.POINT_OVER:
            self._serve()

        self.steps_this_point += 1
        self.total_steps += 1

        move_a, hit_angle_a = action_a
        move_b, hit_angle_b = action_b
        self.player_a.move(move_a, self.field)
        self.player_b.move(move_b, self.field)

        hit_a, hit_b = False, False
        rewards = [0.0, 0.0]
        point_result = None

        if self.ball:
            # Try hits
            if self.player_a.can_hit(self.ball):
                self.player_a.hit_ball(self.ball, hit_angle_a, self.config.ball_speed)
                hit_a, self.rally_count = True, self.rally_count + 1
                rewards[0] += self.config.reward_rally
            elif self.player_b.can_hit(self.ball):
                self.player_b.hit_ball(self.ball, hit_angle_b, self.config.ball_speed)
                hit_b, self.rally_count = True, self.rally_count + 1
                rewards[1] += self.config.reward_rally

            was_in = self.ball.is_in
            wall = self.ball.update(self.field)

            # Reward when ball passes through in-area (is_in turns ON)
            if not was_in and self.ball.is_in and self.ball.last_hit_by is not None:
                rewards[self.ball.last_hit_by] += self.config.reward_in_area

            if wall:
                point_result = self._handle_wall_collision(wall, rewards)

            if self.steps_this_point >= self.config.max_steps_per_point:
                self.state = GameState.POINT_OVER

        return StepResult((rewards[0], rewards[1]), self.state == GameState.GAME_OVER, point_result, (hit_a, hit_b))

    def get_observation(self) -> Dict[str, Any]:
        """Get current game state for AI agents."""
        bx, by = self.ball.position if self.ball else (0, 0)
        bvx, bvy = self.ball.velocity if self.ball else (0, 0)
        return {
            "ball_x": bx, "ball_y": by, "ball_vx": bvx, "ball_vy": bvy,
            "ball_is_in": self.ball.is_in if self.ball else False,
            "player_a_x": self.player_a.x, "player_a_y": self.player_a.y,
            "player_b_x": self.player_b.x, "player_b_y": self.player_b.y,
            "score_a": self.scores[0], "score_b": self.scores[1],
            "rally_count": self.rally_count,
            "field_width": self.config.field_width, "field_height": self.config.field_height,
        }

    def reset(self) -> Dict[str, Any]:
        """Reset game to initial state."""
        self.scores = [0, 0]
        self.total_steps = 0
        self.state = GameState.SERVING
        self._serve()
        return self.get_observation()

    @property
    def is_game_over(self) -> bool:
        return self.state == GameState.GAME_OVER

    @property
    def winner(self) -> Optional[int]:
        return (0 if self.scores[0] > self.scores[1] else 1) if self.is_game_over else None
