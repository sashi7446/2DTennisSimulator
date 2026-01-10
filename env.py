"""Gymnasium environment for 2D Tennis Simulator."""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from config import Config
from game import Game
from player import NUM_MOVEMENT_ACTIONS


class TennisEnv:
    """Gymnasium-compatible two-player tennis environment."""

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium required: pip install gymnasium")

        self.config = config or Config()
        self.game = Game(self.config)
        self.render_mode = render_mode
        self.renderer = None

        self.observation_space = spaces.Dict({
            "ball_x": spaces.Box(0, 1, (1,), np.float32),
            "ball_y": spaces.Box(0, 1, (1,), np.float32),
            "ball_vx": spaces.Box(-1, 1, (1,), np.float32),
            "ball_vy": spaces.Box(-1, 1, (1,), np.float32),
            "ball_is_in": spaces.Discrete(2),
            "player_a_x": spaces.Box(0, 1, (1,), np.float32),
            "player_a_y": spaces.Box(0, 1, (1,), np.float32),
            "player_b_x": spaces.Box(0, 1, (1,), np.float32),
            "player_b_y": spaces.Box(0, 1, (1,), np.float32),
            "score_a": spaces.Box(0, np.inf, (1,), np.int32),
            "score_b": spaces.Box(0, np.inf, (1,), np.int32),
        })
        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_MOVEMENT_ACTIONS),
            spaces.Box(0, 360, (1,), np.float32),
        ))

    def _norm_factors(self) -> Tuple[float, float, float]:
        return self.config.field_width, self.config.field_height, self.config.ball_speed * 2

    def _normalize_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Normalize raw observation."""
        fw, fh, ms = self._norm_factors()
        return {
            "ball_x": np.array([obs["ball_x"] / fw], dtype=np.float32),
            "ball_y": np.array([obs["ball_y"] / fh], dtype=np.float32),
            "ball_vx": np.array([obs["ball_vx"] / ms], dtype=np.float32),
            "ball_vy": np.array([obs["ball_vy"] / ms], dtype=np.float32),
            "ball_is_in": 1 if obs["ball_is_in"] else 0,
            "player_a_x": np.array([obs["player_a_x"] / fw], dtype=np.float32),
            "player_a_y": np.array([obs["player_a_y"] / fh], dtype=np.float32),
            "player_b_x": np.array([obs["player_b_x"] / fw], dtype=np.float32),
            "player_b_y": np.array([obs["player_b_y"] / fh], dtype=np.float32),
            "score_a": np.array([obs["score_a"]], dtype=np.int32),
            "score_b": np.array([obs["score_b"]], dtype=np.int32),
        }

    def _get_flat_observation(self, player_id: int = 0) -> np.ndarray:
        """Get flattened observation from player's perspective."""
        obs = self.game.get_observation()
        fw, fh, ms = self._norm_factors()
        is_a = player_id == 0

        my_prefix, opp_prefix = ("player_a", "player_b") if is_a else ("player_b", "player_a")
        my_score, opp_score = ("score_a", "score_b") if is_a else ("score_b", "score_a")

        return np.array([
            obs["ball_x"] / fw, obs["ball_y"] / fh,
            obs["ball_vx"] / ms, obs["ball_vy"] / ms,
            1.0 if obs["ball_is_in"] else 0.0,
            obs[f"{my_prefix}_x"] / fw, obs[f"{my_prefix}_y"] / fh,
            obs[f"{opp_prefix}_x"] / fw, obs[f"{opp_prefix}_y"] / fh,
            obs[my_score], obs[opp_score],
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.game.reset()
        return self._normalize_observation(obs), {"raw_observation": obs}

    def step(self, action_a: Tuple[int, np.ndarray], action_b: Tuple[int, np.ndarray]):
        """Step environment. Returns (obs, rewards, terminated, truncated, info)."""
        def to_float(a): return float(a[0]) if hasattr(a, '__getitem__') else float(a)
        result = self.game.step((action_a[0], to_float(action_a[1])), (action_b[0], to_float(action_b[1])))
        obs = self.game.get_observation()
        info = {"raw_observation": obs, "point_result": result.point_result,
                "hit_occurred": result.hit_occurred, "scores": self.game.scores}
        return self._normalize_observation(obs), result.rewards, result.done, False, info

    def render(self) -> None:
        if self.render_mode is None:
            return
        if self.renderer is None:
            try:
                import pygame  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pygame is required for visual rendering mode. "
                    "Install with: pip install pygame"
                )
            from renderer import Renderer
            self.renderer = Renderer(self.config)
        self.renderer.render(self.game)
        self.renderer.handle_events()
        self.renderer.tick()

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
            self.renderer = None


class SinglePlayerTennisEnv:
    """Single-player environment (agent vs built-in opponent)."""

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None,
                 opponent_policy: str = "random"):
        self.env = TennisEnv(config, render_mode)
        self.config = self.env.config
        self.opponent_policy = opponent_policy
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(0, 1, (11,), np.float32)

    def _get_opponent_action(self) -> Tuple[int, np.ndarray]:
        if self.opponent_policy == "random":
            return (np.random.randint(0, NUM_MOVEMENT_ACTIONS), np.array([np.random.uniform(0, 360)], dtype=np.float32))
        elif self.opponent_policy == "chase":
            obs = self.env.game.get_observation()
            dx, dy = obs["ball_x"] - obs["player_b_x"], obs["ball_y"] - obs["player_b_y"]
            direction = int((math.degrees(math.atan2(dy, dx)) % 360) / 22.5) % 16
            hit_angle = 180.0 if obs["ball_x"] > self.config.field_width / 2 else 0.0
            return (direction, np.array([hit_angle], dtype=np.float32))
        raise ValueError(f"Unknown opponent policy: {self.opponent_policy}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        _, info = self.env.reset(seed, options)
        return self.env._get_flat_observation(0), info

    def step(self, action: Tuple[int, np.ndarray]):
        _, rewards, terminated, truncated, info = self.env.step(action, self._get_opponent_action())
        return self.env._get_flat_observation(0), rewards[0], terminated, truncated, info

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
