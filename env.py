"""Gymnasium environment for 2D Tennis Simulator."""

from typing import Optional, Tuple, Dict, Any, SupportsFloat
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from config import Config
from game import Game
from player import NUM_MOVEMENT_ACTIONS


class TennisEnv:
    """
    Gymnasium-compatible environment for the 2D Tennis Simulator.

    This environment supports two-player training where each player
    has a hybrid action space:
    - Discrete movement (17 options: 16 directions + stay)
    - Continuous hit angle (0-360 degrees)

    Observations include:
    - Ball position and velocity
    - Both player positions
    - Ball in-flag state
    - Scores
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None,
    ):
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is required. Install with: pip install gymnasium"
            )

        self.config = config or Config()
        self.game = Game(self.config)
        self.render_mode = render_mode
        self.renderer = None

        # Define observation space
        # Normalized observations in range [0, 1] or [-1, 1]
        self.observation_space = spaces.Dict({
            # Ball state (normalized to field dimensions)
            "ball_x": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "ball_y": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "ball_vx": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "ball_vy": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "ball_in_flag": spaces.Discrete(2),
            # Player positions (normalized)
            "player_a_x": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_a_y": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_b_x": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_b_y": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # Game state
            "score_a": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "score_b": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        })

        # Define action space for each player
        # Tuple of (discrete movement, continuous angle)
        self.action_space = spaces.Tuple((
            spaces.Discrete(NUM_MOVEMENT_ACTIONS),  # Movement direction
            spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),  # Hit angle
        ))

    def _normalize_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Normalize raw observation to environment observation space."""
        field_w = self.config.field_width
        field_h = self.config.field_height
        max_speed = self.config.ball_speed * 2  # Normalize velocity

        return {
            "ball_x": np.array([obs["ball_x"] / field_w], dtype=np.float32),
            "ball_y": np.array([obs["ball_y"] / field_h], dtype=np.float32),
            "ball_vx": np.array([obs["ball_vx"] / max_speed], dtype=np.float32),
            "ball_vy": np.array([obs["ball_vy"] / max_speed], dtype=np.float32),
            "ball_in_flag": 1 if obs["ball_in_flag"] else 0,
            "player_a_x": np.array([obs["player_a_x"] / field_w], dtype=np.float32),
            "player_a_y": np.array([obs["player_a_y"] / field_h], dtype=np.float32),
            "player_b_x": np.array([obs["player_b_x"] / field_w], dtype=np.float32),
            "player_b_y": np.array([obs["player_b_y"] / field_h], dtype=np.float32),
            "score_a": np.array([obs["score_a"]], dtype=np.int32),
            "score_b": np.array([obs["score_b"]], dtype=np.int32),
        }

    def _get_flat_observation(self, player_id: int = 0) -> np.ndarray:
        """
        Get a flattened observation array for a specific player.

        The observation is from the perspective of the specified player,
        with their own position first.

        Args:
            player_id: 0 for player A, 1 for player B

        Returns:
            Flattened numpy array of observations
        """
        obs = self.game.get_observation()
        field_w = self.config.field_width
        field_h = self.config.field_height
        max_speed = self.config.ball_speed * 2

        # Ball state
        ball_obs = [
            obs["ball_x"] / field_w,
            obs["ball_y"] / field_h,
            obs["ball_vx"] / max_speed,
            obs["ball_vy"] / max_speed,
            1.0 if obs["ball_in_flag"] else 0.0,
        ]

        # Player states (own position first)
        if player_id == 0:
            player_obs = [
                obs["player_a_x"] / field_w,
                obs["player_a_y"] / field_h,
                obs["player_b_x"] / field_w,
                obs["player_b_y"] / field_h,
            ]
            score_obs = [
                obs["score_a"] / self.config.points_to_win,
                obs["score_b"] / self.config.points_to_win,
            ]
        else:
            player_obs = [
                obs["player_b_x"] / field_w,
                obs["player_b_y"] / field_h,
                obs["player_a_x"] / field_w,
                obs["player_a_y"] / field_h,
            ]
            score_obs = [
                obs["score_b"] / self.config.points_to_win,
                obs["score_a"] / self.config.points_to_win,
            ]

        return np.array(ball_obs + player_obs + score_obs, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        obs = self.game.reset()
        normalized_obs = self._normalize_observation(obs)
        info = {"raw_observation": obs}

        return normalized_obs, info

    def step(
        self,
        action_a: Tuple[int, np.ndarray],
        action_b: Tuple[int, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Tuple[float, float], bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action_a: (movement_direction, hit_angle_array) for player A
            action_b: (movement_direction, hit_angle_array) for player B

        Returns:
            Tuple of (observation, rewards, terminated, truncated, info)
        """
        # Convert numpy arrays to floats for hit angles
        hit_angle_a = float(action_a[1][0]) if hasattr(action_a[1], '__getitem__') else float(action_a[1])
        hit_angle_b = float(action_b[1][0]) if hasattr(action_b[1], '__getitem__') else float(action_b[1])

        # Step the game
        result = self.game.step(
            (action_a[0], hit_angle_a),
            (action_b[0], hit_angle_b),
        )

        # Get observation
        obs = self.game.get_observation()
        normalized_obs = self._normalize_observation(obs)

        # Build info dict
        info = {
            "raw_observation": obs,
            "point_result": result.point_result,
            "hit_occurred": result.hit_occurred,
            "scores": self.game.scores,
        }

        terminated = result.done
        truncated = False

        return normalized_obs, result.rewards, terminated, truncated, info

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode is None:
            return

        if self.renderer is None:
            from renderer import Renderer
            self.renderer = Renderer(self.config)

        self.renderer.render(self.game)
        self.renderer.handle_events()
        self.renderer.tick()

    def close(self) -> None:
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class SinglePlayerTennisEnv:
    """
    Single-player version of the Tennis environment.

    Player A is controlled by the agent, Player B by a simple policy.
    This is useful for single-agent RL algorithms.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None,
        opponent_policy: str = "random",
    ):
        self.env = TennisEnv(config, render_mode)
        self.config = self.env.config
        self.opponent_policy = opponent_policy

        # Single player action space
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

    def _get_opponent_action(self) -> Tuple[int, np.ndarray]:
        """Get action for the opponent based on policy."""
        if self.opponent_policy == "random":
            move = np.random.randint(0, NUM_MOVEMENT_ACTIONS)
            angle = np.random.uniform(0, 360)
            return (move, np.array([angle], dtype=np.float32))
        elif self.opponent_policy == "chase":
            # Simple policy: move toward ball
            obs = self.env.game.get_observation()
            ball_x = obs["ball_x"]
            ball_y = obs["ball_y"]
            player_x = obs["player_b_x"]
            player_y = obs["player_b_y"]

            # Calculate direction to ball
            dx = ball_x - player_x
            dy = ball_y - player_y

            # Convert to discrete direction
            import math
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            direction = int(angle / 22.5) % 16

            # Hit toward opposite side
            hit_angle = 180.0 if ball_x > self.config.field_width / 2 else 0.0

            return (direction, np.array([hit_angle], dtype=np.float32))
        else:
            raise ValueError(f"Unknown opponent policy: {self.opponent_policy}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        _, info = self.env.reset(seed, options)
        obs = self.env._get_flat_observation(player_id=0)
        return obs, info

    def step(
        self,
        action: Tuple[int, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step with player A's action."""
        action_b = self._get_opponent_action()
        _, rewards, terminated, truncated, info = self.env.step(action, action_b)

        obs = self.env._get_flat_observation(player_id=0)
        reward = rewards[0]  # Player A's reward

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment."""
        self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()
