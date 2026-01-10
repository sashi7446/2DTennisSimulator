"""Tests for Gymnasium environment classes."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from config import Config
from player import NUM_MOVEMENT_ACTIONS

# Import env components only if numpy is available
if NUMPY_AVAILABLE:
    from env import GYM_AVAILABLE, SinglePlayerTennisEnv, TennisEnv
else:
    GYM_AVAILABLE = False
    TennisEnv = None
    SinglePlayerTennisEnv = None


def requires_numpy_and_gym(cls):
    """Skip entire test class if numpy or gymnasium not available."""
    if not NUMPY_AVAILABLE or not GYM_AVAILABLE:
        return unittest.skip("numpy and gymnasium required")(cls)
    return cls


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvInit(unittest.TestCase):
    """Test TennisEnv initialization."""

    def test_init_default(self):
        """Test default initialization."""
        env = TennisEnv()
        self.assertIsNotNone(env.game)
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)
        self.assertIsNone(env.render_mode)

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = Config(field_width=1000, field_height=500)
        env = TennisEnv(config=config)
        self.assertEqual(env.config.field_width, 1000)
        self.assertEqual(env.config.field_height, 500)

    def test_init_with_render_mode(self):
        """Test initialization with render mode."""
        env = TennisEnv(render_mode="human")
        self.assertEqual(env.render_mode, "human")


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvObservationSpace(unittest.TestCase):
    """Test TennisEnv observation space."""

    def setUp(self):
        self.env = TennisEnv()

    def test_observation_space_is_dict(self):
        """Test that observation space is a Dict space."""
        from gymnasium import spaces

        self.assertIsInstance(self.env.observation_space, spaces.Dict)

    def test_observation_space_keys(self):
        """Test that observation space has all required keys."""
        expected_keys = {
            "ball_x",
            "ball_y",
            "ball_vx",
            "ball_vy",
            "ball_is_in",
            "player_a_x",
            "player_a_y",
            "player_b_x",
            "player_b_y",
            "score_a",
            "score_b",
        }
        actual_keys = set(self.env.observation_space.spaces.keys())
        self.assertEqual(expected_keys, actual_keys)

    def test_observation_contains_in_space(self):
        """Test that observations are within space bounds."""
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvActionSpace(unittest.TestCase):
    """Test TennisEnv action space."""

    def setUp(self):
        self.env = TennisEnv()

    def test_action_space_is_tuple(self):
        """Test that action space is a Tuple space."""
        from gymnasium import spaces

        self.assertIsInstance(self.env.action_space, spaces.Tuple)

    def test_action_space_components(self):
        """Test action space components."""
        from gymnasium import spaces

        # Movement is Discrete(NUM_MOVEMENT_ACTIONS)
        self.assertIsInstance(self.env.action_space[0], spaces.Discrete)
        self.assertEqual(self.env.action_space[0].n, NUM_MOVEMENT_ACTIONS)

        # Hit angle is Box(0, 360, (1,))
        self.assertIsInstance(self.env.action_space[1], spaces.Box)


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvReset(unittest.TestCase):
    """Test TennisEnv reset."""

    def setUp(self):
        self.env = TennisEnv()

    def test_reset_returns_observation_and_info(self):
        """Test that reset returns (observation, info) tuple."""
        result = self.env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        obs, info = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

    def test_reset_observation_format(self):
        """Test that reset returns correctly formatted observation."""
        obs, _ = self.env.reset()

        # Check key exists and is numpy array
        self.assertIn("ball_x", obs)
        self.assertIsInstance(obs["ball_x"], np.ndarray)

        # Check normalized values are in [0, 1]
        self.assertGreaterEqual(obs["ball_x"][0], 0)
        self.assertLessEqual(obs["ball_x"][0], 1)

    def test_reset_with_seed(self):
        """Test reset with seed."""
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=42)

        # Same seed should produce same initial state
        np.testing.assert_array_equal(obs1["ball_x"], obs2["ball_x"])
        np.testing.assert_array_equal(obs1["ball_y"], obs2["ball_y"])

    def test_reset_info_contains_raw_observation(self):
        """Test that info contains raw observation."""
        _, info = self.env.reset()
        self.assertIn("raw_observation", info)
        raw = info["raw_observation"]
        self.assertIn("ball_x", raw)
        self.assertIn("player_a_x", raw)


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvStep(unittest.TestCase):
    """Test TennisEnv step."""

    def setUp(self):
        self.env = TennisEnv()
        self.env.reset()

    def test_step_returns_five_values(self):
        """Test that step returns (obs, rewards, terminated, truncated, info)."""
        action_a = (0, np.array([0.0]))
        action_b = (0, np.array([180.0]))

        result = self.env.step(action_a, action_b)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

        obs, rewards, terminated, truncated, info = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rewards, tuple)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_rewards_format(self):
        """Test that rewards are tuple of two floats."""
        action_a = (16, np.array([0.0]))  # Stay
        action_b = (16, np.array([180.0]))  # Stay

        _, rewards, _, _, _ = self.env.step(action_a, action_b)

        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(rewards[0], (int, float))
        self.assertIsInstance(rewards[1], (int, float))

    def test_step_observation_in_space(self):
        """Test that step observation is within space."""
        action_a = (8, np.array([45.0]))
        action_b = (8, np.array([135.0]))

        obs, _, _, _, _ = self.env.step(action_a, action_b)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_step_info_contains_required_keys(self):
        """Test that info contains required keys."""
        action_a = (0, np.array([0.0]))
        action_b = (0, np.array([180.0]))

        _, _, _, _, info = self.env.step(action_a, action_b)

        self.assertIn("raw_observation", info)
        self.assertIn("hit_occurred", info)
        self.assertIn("scores", info)

    def test_multiple_steps(self):
        """Test running multiple steps."""
        for _ in range(10):
            action_a = (np.random.randint(0, NUM_MOVEMENT_ACTIONS), np.array([np.random.uniform(0, 360)]))
            action_b = (np.random.randint(0, NUM_MOVEMENT_ACTIONS), np.array([np.random.uniform(0, 360)]))

            obs, rewards, done, truncated, info = self.env.step(action_a, action_b)

            if done:
                self.env.reset()
                continue

            self.assertTrue(self.env.observation_space.contains(obs))


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvNormalization(unittest.TestCase):
    """Test observation normalization."""

    def setUp(self):
        self.env = TennisEnv()
        self.env.reset()

    def test_normalized_positions_in_range(self):
        """Test that normalized positions are in [0, 1]."""
        obs, _ = self.env.reset()

        for key in ["ball_x", "ball_y", "player_a_x", "player_a_y", "player_b_x", "player_b_y"]:
            self.assertGreaterEqual(obs[key][0], 0.0, f"{key} should be >= 0")
            self.assertLessEqual(obs[key][0], 1.0, f"{key} should be <= 1")

    def test_normalized_velocities_in_range(self):
        """Test that normalized velocities are in [-1, 1]."""
        obs, _ = self.env.reset()

        for key in ["ball_vx", "ball_vy"]:
            self.assertGreaterEqual(obs[key][0], -1.0, f"{key} should be >= -1")
            self.assertLessEqual(obs[key][0], 1.0, f"{key} should be <= 1")


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestTennisEnvClose(unittest.TestCase):
    """Test TennisEnv close."""

    def test_close_without_renderer(self):
        """Test close when no renderer was initialized."""
        env = TennisEnv()
        env.close()  # Should not raise

    def test_close_cleans_renderer(self):
        """Test that close cleans up renderer reference."""
        env = TennisEnv()
        env.renderer = None  # Simulate no renderer
        env.close()
        self.assertIsNone(env.renderer)


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestSinglePlayerTennisEnvInit(unittest.TestCase):
    """Test SinglePlayerTennisEnv initialization."""

    def test_init_default(self):
        """Test default initialization."""
        env = SinglePlayerTennisEnv()
        self.assertIsNotNone(env.env)
        self.assertEqual(env.opponent_policy, "random")

    def test_init_with_chase_opponent(self):
        """Test initialization with chase opponent."""
        env = SinglePlayerTennisEnv(opponent_policy="chase")
        self.assertEqual(env.opponent_policy, "chase")

    def test_observation_space_is_box(self):
        """Test that observation space is flat Box."""
        from gymnasium import spaces

        env = SinglePlayerTennisEnv()
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.shape, (11,))


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestSinglePlayerTennisEnvReset(unittest.TestCase):
    """Test SinglePlayerTennisEnv reset."""

    def setUp(self):
        self.env = SinglePlayerTennisEnv()

    def test_reset_returns_flat_observation(self):
        """Test that reset returns flat numpy array."""
        obs, info = self.env.reset()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (11,))

    def test_reset_observation_in_range(self):
        """Test that reset observation values are normalized."""
        obs, _ = self.env.reset()

        # Most values should be in [0, 1]
        for i in range(9):
            self.assertGreaterEqual(obs[i], 0.0)
            self.assertLessEqual(obs[i], 1.0)


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestSinglePlayerTennisEnvStep(unittest.TestCase):
    """Test SinglePlayerTennisEnv step."""

    def setUp(self):
        self.env = SinglePlayerTennisEnv()
        self.env.reset()

    def test_step_returns_five_values(self):
        """Test that step returns (obs, reward, terminated, truncated, info)."""
        action = (0, np.array([0.0]))
        result = self.env.step(action)

        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result

        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_observation_shape(self):
        """Test that step returns correct observation shape."""
        action = (8, np.array([90.0]))
        obs, _, _, _, _ = self.env.step(action)

        self.assertEqual(obs.shape, (11,))

    def test_multiple_steps(self):
        """Test running multiple steps."""
        for _ in range(10):
            action = (np.random.randint(0, NUM_MOVEMENT_ACTIONS), np.array([np.random.uniform(0, 360)]))

            obs, reward, done, truncated, info = self.env.step(action)

            if done:
                self.env.reset()
                continue

            self.assertEqual(obs.shape, (11,))


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestSinglePlayerTennisEnvOpponents(unittest.TestCase):
    """Test SinglePlayerTennisEnv opponent policies."""

    def test_random_opponent(self):
        """Test random opponent policy."""
        env = SinglePlayerTennisEnv(opponent_policy="random")
        env.reset()

        # Run several steps to ensure opponent acts
        for _ in range(5):
            action = (8, np.array([90.0]))
            env.step(action)

    def test_chase_opponent(self):
        """Test chase opponent policy."""
        env = SinglePlayerTennisEnv(opponent_policy="chase")
        env.reset()

        # Run several steps to ensure opponent acts
        for _ in range(5):
            action = (8, np.array([90.0]))
            env.step(action)

    def test_invalid_opponent_raises_error(self):
        """Test that invalid opponent policy raises error."""
        env = SinglePlayerTennisEnv(opponent_policy="invalid")
        env.reset()

        with self.assertRaises(ValueError):
            env.step((0, np.array([0.0])))


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestSinglePlayerTennisEnvClose(unittest.TestCase):
    """Test SinglePlayerTennisEnv close."""

    def test_close(self):
        """Test close method."""
        env = SinglePlayerTennisEnv()
        env.reset()
        env.close()  # Should not raise


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium not installed")
class TestFlatObservation(unittest.TestCase):
    """Test flat observation generation."""

    def test_flat_observation_player_a(self):
        """Test flat observation from player A perspective."""
        env = TennisEnv()
        env.reset()

        obs = env._get_flat_observation(player_id=0)
        self.assertEqual(len(obs), 11)
        self.assertIsInstance(obs, np.ndarray)

    def test_flat_observation_player_b(self):
        """Test flat observation from player B perspective."""
        env = TennisEnv()
        env.reset()

        obs = env._get_flat_observation(player_id=1)
        self.assertEqual(len(obs), 11)

    def test_flat_observation_perspective_differs(self):
        """Test that perspectives give different observations."""
        env = TennisEnv()
        env.reset()

        obs_a = env._get_flat_observation(player_id=0)
        obs_b = env._get_flat_observation(player_id=1)

        # My position and opponent position should be swapped
        # obs[5:7] is my position, obs[7:9] is opponent position
        np.testing.assert_array_almost_equal(obs_a[5:7], obs_b[7:9])
        np.testing.assert_array_almost_equal(obs_a[7:9], obs_b[5:7])


if __name__ == "__main__":
    unittest.main()
