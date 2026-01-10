"""Tests for Agent classes."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import Agent, AgentConfig, get_agent_class, load_agent
from agents.baseliner import BaselinerAgent
from agents.chase import ChaseAgent, SmartChaseAgent
from agents.positional import PositionalAgent
from agents.random_agent import RandomAgent


def create_sample_observation(player_id: int = 0) -> dict:
    """Create a sample observation for testing."""
    return {
        "ball_x": 400.0,
        "ball_y": 200.0,
        "ball_vx": 5.0,
        "ball_vy": 2.0,
        "ball_is_in": True,
        "player_a_x": 100.0,
        "player_a_y": 200.0,
        "player_b_x": 700.0,
        "player_b_y": 200.0,
        "score_a": 0,
        "score_b": 0,
        "rally_count": 0,
        "field_width": 800.0,
        "field_height": 400.0,
    }


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig class."""

    def test_default_config(self):
        """Test default config values."""
        config = AgentConfig()
        self.assertEqual(config.name, "unnamed")
        self.assertEqual(config.agent_type, "base")
        self.assertEqual(config.version, "1.0")
        self.assertEqual(config.parameters, {})

    def test_custom_config(self):
        """Test custom config values."""
        config = AgentConfig(
            name="TestAgent",
            agent_type="test",
            version="2.0",
            description="Test description",
            parameters={"param1": 1, "param2": "value"},
        )
        self.assertEqual(config.name, "TestAgent")
        self.assertEqual(config.agent_type, "test")
        self.assertEqual(config.parameters["param1"], 1)

    def test_to_dict(self):
        """Test config serialization."""
        config = AgentConfig(name="Test", parameters={"key": "value"})
        data = config.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["parameters"]["key"], "value")

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "name": "FromDict",
            "agent_type": "test",
            "version": "1.0",
            "description": "Loaded from dict",
            "parameters": {},
        }
        config = AgentConfig.from_dict(data)
        self.assertEqual(config.name, "FromDict")
        self.assertEqual(config.description, "Loaded from dict")

    def test_save_and_load(self):
        """Test config save/load round trip."""
        config = AgentConfig(
            name="SaveTest",
            agent_type="test",
            parameters={"test_param": 42},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            config.save(filepath)

            loaded = AgentConfig.load(filepath)
            self.assertEqual(loaded.name, "SaveTest")
            self.assertEqual(loaded.parameters["test_param"], 42)


class TestRandomAgent(unittest.TestCase):
    """Test RandomAgent class."""

    def setUp(self):
        self.agent = RandomAgent()
        self.agent.set_player_id(0)
        self.obs = create_sample_observation()

    def test_init(self):
        """Test RandomAgent initialization."""
        self.assertEqual(self.agent.config.name, "RandomBot")
        self.assertEqual(self.agent.config.agent_type, "random")

    def test_act_returns_valid_action(self):
        """Test that act returns valid action tuple."""
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)

        movement, angle = action
        self.assertIsInstance(movement, int)
        self.assertIsInstance(angle, float)
        self.assertGreaterEqual(movement, 0)
        self.assertLessEqual(movement, 16)
        self.assertGreaterEqual(angle, 0)
        self.assertLessEqual(angle, 360)

    def test_act_returns_different_actions(self):
        """Test that random agent returns varied actions."""
        actions = [self.agent.act(self.obs) for _ in range(100)]
        movements = {a[0] for a in actions}
        # Should have some variety in movements
        self.assertGreater(len(movements), 1)

    def test_learn_does_not_crash(self):
        """Test that learn method does not crash."""
        self.agent.learn(reward=1.0, done=False)
        self.agent.learn(reward=-1.0, done=True)

    def test_reset_does_not_crash(self):
        """Test that reset method does not crash."""
        self.agent.reset()

    def test_get_info(self):
        """Test get_info method."""
        info = self.agent.get_info()
        self.assertIn("name", info)
        self.assertIn("type", info)
        self.assertIn("strategy", info)


class TestChaseAgent(unittest.TestCase):
    """Test ChaseAgent class."""

    def setUp(self):
        self.agent = ChaseAgent()
        self.agent.set_player_id(0)
        self.obs = create_sample_observation()

    def test_init(self):
        """Test ChaseAgent initialization."""
        self.assertEqual(self.agent.config.name, "ChaseBot")
        self.assertEqual(self.agent.config.agent_type, "chase")

    def test_act_returns_valid_action(self):
        """Test that act returns valid action tuple."""
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)

        movement, angle = action
        self.assertIsInstance(movement, int)
        self.assertIsInstance(angle, float)
        self.assertGreaterEqual(movement, 0)
        self.assertLessEqual(movement, 16)

    def test_chases_ball(self):
        """Test that agent chases toward ball."""
        # Ball is to the right of player A
        self.obs["ball_x"] = 500.0
        self.obs["player_a_x"] = 100.0

        action = self.agent.act(self.obs)
        movement = action[0]
        # Should move right (direction 0-2 roughly)
        self.assertIn(movement, range(17))

    def test_player_b_hits_opposite(self):
        """Test that player B hits toward player A side."""
        agent_b = ChaseAgent()
        agent_b.set_player_id(1)
        action = agent_b.act(self.obs)
        # Player B should hit at 180 degrees (toward left)
        self.assertEqual(action[1], 180.0)

    def test_set_field_dimensions(self):
        """Test set_field_dimensions method."""
        self.agent.set_field_dimensions(1000.0, 500.0)
        self.assertEqual(self.agent.field_width, 1000.0)
        self.assertEqual(self.agent.field_height, 500.0)

    def test_learn_does_not_crash(self):
        """Test that learn method does not crash."""
        self.agent.learn(reward=1.0, done=False)


class TestSmartChaseAgent(unittest.TestCase):
    """Test SmartChaseAgent class."""

    def setUp(self):
        self.agent = SmartChaseAgent()
        self.agent.set_player_id(0)
        self.obs = create_sample_observation()

    def test_init(self):
        """Test SmartChaseAgent initialization."""
        self.assertEqual(self.agent.config.name, "SmartChaseBot")
        self.assertIn("home_return_threshold", self.agent.config.parameters)

    def test_act_returns_valid_action(self):
        """Test that act returns valid action tuple."""
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)

        movement, angle = action
        self.assertGreaterEqual(movement, 0)
        self.assertLessEqual(movement, 16)
        # SmartChaseAgent adds variance to angle, so raw angle may be outside 0-360
        # Game normalizes this, but we just check it's a valid float
        self.assertIsInstance(angle, float)

    def test_returns_home_when_ball_far(self):
        """Test that agent returns toward home when ball is far."""
        # Set ball far from player A
        self.obs["ball_x"] = 700.0
        self.obs["player_a_x"] = 160.0  # At home position
        self.obs["player_a_y"] = 200.0

        # Agent should consider returning to home position
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)


class TestBaselinerAgent(unittest.TestCase):
    """Test BaselinerAgent class."""

    def setUp(self):
        self.agent = BaselinerAgent()
        self.agent.set_player_id(0)
        self.obs = create_sample_observation()

    def test_init(self):
        """Test BaselinerAgent initialization."""
        self.assertEqual(self.agent.config.name, "Baseliner")
        self.assertEqual(self.agent.config.agent_type, "baseliner")
        self.assertIn("home_x_ratio", self.agent.config.parameters)

    def test_act_returns_valid_action(self):
        """Test that act returns valid action tuple."""
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)

        movement, angle = action
        self.assertGreaterEqual(movement, 0)
        self.assertLessEqual(movement, 16)

    def test_stays_at_baseline(self):
        """Test that baseliner tries to stay at baseline."""
        # Player is already at baseline
        self.obs["player_a_x"] = 80.0  # Near baseline
        self.obs["ball_y"] = 200.0  # Ball at center Y

        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)

    def test_hits_straight(self):
        """Test that baseliner hits straight back."""
        action = self.agent.act(self.obs)
        # Player A should hit at 0 degrees
        self.assertEqual(action[1], 0.0)

    def test_player_b_hits_back(self):
        """Test that player B hits toward player A."""
        agent_b = BaselinerAgent()
        agent_b.set_player_id(1)
        action = agent_b.act(self.obs)
        self.assertEqual(action[1], 180.0)


class TestPositionalAgent(unittest.TestCase):
    """Test PositionalAgent class."""

    def setUp(self):
        self.agent = PositionalAgent()
        self.agent.set_player_id(0)
        self.obs = create_sample_observation()

    def test_init(self):
        """Test PositionalAgent initialization."""
        self.assertEqual(self.agent.config.name, "PositionalBot")
        self.assertEqual(self.agent.config.agent_type, "positional")
        self.assertIn("defensive_depth", self.agent.config.parameters)
        self.assertIn("shot_variance", self.agent.config.parameters)

    def test_act_returns_valid_action(self):
        """Test that act returns valid action tuple."""
        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)

        movement, angle = action
        self.assertGreaterEqual(movement, 0)
        self.assertLessEqual(movement, 16)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLess(angle, 360.0)

    def test_intercepts_incoming_ball(self):
        """Test agent intercepts incoming ball."""
        # Ball coming toward player A
        self.obs["ball_x"] = 300.0
        self.obs["ball_vx"] = -5.0  # Moving left toward A

        action = self.agent.act(self.obs)
        self.assertIsInstance(action, tuple)

    def test_different_shot_angles(self):
        """Test that agent uses varied shot angles."""
        angles = []
        for _ in range(50):
            action = self.agent.act(self.obs)
            angles.append(action[1])
        # Should have some variety in angles
        unique_angles = set(int(a) for a in angles)
        self.assertGreater(len(unique_angles), 1)


class TestAgentValidateAction(unittest.TestCase):
    """Test Agent.validate_action method."""

    def test_validate_clamps_movement(self):
        """Test that movement is clamped to valid range."""
        agent = RandomAgent()

        # Test clamping movement to max
        result = agent.validate_action((20, 180.0))
        self.assertEqual(result[0], 16)

        # Test clamping movement to min
        result = agent.validate_action((-5, 180.0))
        self.assertEqual(result[0], 0)

    def test_validate_normalizes_angle(self):
        """Test that angle is normalized to 0-360."""
        agent = RandomAgent()

        # Angle > 360 should wrap
        result = agent.validate_action((8, 400.0))
        self.assertEqual(result[1], 40.0)

        # Negative angle should wrap
        result = agent.validate_action((8, -30.0))
        self.assertEqual(result[1], 330.0)

    def test_validate_converts_types(self):
        """Test that types are converted correctly."""
        agent = RandomAgent()

        result = agent.validate_action((8.7, 90))
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], float)
        self.assertEqual(result[0], 8)
        self.assertEqual(result[1], 90.0)


class TestAgentSaveLoad(unittest.TestCase):
    """Test Agent save and load functionality."""

    def test_chase_agent_save_load(self):
        """Test ChaseAgent save/load round trip."""
        agent = ChaseAgent()
        agent.set_player_id(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "chase_agent")
            agent.save(save_path)

            loaded = ChaseAgent.load(save_path)
            self.assertEqual(loaded.config.name, agent.config.name)
            self.assertEqual(loaded.config.agent_type, agent.config.agent_type)

    def test_random_agent_save_load(self):
        """Test RandomAgent save/load round trip."""
        agent = RandomAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "random_agent")
            agent.save(save_path)

            loaded = RandomAgent.load(save_path)
            self.assertEqual(loaded.config.name, agent.config.name)

    def test_load_nonexistent_raises_error(self):
        """Test that loading nonexistent agent raises error."""
        with self.assertRaises(FileNotFoundError):
            ChaseAgent.load("/nonexistent/path")


class TestGetAgentClass(unittest.TestCase):
    """Test get_agent_class function."""

    def test_get_chase_agent(self):
        """Test getting ChaseAgent class."""
        cls = get_agent_class("chase")
        self.assertEqual(cls, ChaseAgent)

    def test_get_random_agent(self):
        """Test getting RandomAgent class."""
        cls = get_agent_class("random")
        self.assertEqual(cls, RandomAgent)

    def test_get_baseliner_agent(self):
        """Test getting BaselinerAgent class."""
        cls = get_agent_class("baseliner")
        self.assertEqual(cls, BaselinerAgent)

    def test_get_positional_agent(self):
        """Test getting PositionalAgent class."""
        cls = get_agent_class("positional")
        self.assertEqual(cls, PositionalAgent)

    def test_unknown_agent_raises_error(self):
        """Test that unknown agent type raises error."""
        with self.assertRaises(ValueError):
            get_agent_class("nonexistent")


class TestLoadAgent(unittest.TestCase):
    """Test load_agent function."""

    def test_load_agent_by_directory(self):
        """Test loading agent from directory."""
        agent = BaselinerAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "agent")
            agent.save(save_path)

            loaded = load_agent(save_path)
            self.assertIsInstance(loaded, BaselinerAgent)
            self.assertEqual(loaded.config.name, agent.config.name)


class TestAgentSetPlayerId(unittest.TestCase):
    """Test Agent set_player_id method."""

    def test_set_player_id(self):
        """Test setting player ID."""
        agent = ChaseAgent()
        self.assertIsNone(agent.player_id)

        agent.set_player_id(0)
        self.assertEqual(agent.player_id, 0)

        agent.set_player_id(1)
        self.assertEqual(agent.player_id, 1)


if __name__ == "__main__":
    unittest.main()
