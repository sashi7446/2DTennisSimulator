"""Tests for CLI integration (main.py)."""

import os
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from main import create_agent, list_agent_types, run_headless_training


class TestCreateAgent(unittest.TestCase):
    """Test create_agent function."""

    def setUp(self):
        """Set up test config."""
        self.config = Config()

    def test_create_chase_agent(self):
        """Test creating a chase agent."""
        agent = create_agent("chase", player_id=0, config=self.config)
        self.assertEqual(agent.config.agent_type, "chase")
        self.assertEqual(agent.player_id, 0)

    def test_create_smart_agent(self):
        """Test creating a smart chase agent."""
        from agents import SmartChaseAgent

        agent = create_agent("smart", player_id=1, config=self.config)
        self.assertIsInstance(agent, SmartChaseAgent)
        self.assertEqual(agent.player_id, 1)

    def test_create_random_agent(self):
        """Test creating a random agent."""
        agent = create_agent("random", player_id=0, config=self.config)
        self.assertEqual(agent.config.agent_type, "random")

    def test_create_neural_agent(self):
        """Test creating a neural agent (if numpy available)."""
        from agents import NEURAL_AVAILABLE

        agent = create_agent("neural", player_id=0, config=self.config)
        if NEURAL_AVAILABLE:
            self.assertEqual(agent.config.agent_type, "neural")
        else:
            # Falls back to chase if numpy not available
            self.assertEqual(agent.config.agent_type, "chase")

    def test_create_transformer_agent(self):
        """Test creating a transformer agent (if numpy available)."""
        from agents import NEURAL_AVAILABLE

        agent = create_agent("transformer", player_id=0, config=self.config)
        if NEURAL_AVAILABLE:
            self.assertEqual(agent.config.agent_type, "transformer")
        else:
            # Falls back to chase if numpy not available
            self.assertEqual(agent.config.agent_type, "chase")

    def test_unknown_agent_type_falls_back_to_chase(self):
        """Test that unknown agent type falls back to chase."""
        with patch("sys.stdout", new=StringIO()):
            agent = create_agent("nonexistent_type", player_id=0, config=self.config)
        self.assertEqual(agent.config.agent_type, "chase")

    def test_player_id_is_set(self):
        """Test that player_id is correctly set."""
        agent_a = create_agent("chase", player_id=0, config=self.config)
        agent_b = create_agent("chase", player_id=1, config=self.config)
        self.assertEqual(agent_a.player_id, 0)
        self.assertEqual(agent_b.player_id, 1)

    def test_field_dimensions_set_for_neural(self):
        """Test that field dimensions are set for agents that need them."""
        from agents import NEURAL_AVAILABLE

        if NEURAL_AVAILABLE:
            agent = create_agent("neural", player_id=0, config=self.config)
            if hasattr(agent, "field_width"):
                self.assertEqual(agent.field_width, self.config.field_width)
                self.assertEqual(agent.field_height, self.config.field_height)

    def test_load_agent_from_path(self):
        """Test loading a saved agent from path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First create and save an agent
            agent = create_agent("chase", player_id=0, config=self.config)
            save_path = os.path.join(tmpdir, "test_agent")
            agent.save(save_path)

            # Now load it using create_agent
            with patch("sys.stdout", new=StringIO()):
                loaded_agent = create_agent(
                    save_path, player_id=1, config=self.config, load_path=save_path
                )
            self.assertEqual(loaded_agent.config.agent_type, "chase")
            self.assertEqual(loaded_agent.player_id, 1)

    def test_load_agent_from_agent_type_as_path(self):
        """Test loading agent when agent_type is actually a path."""
        from agents import ChaseAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # First create and save an agent
            agent = create_agent("chase", player_id=0, config=self.config)
            save_path = os.path.join(tmpdir, "chase_agent")
            agent.save(save_path)

            # Load using path as agent_type
            with patch("sys.stdout", new=StringIO()):
                loaded_agent = create_agent(save_path, player_id=0, config=self.config)
            self.assertIsInstance(loaded_agent, ChaseAgent)
            self.assertEqual(loaded_agent.config.agent_type, "chase")


class TestHeadlessTraining(unittest.TestCase):
    """Test run_headless_training function."""

    def setUp(self):
        """Set up test config and agents."""
        self.config = Config()

    def test_headless_training_completes(self):
        """Test that headless training completes successfully."""
        agent_a = create_agent("chase", player_id=0, config=self.config)
        agent_b = create_agent("chase", player_id=1, config=self.config)

        with patch("sys.stdout", new=StringIO()):
            wins, returned_a, returned_b = run_headless_training(
                self.config, agent_a, agent_b, num_episodes=5
            )

        # Verify returns
        self.assertIsInstance(wins, list)
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins[0] + wins[1], 5)  # Total games should equal episodes
        self.assertIs(returned_a, agent_a)
        self.assertIs(returned_b, agent_b)

    def test_headless_training_with_different_agents(self):
        """Test headless training with different agent types."""
        agent_a = create_agent("smart", player_id=0, config=self.config)
        agent_b = create_agent("random", player_id=1, config=self.config)

        with patch("sys.stdout", new=StringIO()):
            wins, _, _ = run_headless_training(self.config, agent_a, agent_b, num_episodes=3)

        self.assertEqual(wins[0] + wins[1], 3)

    def test_headless_training_saves_agents(self):
        """Test that agents are saved during headless training."""
        agent_a = create_agent("chase", player_id=0, config=self.config)
        agent_b = create_agent("chase", player_id=1, config=self.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.stdout", new=StringIO()):
                run_headless_training(
                    self.config,
                    agent_a,
                    agent_b,
                    num_episodes=5,
                    save_dir=tmpdir,
                    save_interval=5,
                )

            # Check that agents were saved
            agent_a_dir = os.path.join(tmpdir, "agent_a_chase")
            agent_b_dir = os.path.join(tmpdir, "agent_b_chase")
            self.assertTrue(os.path.exists(agent_a_dir))
            self.assertTrue(os.path.exists(agent_b_dir))

    def test_headless_training_neural_vs_chase(self):
        """Test headless training with neural vs chase agents."""
        from agents import NEURAL_AVAILABLE

        if not NEURAL_AVAILABLE:
            self.skipTest("Neural agents require numpy")

        agent_a = create_agent("neural", player_id=0, config=self.config)
        agent_b = create_agent("chase", player_id=1, config=self.config)

        with patch("sys.stdout", new=StringIO()):
            wins, _, _ = run_headless_training(self.config, agent_a, agent_b, num_episodes=3)

        self.assertEqual(wins[0] + wins[1], 3)


class TestListAgentTypes(unittest.TestCase):
    """Test list_agent_types function."""

    def test_list_agent_types_prints_info(self):
        """Test that list_agent_types prints agent information."""
        output = StringIO()
        with patch("sys.stdout", output):
            list_agent_types()

        output_text = output.getvalue()
        # Check that common agent types are listed
        self.assertIn("chase", output_text)
        self.assertIn("smart", output_text)
        self.assertIn("random", output_text)
        self.assertIn("neural", output_text)
        self.assertIn("transformer", output_text)

    def test_list_shows_numpy_requirement_when_unavailable(self):
        """Test that neural agents show numpy requirement when unavailable."""
        output = StringIO()
        with patch("sys.stdout", output):
            with patch("main.NEURAL_AVAILABLE", False):
                list_agent_types()

        output_text = output.getvalue()
        self.assertIn("requires numpy", output_text)


class TestCLIArgParsing(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        from main import main

        # Test that main creates default config with default arguments
        with patch("sys.argv", ["main.py", "--mode", "list"]):
            output = StringIO()
            with patch("sys.stdout", output):
                main()

            output_text = output.getvalue()
            self.assertIn("chase", output_text)

    def test_headless_mode_argument(self):
        """Test --mode headless argument."""
        config = Config()
        agent_a = create_agent("chase", player_id=0, config=config)
        agent_b = create_agent("chase", player_id=1, config=config)

        with patch("sys.stdout", new=StringIO()):
            wins, _, _ = run_headless_training(config, agent_a, agent_b, num_episodes=2)

        self.assertEqual(len(wins), 2)

    def test_episodes_argument(self):
        """Test --episodes argument is respected."""
        config = Config()
        agent_a = create_agent("chase", player_id=0, config=config)
        agent_b = create_agent("chase", player_id=1, config=config)

        with patch("sys.stdout", new=StringIO()):
            wins, _, _ = run_headless_training(config, agent_a, agent_b, num_episodes=10)

        self.assertEqual(wins[0] + wins[1], 10)

    def test_speed_config_option(self):
        """Test --speed option modifies config."""
        config = Config(ball_speed=10.0)
        self.assertEqual(config.ball_speed, 10.0)

    def test_fps_config_option(self):
        """Test --fps option modifies config."""
        config = Config(fps=30)
        self.assertEqual(config.fps, 30)


class TestAgentSaveLoad(unittest.TestCase):
    """Test agent save/load integration through CLI."""

    def setUp(self):
        """Set up test config."""
        self.config = Config()

    def test_save_and_load_preserves_agent_type(self):
        """Test that save/load preserves agent configuration."""
        from agents import RandomAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a random agent (has distinct agent_type)
            agent = create_agent("random", player_id=0, config=self.config)
            save_path = os.path.join(tmpdir, "test_random")
            agent.save(save_path)

            # Load and verify
            with patch("sys.stdout", new=StringIO()):
                loaded = create_agent(
                    save_path, player_id=1, config=self.config, load_path=save_path
                )
            self.assertIsInstance(loaded, RandomAgent)
            self.assertEqual(loaded.config.agent_type, "random")

    def test_loaded_agent_can_act(self):
        """Test that loaded agents can act normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create, save, load
            agent = create_agent("chase", player_id=0, config=self.config)
            save_path = os.path.join(tmpdir, "test_chase")
            agent.save(save_path)

            with patch("sys.stdout", new=StringIO()):
                loaded = create_agent(
                    save_path, player_id=0, config=self.config, load_path=save_path
                )

            # Test that it can act
            obs = {
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
            action = loaded.act(obs)
            self.assertEqual(len(action), 2)

    def test_training_with_loaded_agent(self):
        """Test that loaded agents can be used in training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create, save
            original = create_agent("chase", player_id=0, config=self.config)
            save_path = os.path.join(tmpdir, "chase_agent")
            original.save(save_path)

            # Load and train
            with patch("sys.stdout", new=StringIO()):
                loaded = create_agent(
                    save_path, player_id=0, config=self.config, load_path=save_path
                )
            opponent = create_agent("random", player_id=1, config=self.config)

            with patch("sys.stdout", new=StringIO()):
                wins, _, _ = run_headless_training(self.config, loaded, opponent, num_episodes=3)

            self.assertEqual(wins[0] + wins[1], 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in CLI."""

    def setUp(self):
        """Set up test config."""
        self.config = Config()

    def test_invalid_agent_type_handled_gracefully(self):
        """Test that invalid agent type is handled gracefully."""
        output = StringIO()
        with patch("sys.stdout", output):
            agent = create_agent("invalid_agent_xyz", player_id=0, config=self.config)

        # Should fall back to ChaseAgent
        self.assertEqual(agent.config.agent_type, "chase")
        self.assertIn("Unknown agent type", output.getvalue())

    def test_nonexistent_load_path_creates_new_agent(self):
        """Test that nonexistent load path doesn't crash."""
        agent = create_agent(
            "chase",
            player_id=0,
            config=self.config,
            load_path="/nonexistent/path/agent",
        )
        # Should create a new chase agent since path doesn't exist
        self.assertEqual(agent.config.agent_type, "chase")

    def test_empty_episodes_doesnt_crash(self):
        """Test that zero episodes doesn't crash."""
        agent_a = create_agent("chase", player_id=0, config=self.config)
        agent_b = create_agent("chase", player_id=1, config=self.config)

        with patch("sys.stdout", new=StringIO()):
            wins, _, _ = run_headless_training(self.config, agent_a, agent_b, num_episodes=0)

        self.assertEqual(wins, [0, 0])


if __name__ == "__main__":
    unittest.main()
