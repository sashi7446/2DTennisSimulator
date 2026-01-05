"""Tests for Game class."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from config import Config
from game import Game, GameState, PointResult, StepResult


class TestGameInitialization(unittest.TestCase):
    """Test Game initialization."""

    def test_initial_state(self):
        """Game should start in PLAYING state after serve."""
        game = Game()
        self.assertEqual(game.state, GameState.PLAYING)
        self.assertIsNotNone(game.ball)
        self.assertEqual(game.scores, [0, 0])

    def test_initial_ball_state(self):
        """Ball should start at center with velocity."""
        game = Game()
        center = game.field.center
        self.assertEqual(game.ball.x, center[0])
        self.assertEqual(game.ball.y, center[1])
        self.assertFalse(game.ball.in_flag)
        self.assertNotEqual(game.ball.vx, 0)  # Should have velocity

    def test_initial_player_positions(self):
        """Players should start on opposite sides."""
        game = Game()
        center_x = game.config.field_width / 2
        self.assertLess(game.player_a.x, center_x)
        self.assertGreater(game.player_b.x, center_x)

    def test_custom_config(self):
        """Game should accept custom config."""
        config = Config(field_width=1000, ball_speed=10.0)
        game = Game(config)
        self.assertEqual(game.config.field_width, 1000)
        self.assertEqual(game.config.ball_speed, 10.0)


class TestGameStep(unittest.TestCase):
    """Test Game step mechanics."""

    def setUp(self):
        self.game = Game()

    def test_step_increments_counters(self):
        """Step should increment step counters."""
        old_steps = self.game.steps_this_point
        old_total = self.game.total_steps
        self.game.step((16, 0), (16, 0))  # Stay actions
        self.assertEqual(self.game.steps_this_point, old_steps + 1)
        self.assertEqual(self.game.total_steps, old_total + 1)

    def test_step_moves_players(self):
        """Step should move players according to actions."""
        old_x_a = self.game.player_a.x
        self.game.step((0, 0), (16, 0))  # Player A moves right
        self.assertGreater(self.game.player_a.x, old_x_a)

    def test_step_updates_ball(self):
        """Step should update ball position."""
        old_ball_x = self.game.ball.x
        self.game.step((16, 0), (16, 0))
        # Ball should have moved
        self.assertNotEqual(self.game.ball.x, old_ball_x)

    def test_step_returns_result(self):
        """Step should return StepResult."""
        result = self.game.step((16, 0), (16, 0))
        self.assertIsInstance(result, StepResult)
        self.assertIsInstance(result.rewards, tuple)
        self.assertIsInstance(result.done, bool)
        self.assertIsInstance(result.hit_occurred, tuple)

    def test_game_over_no_action(self):
        """Step should do nothing when game is over."""
        self.game.state = GameState.GAME_OVER
        result = self.game.step((0, 0), (0, 0))
        self.assertEqual(result.rewards, (0.0, 0.0))
        self.assertTrue(result.done)


class TestGameHitting(unittest.TestCase):
    """Test ball hitting mechanics."""

    def setUp(self):
        self.game = Game()

    def test_hit_when_in_range_and_in_flag(self):
        """Player should hit when in range and in_flag is ON."""
        # Position ball near player A with in_flag ON
        self.game.ball.x = self.game.player_a.x + 10
        self.game.ball.y = self.game.player_a.y
        self.game.ball.in_flag = True
        self.game.ball.vx = 0
        self.game.ball.vy = 0

        result = self.game.step((16, 45), (16, 0))  # Player A stays, hits at 45 deg

        self.assertTrue(result.hit_occurred[0])  # Player A hit
        self.assertFalse(result.hit_occurred[1])  # Player B didn't hit

    def test_no_hit_when_in_flag_off(self):
        """Player should NOT hit when in_flag is OFF."""
        # Position ball near player A but in_flag OFF
        self.game.ball.x = self.game.player_a.x + 10
        self.game.ball.y = self.game.player_a.y
        self.game.ball.in_flag = False
        self.game.ball.vx = 0
        self.game.ball.vy = 0

        result = self.game.step((16, 45), (16, 0))

        self.assertFalse(result.hit_occurred[0])
        self.assertFalse(result.hit_occurred[1])

    def test_no_hit_when_out_of_range(self):
        """Player should NOT hit when ball is out of range."""
        # Position ball far from both players
        self.game.ball.x = self.game.config.field_width / 2
        self.game.ball.y = self.game.config.field_height / 2
        self.game.ball.in_flag = True
        self.game.ball.vx = 0
        self.game.ball.vy = 0

        result = self.game.step((16, 45), (16, 45))

        self.assertFalse(result.hit_occurred[0])
        self.assertFalse(result.hit_occurred[1])

    def test_hit_resets_in_flag(self):
        """Hitting ball should reset in_flag to OFF."""
        self.game.ball.x = self.game.player_a.x + 10
        self.game.ball.y = self.game.player_a.y
        self.game.ball.in_flag = True

        self.game.step((16, 45), (16, 0))

        self.assertFalse(self.game.ball.in_flag)

    def test_hit_gives_rally_reward(self):
        """Hitting should give rally reward (configured value)."""
        self.game.ball.x = self.game.player_a.x + 10
        self.game.ball.y = self.game.player_a.y
        self.game.ball.in_flag = True

        result = self.game.step((16, 45), (16, 0))

        # Rally reward is configured in config (default is 0 for sparse rewards)
        self.assertEqual(result.rewards[0], self.game.config.reward_rally)

    def test_rally_count_increments(self):
        """Rally count should increment on hit."""
        old_rally = self.game.rally_count
        self.game.ball.x = self.game.player_a.x + 10
        self.game.ball.y = self.game.player_a.y
        self.game.ball.in_flag = True

        self.game.step((16, 45), (16, 0))

        self.assertEqual(self.game.rally_count, old_rally + 1)


class TestGamePointScoring(unittest.TestCase):
    """Test point scoring mechanics."""

    def setUp(self):
        self.config = Config()
        self.game = Game(self.config)

    def test_point_win_with_in_flag_on(self):
        """Player should win point when ball hits wall with in_flag ON."""
        # Position ball near wall, going toward it, with in_flag ON
        self.game.ball.x = self.config.field_width - 10
        self.game.ball.y = self.config.field_height / 2
        self.game.ball.vx = 20  # Fast toward right wall
        self.game.ball.vy = 0
        self.game.ball.in_flag = True
        self.game.ball.last_hit_by = 0  # Player A hit it

        old_score_a = self.game.scores[0]
        result = self.game.step((16, 0), (16, 0))

        self.assertIsNotNone(result.point_result)
        self.assertEqual(result.point_result.winner, 0)  # Player A wins
        self.assertEqual(result.point_result.reason, "in")
        self.assertEqual(self.game.scores[0], old_score_a + 1)

    def test_point_loss_with_in_flag_off(self):
        """Player should lose point when ball hits wall with in_flag OFF."""
        # Position ball near wall, going toward it, with in_flag OFF (out shot)
        self.game.ball.x = self.config.field_width - 10
        self.game.ball.y = self.config.field_height / 2
        self.game.ball.vx = 20
        self.game.ball.vy = 0
        self.game.ball.in_flag = False
        self.game.ball.last_hit_by = 0  # Player A hit it (but it was out)

        old_score_b = self.game.scores[1]
        result = self.game.step((16, 0), (16, 0))

        self.assertIsNotNone(result.point_result)
        self.assertEqual(result.point_result.winner, 1)  # Player B wins (A lost)
        self.assertEqual(result.point_result.reason, "out")
        self.assertEqual(self.game.scores[1], old_score_b + 1)

    def test_point_win_reward(self):
        """Winner should get positive reward."""
        self.game.ball.x = self.config.field_width - 10
        self.game.ball.y = self.config.field_height / 2
        self.game.ball.vx = 20
        self.game.ball.vy = 0
        self.game.ball.in_flag = True
        self.game.ball.last_hit_by = 0

        result = self.game.step((16, 0), (16, 0))

        self.assertEqual(result.rewards[0], self.config.reward_point_win)
        self.assertEqual(result.rewards[1], self.config.reward_point_lose)

    def test_point_loss_reward(self):
        """Loser should get negative reward."""
        self.game.ball.x = self.config.field_width - 10
        self.game.ball.y = self.config.field_height / 2
        self.game.ball.vx = 20
        self.game.ball.vy = 0
        self.game.ball.in_flag = False  # Out
        self.game.ball.last_hit_by = 0

        result = self.game.step((16, 0), (16, 0))

        self.assertEqual(result.rewards[0], self.config.reward_point_lose)  # A loses
        self.assertEqual(result.rewards[1], self.config.reward_point_win)  # B wins


class TestGameOver(unittest.TestCase):
    """Test game over conditions."""

    def setUp(self):
        self.config = Config()  # 1 point = 1 episode (always)
        self.game = Game(self.config)

    def test_game_over_after_one_point(self):
        """Game should be over after 1 point (1 point = 1 episode)."""
        # Force a point win for player A
        self.game.ball.x = self.config.field_width - 5
        self.game.ball.vx = 10
        self.game.ball.vy = 0
        self.game.ball.in_flag = True
        self.game.ball.last_hit_by = 0

        result = self.game.step((16, 0), (16, 0))

        self.assertTrue(result.done)
        self.assertEqual(self.game.state, GameState.GAME_OVER)
        self.assertEqual(self.game.winner, 0)
        self.assertEqual(self.game.scores[0], 1)

    def test_game_starts_fresh(self):
        """Game should start with 0-0 score."""
        self.assertEqual(self.game.scores[0], 0)
        self.assertEqual(self.game.scores[1], 0)
        self.assertFalse(self.game.is_game_over)


class TestGameReset(unittest.TestCase):
    """Test game reset."""

    def test_reset_clears_scores(self):
        """Reset should clear scores."""
        game = Game()
        game.scores = [5, 3]
        game.reset()
        self.assertEqual(game.scores, [0, 0])

    def test_reset_clears_steps(self):
        """Reset should clear step counters."""
        game = Game()
        game.total_steps = 100
        game.reset()
        self.assertEqual(game.total_steps, 0)

    def test_reset_starts_new_serve(self):
        """Reset should start a new serve."""
        game = Game()
        game.state = GameState.GAME_OVER
        game.reset()
        self.assertEqual(game.state, GameState.PLAYING)
        self.assertIsNotNone(game.ball)

    def test_reset_returns_observation(self):
        """Reset should return initial observation."""
        game = Game()
        obs = game.reset()
        self.assertIn("ball_x", obs)
        self.assertIn("player_a_x", obs)


class TestGameObservation(unittest.TestCase):
    """Test game observation."""

    def test_observation_has_all_keys(self):
        """Observation should have all required keys."""
        game = Game()
        obs = game.get_observation()

        required_keys = [
            "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_in_flag",
            "player_a_x", "player_a_y", "player_b_x", "player_b_y",
            "score_a", "score_b", "rally_count",
            "field_width", "field_height",
        ]
        for key in required_keys:
            self.assertIn(key, obs)

    def test_observation_values_correct(self):
        """Observation values should match game state."""
        game = Game()
        obs = game.get_observation()

        self.assertEqual(obs["ball_x"], game.ball.x)
        self.assertEqual(obs["ball_y"], game.ball.y)
        self.assertEqual(obs["player_a_x"], game.player_a.x)
        self.assertEqual(obs["score_a"], game.scores[0])
        self.assertEqual(obs["ball_in_flag"], game.ball.in_flag)


class TestServeLogic(unittest.TestCase):
    """Test serve and point restart logic."""

    def test_new_point_after_point_over(self):
        """New point should start after POINT_OVER state."""
        game = Game()
        game.state = GameState.POINT_OVER

        # Next step should start a new serve
        game.step((16, 0), (16, 0))

        self.assertEqual(game.state, GameState.PLAYING)
        self.assertFalse(game.ball.in_flag)

    def test_players_reset_on_new_point(self):
        """Players should return to start positions on new point."""
        game = Game()

        # Move players
        game.player_a.x = 50
        game.player_b.x = 750
        game.state = GameState.POINT_OVER

        # Start new point
        game.step((16, 0), (16, 0))

        # Players should be back at start positions
        start_a, start_b = game.field.get_player_start_positions()
        self.assertEqual(game.player_a.x, start_a[0])
        self.assertEqual(game.player_b.x, start_b[0])


if __name__ == "__main__":
    unittest.main()
