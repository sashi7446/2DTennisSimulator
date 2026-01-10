"""Tests for Player class."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from ball import Ball
from config import Config
from field import Field
from player import NUM_MOVEMENT_ACTIONS, Player, create_players, direction_to_angle


class TestDirectionToAngle(unittest.TestCase):
    """Test direction_to_angle function."""

    def test_direction_0_is_right(self):
        """Direction 0 should be 0 degrees (right)."""
        self.assertEqual(direction_to_angle(0), 0.0)

    def test_direction_4_is_down(self):
        """Direction 4 should be 90 degrees (down)."""
        self.assertEqual(direction_to_angle(4), 90.0)

    def test_direction_8_is_left(self):
        """Direction 8 should be 180 degrees (left)."""
        self.assertEqual(direction_to_angle(8), 180.0)

    def test_direction_12_is_up(self):
        """Direction 12 should be 270 degrees (up)."""
        self.assertEqual(direction_to_angle(12), 270.0)

    def test_direction_16_is_stay(self):
        """Direction 16 (stay) should return None."""
        self.assertIsNone(direction_to_angle(16))

    def test_all_directions_unique(self):
        """All 16 directions should have unique angles."""
        angles = [direction_to_angle(i) for i in range(16)]
        self.assertEqual(len(set(angles)), 16)

    def test_directions_evenly_spaced(self):
        """Directions should be 22.5 degrees apart."""
        for i in range(15):
            angle1 = direction_to_angle(i)
            angle2 = direction_to_angle(i + 1)
            self.assertAlmostEqual(angle2 - angle1, 22.5, places=5)


class TestPlayer(unittest.TestCase):
    """Test Player class."""

    def setUp(self):
        self.config = Config()
        self.field = Field(self.config)
        self.player = Player(
            player_id=0,
            x=100,
            y=100,
            speed=self.config.player_speed,
            radius=self.config.player_radius,
            reach_distance=self.config.reach_distance,
        )

    def test_initial_state(self):
        """Player should have correct initial state."""
        self.assertEqual(self.player.player_id, 0)
        self.assertEqual(self.player.x, 100)
        self.assertEqual(self.player.y, 100)

    def test_move_right(self):
        """Moving right (direction 0) should increase x."""
        old_x = self.player.x
        self.player.move(0, self.field)
        self.assertGreater(self.player.x, old_x)
        self.assertEqual(self.player.y, 100)  # y unchanged

    def test_move_down(self):
        """Moving down (direction 4) should increase y."""
        old_y = self.player.y
        self.player.move(4, self.field)
        self.assertGreater(self.player.y, old_y)
        self.assertEqual(self.player.x, 100)  # x unchanged

    def test_move_left(self):
        """Moving left (direction 8) should decrease x."""
        old_x = self.player.x
        self.player.move(8, self.field)
        self.assertLess(self.player.x, old_x)

    def test_move_up(self):
        """Moving up (direction 12) should decrease y."""
        old_y = self.player.y
        self.player.move(12, self.field)
        self.assertLess(self.player.y, old_y)

    def test_move_stay(self):
        """Stay (direction 16) should not change position."""
        old_x, old_y = self.player.x, self.player.y
        self.player.move(16, self.field)
        self.assertEqual(self.player.x, old_x)
        self.assertEqual(self.player.y, old_y)

    def test_move_distance(self):
        """Move distance should match player speed."""
        old_x = self.player.x
        self.player.move(0, self.field)  # Move right
        distance = self.player.x - old_x
        self.assertAlmostEqual(distance, self.config.player_speed, places=5)

    def test_move_clamped_to_field(self):
        """Player should not move outside field walls."""
        # Move toward left wall
        self.player.x = self.player.radius + 1
        self.player.move(8, self.field)  # Move left
        self.assertGreaterEqual(self.player.x, self.player.radius)

        # Move toward right wall
        self.player.x = self.config.field_width - self.player.radius - 1
        self.player.move(0, self.field)  # Move right
        self.assertLessEqual(self.player.x, self.config.field_width - self.player.radius)

    def test_can_hit_requires_is_in(self):
        """can_hit should return False if is_in is OFF."""
        ball = Ball(x=100, y=100)  # Same position as player
        ball.is_in = False
        self.assertFalse(self.player.can_hit(ball))

    def test_can_hit_requires_distance(self):
        """can_hit should return False if ball is too far."""
        ball = Ball(x=200, y=200)  # Far from player
        ball.is_in = True
        self.assertFalse(self.player.can_hit(ball))

    def test_can_hit_success(self):
        """can_hit should return True when conditions met."""
        ball = Ball(x=100 + self.config.reach_distance - 1, y=100)
        ball.is_in = True
        self.assertTrue(self.player.can_hit(ball))

    def test_can_hit_at_exact_reach(self):
        """can_hit should return True at exactly reach distance."""
        ball = Ball(x=100 + self.config.reach_distance, y=100)
        ball.is_in = True
        self.assertTrue(self.player.can_hit(ball))

    def test_can_hit_just_beyond_reach(self):
        """can_hit should return False just beyond reach distance."""
        ball = Ball(x=100 + self.config.reach_distance + 0.1, y=100)
        ball.is_in = True
        self.assertFalse(self.player.can_hit(ball))

    def test_hit_ball_success(self):
        """hit_ball should succeed and modify ball state."""
        ball = Ball(x=110, y=100)
        ball.is_in = True

        result = self.player.hit_ball(ball, 45.0, 5.0)

        self.assertTrue(result)
        self.assertFalse(ball.is_in)  # Should be reset
        self.assertEqual(ball.last_hit_by, 0)

    def test_hit_ball_failure_no_is_in(self):
        """hit_ball should fail if is_in is OFF."""
        ball = Ball(x=110, y=100)
        ball.is_in = False

        result = self.player.hit_ball(ball, 45.0, 5.0)

        self.assertFalse(result)
        self.assertIsNone(ball.last_hit_by)  # Should not change

    def test_hit_ball_failure_too_far(self):
        """hit_ball should fail if ball is too far."""
        ball = Ball(x=500, y=500)
        ball.is_in = True

        result = self.player.hit_ball(ball, 45.0, 5.0)

        self.assertFalse(result)

    def test_distance_to_ball(self):
        """distance_to_ball should calculate correct distance."""
        ball = Ball(x=103, y=104)
        self.assertAlmostEqual(self.player.distance_to_ball(ball), 5.0, places=5)

    def test_position_property(self):
        """position property should return tuple."""
        self.assertEqual(self.player.position, (100, 100))

    def test_reset(self):
        """reset should move player to new position."""
        self.player.reset(200, 150)
        self.assertEqual(self.player.x, 200)
        self.assertEqual(self.player.y, 150)


class TestCreatePlayers(unittest.TestCase):
    """Test create_players function."""

    def setUp(self):
        self.config = Config()
        self.field = Field(self.config)

    def test_creates_two_players(self):
        """Should create two distinct players."""
        player_a, player_b = create_players(self.field, self.config)
        self.assertEqual(player_a.player_id, 0)
        self.assertEqual(player_b.player_id, 1)

    def test_players_on_opposite_sides(self):
        """Players should start on opposite sides."""
        player_a, player_b = create_players(self.field, self.config)
        center_x = self.config.field_width / 2
        self.assertLess(player_a.x, center_x)
        self.assertGreater(player_b.x, center_x)

    def test_players_have_correct_properties(self):
        """Players should have config properties."""
        player_a, player_b = create_players(self.field, self.config)

        self.assertEqual(player_a.speed, self.config.player_speed)
        self.assertEqual(player_a.radius, self.config.player_radius)
        self.assertEqual(player_a.reach_distance, self.config.reach_distance)

        self.assertEqual(player_b.speed, self.config.player_speed)
        self.assertEqual(player_b.radius, self.config.player_radius)
        self.assertEqual(player_b.reach_distance, self.config.reach_distance)


class TestNumMovementActions(unittest.TestCase):
    """Test NUM_MOVEMENT_ACTIONS constant."""

    def test_num_actions(self):
        """Should have 17 actions (16 directions + stay)."""
        self.assertEqual(NUM_MOVEMENT_ACTIONS, 17)


if __name__ == "__main__":
    unittest.main()
