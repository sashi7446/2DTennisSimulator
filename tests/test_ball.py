"""Tests for Ball class."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from ball import Ball, create_serve_ball
from config import Config
from field import Field


class TestBall(unittest.TestCase):
    """Test Ball class."""

    def setUp(self):
        self.config = Config()
        self.field = Field(self.config)

    def test_initial_state(self):
        """Ball should start with is_in OFF."""
        ball = Ball(x=100, y=100)
        self.assertFalse(ball.is_in)
        self.assertIsNone(ball.last_hit_by)
        self.assertEqual(ball.vx, 0)
        self.assertEqual(ball.vy, 0)

    def test_set_velocity_from_angle_right(self):
        """Angle 0 should move right."""
        ball = Ball(x=100, y=100)
        ball.set_velocity_from_angle(0, 5.0)
        self.assertAlmostEqual(ball.vx, 5.0, places=5)
        self.assertAlmostEqual(ball.vy, 0.0, places=5)

    def test_set_velocity_from_angle_down(self):
        """Angle 90 should move down."""
        ball = Ball(x=100, y=100)
        ball.set_velocity_from_angle(90, 5.0)
        self.assertAlmostEqual(ball.vx, 0.0, places=5)
        self.assertAlmostEqual(ball.vy, 5.0, places=5)

    def test_set_velocity_from_angle_left(self):
        """Angle 180 should move left."""
        ball = Ball(x=100, y=100)
        ball.set_velocity_from_angle(180, 5.0)
        self.assertAlmostEqual(ball.vx, -5.0, places=5)
        self.assertAlmostEqual(ball.vy, 0.0, places=5)

    def test_set_velocity_from_angle_up(self):
        """Angle 270 (or -90) should move up."""
        ball = Ball(x=100, y=100)
        ball.set_velocity_from_angle(270, 5.0)
        self.assertAlmostEqual(ball.vx, 0.0, places=5)
        self.assertAlmostEqual(ball.vy, -5.0, places=5)

    def test_get_angle(self):
        """get_angle should return correct direction."""
        ball = Ball(x=100, y=100)
        ball.vx = 5.0
        ball.vy = 0.0
        self.assertAlmostEqual(ball.get_angle(), 0.0, places=5)

        ball.vx = 0.0
        ball.vy = 5.0
        self.assertAlmostEqual(ball.get_angle(), 90.0, places=5)

    def test_get_speed(self):
        """get_speed should return correct magnitude."""
        ball = Ball(x=100, y=100)
        ball.vx = 3.0
        ball.vy = 4.0
        self.assertAlmostEqual(ball.get_speed(), 5.0, places=5)

    def test_update_position(self):
        """Ball should move according to velocity."""
        ball = Ball(x=100, y=100, vx=5.0, vy=-3.0)
        ball.update(self.field)
        self.assertEqual(ball.x, 105.0)
        self.assertEqual(ball.y, 97.0)

    def test_update_is_in_turns_on(self):
        """In-flag should turn ON when entering OPPONENT'S in-area."""
        # Player B (id=1) hit the ball, so it must pass through Area A
        center_a = self.field.area_a.center
        ball = Ball(x=center_a[0] - 10, y=center_a[1], vx=15.0, vy=0.0)
        ball.last_hit_by = 1  # Player B hit it
        self.assertFalse(ball.is_in)

        # Move into Area A (opponent's area for Player B)
        ball.update(self.field)
        self.assertTrue(ball.is_in)

    def test_update_is_in_requires_opponent_area(self):
        """In-flag should NOT turn ON when entering own in-area."""
        # Player A (id=0) hit the ball, so it must pass through Area B (not A)
        center_a = self.field.area_a.center
        ball = Ball(x=center_a[0] - 10, y=center_a[1], vx=15.0, vy=0.0)
        ball.last_hit_by = 0  # Player A hit it
        self.assertFalse(ball.is_in)

        # Move into Area A (own area for Player A - should NOT turn on)
        ball.update(self.field)
        self.assertFalse(ball.is_in)  # Should stay OFF

    def test_update_is_in_player_a_to_area_b(self):
        """Player A's shots must pass through Area B to turn is_in ON."""
        # Position ball just before Area B
        center_b = self.field.area_b.center
        ball = Ball(x=center_b[0] - 10, y=center_b[1], vx=15.0, vy=0.0)
        ball.last_hit_by = 0  # Player A hit it
        self.assertFalse(ball.is_in)

        # Move into Area B (opponent's area for Player A)
        ball.update(self.field)
        self.assertTrue(ball.is_in)

    def test_update_is_in_stays_on(self):
        """In-flag should stay ON after leaving in-area."""
        center_a = self.field.area_a.center
        ball = Ball(x=center_a[0], y=center_a[1], vx=50.0, vy=0.0)
        ball.is_in = True

        # Move out of area
        ball.update(self.field)
        self.assertTrue(ball.is_in)  # Should stay ON

    def test_hit_resets_is_in(self):
        """Hitting ball should reset is_in to OFF."""
        ball = Ball(x=100, y=100)
        ball.is_in = True

        ball.hit(player_id=0, angle_degrees=45, speed=5.0)

        self.assertFalse(ball.is_in)
        self.assertEqual(ball.last_hit_by, 0)

    def test_hit_sets_velocity(self):
        """Hitting ball should set new velocity."""
        ball = Ball(x=100, y=100)
        ball.hit(player_id=1, angle_degrees=0, speed=10.0)

        self.assertAlmostEqual(ball.vx, 10.0, places=5)
        self.assertAlmostEqual(ball.vy, 0.0, places=5)
        self.assertEqual(ball.last_hit_by, 1)

    def test_distance_to(self):
        """distance_to should calculate correct distance."""
        ball = Ball(x=0, y=0)
        self.assertAlmostEqual(ball.distance_to(3, 4), 5.0, places=5)
        self.assertAlmostEqual(ball.distance_to(0, 0), 0.0, places=5)

    def test_reset(self):
        """Reset should clear all state."""
        ball = Ball(x=100, y=100, vx=5.0, vy=5.0)
        ball.is_in = True
        ball.last_hit_by = 1

        ball.reset(200, 150)

        self.assertEqual(ball.x, 200)
        self.assertEqual(ball.y, 150)
        self.assertEqual(ball.vx, 0.0)
        self.assertEqual(ball.vy, 0.0)
        self.assertFalse(ball.is_in)
        self.assertIsNone(ball.last_hit_by)

    def test_wall_collision_detected(self):
        """Update should return wall name on collision."""
        # Near left wall
        ball = Ball(x=5, y=100, vx=-10, vy=0, radius=5)
        wall = ball.update(self.field)
        self.assertEqual(wall, "left")

    def test_no_wall_collision(self):
        """Update should return None when no collision."""
        ball = Ball(x=100, y=100, vx=5, vy=0)
        wall = ball.update(self.field)
        self.assertIsNone(wall)


class TestCreateServeBall(unittest.TestCase):
    """Test create_serve_ball function."""

    def setUp(self):
        self.config = Config()
        self.field = Field(self.config)

    def test_serve_starts_at_center(self):
        """Serve ball should start at field center."""
        ball = create_serve_ball(self.field, self.config, direction=1)
        center = self.field.center
        self.assertEqual(ball.x, center[0])
        self.assertEqual(ball.y, center[1])

    def test_serve_is_in_off(self):
        """Serve ball should have is_in OFF."""
        ball = create_serve_ball(self.field, self.config)
        self.assertFalse(ball.is_in)

    def test_serve_has_correct_speed(self):
        """Serve ball should have configured speed."""
        ball = create_serve_ball(self.field, self.config)
        self.assertAlmostEqual(ball.get_speed(), self.config.ball_speed, places=5)

    def test_serve_direction_right(self):
        """Direction 1 should send ball right."""
        ball = create_serve_ball(self.field, self.config, direction=1)
        self.assertGreater(ball.vx, 0)

    def test_serve_direction_left(self):
        """Direction -1 should send ball left."""
        ball = create_serve_ball(self.field, self.config, direction=-1)
        self.assertLess(ball.vx, 0)

    def test_serve_angle_within_range(self):
        """Serve angle should be within ±serve_angle_range."""
        for _ in range(100):  # Test multiple random serves
            ball = create_serve_ball(self.field, self.config, direction=1)
            angle = ball.get_angle()
            # For right-going ball, angle should be between -15 and +15
            self.assertGreaterEqual(angle, -self.config.serve_angle_range)
            self.assertLessEqual(angle, self.config.serve_angle_range)

    def test_serve_will_pass_through_in_area(self):
        """Serve should be aimed to pass through an in-area."""
        # This is a functional test - simulate ball until it hits wall
        # and verify it passed through an in-area
        for _ in range(20):
            ball = create_serve_ball(self.field, self.config)
            passed_in_area = False

            for _ in range(1000):  # Max steps
                wall = ball.update(self.field)
                if ball.is_in:
                    passed_in_area = True
                if wall:
                    break

            self.assertTrue(
                passed_in_area,
                f"Ball did not pass through in-area. "
                f"Start: {self.field.center}, "
                f"Velocity: ({ball.vx}, {ball.vy})",
            )


if __name__ == "__main__":
    unittest.main()
