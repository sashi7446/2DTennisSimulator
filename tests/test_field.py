"""Tests for Field class."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from config import Config
from field import Field, Rectangle


class TestRectangle(unittest.TestCase):
    """Test Rectangle class."""

    def test_contains_point_inside(self):
        """Point inside rectangle should return True."""
        rect = Rectangle(10, 10, 100, 50)
        # Center
        self.assertTrue(rect.contains_point(60, 35))
        # Corners (on boundary)
        self.assertTrue(rect.contains_point(10, 10))  # Top-left
        self.assertTrue(rect.contains_point(110, 60))  # Bottom-right

    def test_contains_point_outside(self):
        """Point outside rectangle should return False."""
        rect = Rectangle(10, 10, 100, 50)
        self.assertFalse(rect.contains_point(0, 0))  # Before
        self.assertFalse(rect.contains_point(200, 200))  # After
        self.assertFalse(rect.contains_point(60, 0))  # Above
        self.assertFalse(rect.contains_point(60, 100))  # Below

    def test_center(self):
        """Center calculation should be correct."""
        rect = Rectangle(0, 0, 100, 50)
        self.assertEqual(rect.center, (50, 25))

        rect2 = Rectangle(10, 20, 100, 50)
        self.assertEqual(rect2.center, (60, 45))

    def test_bounds(self):
        """Bounds should return (left, top, right, bottom)."""
        rect = Rectangle(10, 20, 100, 50)
        self.assertEqual(rect.bounds, (10, 20, 110, 70))


class TestField(unittest.TestCase):
    """Test Field class."""

    def setUp(self):
        self.config = Config()
        self.field = Field(self.config)

    def test_field_center(self):
        """Field center should be correct."""
        expected_center = (self.config.field_width / 2, self.config.field_height / 2)
        self.assertEqual(self.field.center, expected_center)

    def test_area_positions(self):
        """In-areas should be positioned correctly with gap between them."""
        # Area A should be to the left of center
        self.assertLess(self.field.area_a.x + self.field.area_a.width,
                        self.config.field_width / 2)
        # Area B should be to the right of center
        self.assertGreater(self.field.area_b.x, self.config.field_width / 2)

        # Gap between areas should match config
        gap = self.field.area_b.x - (self.field.area_a.x + self.field.area_a.width)
        self.assertEqual(gap, self.config.area_gap)

    def test_area_dimensions(self):
        """In-areas should have correct dimensions."""
        self.assertEqual(self.field.area_a.width, self.config.area_width)
        self.assertEqual(self.field.area_a.height, self.config.area_height)
        self.assertEqual(self.field.area_b.width, self.config.area_width)
        self.assertEqual(self.field.area_b.height, self.config.area_height)

    def test_is_in_area(self):
        """is_in_area should detect points in either area."""
        # Center of Area A
        center_a = self.field.area_a.center
        self.assertTrue(self.field.is_in_area(*center_a))
        self.assertTrue(self.field.is_in_area_a(*center_a))
        self.assertFalse(self.field.is_in_area_b(*center_a))

        # Center of Area B
        center_b = self.field.area_b.center
        self.assertTrue(self.field.is_in_area(*center_b))
        self.assertFalse(self.field.is_in_area_a(*center_b))
        self.assertTrue(self.field.is_in_area_b(*center_b))

        # In the gap (should not be in any area)
        gap_x = self.config.field_width / 2
        gap_y = self.config.field_height / 2
        self.assertFalse(self.field.is_in_area(gap_x, gap_y))

    def test_wall_collision_left(self):
        """Detect collision with left wall."""
        self.assertEqual(self.field.check_wall_collision(0, 100), "left")
        self.assertEqual(self.field.check_wall_collision(5, 100, radius=5), "left")
        self.assertIsNone(self.field.check_wall_collision(10, 100))

    def test_wall_collision_right(self):
        """Detect collision with right wall."""
        w = self.config.field_width
        self.assertEqual(self.field.check_wall_collision(w, 100), "right")
        self.assertEqual(self.field.check_wall_collision(w - 5, 100, radius=5), "right")
        self.assertIsNone(self.field.check_wall_collision(w - 10, 100))

    def test_wall_collision_top(self):
        """Detect collision with top wall."""
        self.assertEqual(self.field.check_wall_collision(100, 0), "top")
        self.assertEqual(self.field.check_wall_collision(100, 5, radius=5), "top")
        self.assertIsNone(self.field.check_wall_collision(100, 10))

    def test_wall_collision_bottom(self):
        """Detect collision with bottom wall."""
        h = self.config.field_height
        self.assertEqual(self.field.check_wall_collision(100, h), "bottom")
        self.assertEqual(self.field.check_wall_collision(100, h - 5, radius=5), "bottom")
        self.assertIsNone(self.field.check_wall_collision(100, h - 10))

    def test_clamp_position(self):
        """Clamp should keep position within walls."""
        # Inside - no change
        self.assertEqual(self.field.clamp_position(100, 100), (100, 100))

        # Outside left
        x, y = self.field.clamp_position(-10, 100)
        self.assertGreaterEqual(x, 0)

        # Outside right
        x, y = self.field.clamp_position(1000, 100)
        self.assertLessEqual(x, self.config.field_width)

        # With radius
        x, y = self.field.clamp_position(0, 100, radius=10)
        self.assertEqual(x, 10)

    def test_player_start_positions(self):
        """Player start positions should be on opposite sides."""
        pos_a, pos_b = self.field.get_player_start_positions()

        # Player A should be on the left
        self.assertLess(pos_a[0], self.config.field_width / 2)

        # Player B should be on the right
        self.assertGreater(pos_b[0], self.config.field_width / 2)

        # Both should be vertically centered
        self.assertEqual(pos_a[1], self.config.field_height / 2)
        self.assertEqual(pos_b[1], self.config.field_height / 2)


if __name__ == "__main__":
    unittest.main()
