"""Tests for the live heatmap overlay and its H-key toggle.

Rendering runs against SDL's dummy video driver, so these need no display.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config import Config  # noqa: E402
from game import Game  # noqa: E402
from heatmap import LIVE_COLOR_STOPS, PositionHeatmap, colorize  # noqa: E402
from stats_tracker import StatsTracker  # noqa: E402

try:
    import numpy as np
    import pygame

    from input_handler import InputHandler, InputState
    from renderer import GameRenderer, HeatmapOverlay

    PYGAME_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without pygame
    PYGAME_AVAILABLE = False


class TestHeatmapToggleState(unittest.TestCase):
    """The toggle is plain state, so it needs no display at all."""

    def setUp(self):
        if not PYGAME_AVAILABLE:
            self.skipTest("pygame not installed")
        self.state = InputState()

    def test_starts_off(self):
        self.assertFalse(self.state.show_heatmap)
        self.assertIsNone(self.state.heatmap_player)

    def test_cycles_both_a_b_then_off(self):
        expected = [(True, None), (True, 0), (True, 1), (False, None)]
        for show, player in expected:
            self.state.cycle_heatmap()
            self.assertEqual((self.state.show_heatmap, self.state.heatmap_player), (show, player))

    def test_h_is_bound_outside_debug_mode(self):
        handler = InputHandler(debug_mode=False)
        self.assertIn(pygame.K_h, handler._key_bindings)
        handler._key_bindings[pygame.K_h]()
        self.assertTrue(handler.state.show_heatmap)


class TestLivePalette(unittest.TestCase):
    def test_live_palette_avoids_the_dark_end(self):
        # Anything near-black would vanish into the green court
        darkest = colorize(np.array([0.0]), LIVE_COLOR_STOPS)[0]
        self.assertGreater(int(darkest.sum()), 200)

    def test_live_palette_differs_from_the_png_palette(self):
        live = colorize(np.array([0.0]), LIVE_COLOR_STOPS)[0]
        png = colorize(np.array([0.0]))[0]
        self.assertFalse(np.array_equal(live, png))


class TestHeatmapOverlay(unittest.TestCase):
    def setUp(self):
        if not PYGAME_AVAILABLE:
            self.skipTest("pygame not installed")
        self.config = Config()
        self.heatmap = PositionHeatmap(self.config, grid_width=8, grid_height=4)
        self.overlay = HeatmapOverlay()
        pygame.init()
        self.screen = pygame.Surface((self.config.field_width, self.config.field_height))

    def _identity(self, x, y):
        return (int(x), int(y))

    def test_empty_grid_draws_nothing(self):
        before = pygame.image.tostring(self.screen, "RGB")
        self.overlay.draw(self.screen, self.heatmap, None, self._identity)
        self.assertEqual(pygame.image.tostring(self.screen, "RGB"), before)

    def test_missing_heatmap_is_ignored(self):
        self.overlay.draw(self.screen, None, None, self._identity)  # must not raise

    def test_recorded_frames_change_the_screen(self):
        for _ in range(30):
            self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))
        before = pygame.image.tostring(self.screen, "RGB")
        self.overlay.draw(self.screen, self.heatmap, None, self._identity)
        self.assertNotEqual(pygame.image.tostring(self.screen, "RGB"), before)

    def test_surface_is_cached_between_close_frames(self):
        for _ in range(30):
            self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))
        first = self.overlay._get_surface(self.heatmap, None, 150)
        second = self.overlay._get_surface(self.heatmap, None, 150)
        self.assertIs(first, second)

    def test_surface_is_rebuilt_once_enough_new_data_arrives(self):
        for _ in range(30):
            self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))
        first = self.overlay._get_surface(self.heatmap, None, 150)
        for _ in range(HeatmapOverlay.REBUILD_INTERVAL + 1):
            self.heatmap.record_frame((400.0, 200.0), (400.0, 200.0))
        self.assertIsNot(self.overlay._get_surface(self.heatmap, None, 150), first)

    def test_switching_player_rebuilds_the_surface(self):
        for _ in range(30):
            self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))
        both = self.overlay._get_surface(self.heatmap, None, 150)
        only_a = self.overlay._get_surface(self.heatmap, 0, 150)
        self.assertIsNot(both, only_a)

    def test_surface_covers_the_court(self):
        self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))
        surface = self.overlay._get_surface(self.heatmap, None, 150)
        self.assertEqual(surface.get_size(), (self.config.field_width, self.config.field_height))


class TestRendererIntegration(unittest.TestCase):
    def setUp(self):
        if not PYGAME_AVAILABLE:
            self.skipTest("pygame not installed")
        self.config = Config()
        self.renderer = GameRenderer(self.config)
        self.game = Game(self.config)
        self.stats = StatsTracker(heatmap=PositionHeatmap(self.config))
        for _ in range(40):
            self.stats.record_positions((100.0, 100.0), (700.0, 300.0))

    def tearDown(self):
        if PYGAME_AVAILABLE:
            self.renderer.close()

    def test_render_without_input_state_leaves_the_tint_off(self):
        self.renderer.render(self.game, stats=self.stats)  # must not raise

    def test_toggle_changes_what_is_drawn(self):
        state = InputState()
        self.renderer.render(self.game, stats=self.stats, input_state=state)
        off = pygame.image.tostring(self.renderer.screen, "RGB")

        state.cycle_heatmap()
        self.renderer.render(self.game, stats=self.stats, input_state=state)
        on = pygame.image.tostring(self.renderer.screen, "RGB")

        self.assertNotEqual(off, on)

    def test_stats_without_a_grid_is_safe(self):
        state = InputState()
        state.cycle_heatmap()
        self.renderer.render(self.game, stats=StatsTracker(), input_state=state)


if __name__ == "__main__":
    unittest.main()
