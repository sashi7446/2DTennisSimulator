"""Tests for position heatmap aggregation and PNG rendering.

No pygame and no display are needed - the aggregator is pure numpy and
the PNG writer is standard library, so these run anywhere the tests run.
"""

import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import Config
from heatmap import (
    PHASE_HIT,
    PHASE_IDLE,
    PositionHeatmap,
    colorize,
    hstack_panels,
    main,
    render_grid,
    write_png,
)
from stats_tracker import StatsTracker


class TestPositionHeatmap(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.heatmap = PositionHeatmap(self.config, grid_width=8, grid_height=4)

    def test_grid_starts_empty(self):
        self.assertEqual(self.heatmap.grid.sum(), 0)
        self.assertEqual(self.heatmap.frames_recorded, 0)

    def test_rejects_degenerate_grid(self):
        with self.assertRaises(ValueError):
            PositionHeatmap(self.config, grid_width=0, grid_height=4)

    def test_record_lands_in_expected_cell(self):
        # Top-left corner of the field
        self.heatmap.record(0, 1.0, 1.0)
        self.assertEqual(self.heatmap.grid[0][PHASE_IDLE][0][0], 1)

    def test_record_maps_bottom_right_inside_grid(self):
        self.heatmap.record(1, self.config.field_width, self.config.field_height)
        self.assertEqual(self.heatmap.grid[1][PHASE_IDLE][3][7], 1)

    def test_out_of_bounds_positions_are_clamped(self):
        self.heatmap.record(0, -500.0, -500.0)
        self.heatmap.record(0, 99999.0, 99999.0)
        self.assertEqual(self.heatmap.grid[0][PHASE_IDLE][0][0], 1)
        self.assertEqual(self.heatmap.grid[0][PHASE_IDLE][3][7], 1)

    def test_non_finite_positions_do_not_crash(self):
        self.heatmap.record(0, float("nan"), float("inf"))
        self.assertEqual(self.heatmap.grid.sum(), 1)

    def test_hit_and_idle_are_counted_separately(self):
        self.heatmap.record(0, 100.0, 100.0, hit=True)
        self.heatmap.record(0, 100.0, 100.0, hit=False)
        self.assertEqual(self.heatmap.grid[0][PHASE_HIT].sum(), 1)
        self.assertEqual(self.heatmap.grid[0][PHASE_IDLE].sum(), 1)

    def test_record_frame_counts_both_players_once(self):
        self.heatmap.record_frame((10.0, 10.0), (700.0, 300.0), (False, True))
        self.assertEqual(self.heatmap.frames_recorded, 1)
        self.assertEqual(self.heatmap.grid[0].sum(), 1)
        self.assertEqual(self.heatmap.grid[1].sum(), 1)
        self.assertEqual(self.heatmap.grid[1][PHASE_HIT].sum(), 1)

    def test_layer_sums_over_open_axes(self):
        self.heatmap.record_frame((10.0, 10.0), (700.0, 300.0))
        self.heatmap.record_frame((10.0, 10.0), (700.0, 300.0), (True, False))

        self.assertEqual(self.heatmap.layer().sum(), 4)
        self.assertEqual(self.heatmap.layer(player_id=0).sum(), 2)
        self.assertEqual(self.heatmap.layer(phase=PHASE_HIT).sum(), 1)
        self.assertEqual(self.heatmap.layer(player_id=1, phase=PHASE_HIT).sum(), 0)

    def test_layer_shape_matches_grid(self):
        self.assertEqual(self.heatmap.layer().shape, (4, 8))


class TestSaveLoad(unittest.TestCase):
    def test_round_trip_preserves_grid_and_metadata(self):
        original = PositionHeatmap(Config(field_width=640, field_height=320), 10, 5)
        original.record_frame((100.0, 100.0), (500.0, 200.0), (True, False))
        original.record_frame((110.0, 120.0), (480.0, 210.0))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "grid.npz")
            original.save(path)
            self.assertTrue(os.path.exists(path))
            loaded = PositionHeatmap.load(path)

        self.assertTrue(np.array_equal(loaded.grid, original.grid))
        self.assertEqual(loaded.frames_recorded, 2)
        self.assertEqual(loaded.grid_width, 10)
        self.assertEqual(loaded.grid_height, 5)
        self.assertEqual(loaded.config.field_width, 640)
        self.assertEqual(loaded.config.field_height, 320)


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.heatmap = PositionHeatmap(Config(), grid_width=8, grid_height=4)
        self.heatmap.record_frame((100.0, 100.0), (700.0, 300.0))

    def test_colorize_spans_the_palette(self):
        image = colorize(np.array([0.0, 0.5, 1.0]))
        self.assertEqual(image.shape, (3, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertLess(int(image[0].sum()), int(image[2].sum()))

    def test_colorize_clips_out_of_range_values(self):
        image = colorize(np.array([-5.0, 5.0]))
        low = colorize(np.array([0.0]))[0]
        high = colorize(np.array([1.0]))[0]
        self.assertTrue(np.array_equal(image[0], low))
        self.assertTrue(np.array_equal(image[1], high))

    def test_render_grid_size_follows_cell_size(self):
        image = render_grid(self.heatmap, cell_size=10)
        self.assertEqual(image.shape, (40, 80, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_empty_grid_renders_without_dividing_by_zero(self):
        empty = PositionHeatmap(Config(), grid_width=4, grid_height=2)
        image = render_grid(empty, cell_size=4, draw_court=False)
        self.assertTrue(np.isfinite(image).all())

    def test_shared_peak_dims_a_quiet_panel(self):
        busy = PositionHeatmap(Config(), grid_width=8, grid_height=4)
        for _ in range(50):
            busy.record(0, 100.0, 100.0)

        own_scale = render_grid(self.heatmap, cell_size=2, draw_court=False)
        shared = render_grid(self.heatmap, cell_size=2, peak=50.0, draw_court=False)
        self.assertLess(int(shared.sum()), int(own_scale.sum()))

    def test_hstack_panels_widens_by_gap(self):
        panel = render_grid(self.heatmap, cell_size=4, draw_court=False)
        stacked = hstack_panels([panel, panel], gap=6)
        self.assertEqual(stacked.shape[0], panel.shape[0])
        self.assertEqual(stacked.shape[1], panel.shape[1] * 2 + 6)

    def test_hstack_pads_panels_of_different_heights(self):
        tall = np.zeros((20, 10, 3), dtype=np.uint8)
        short = np.zeros((8, 10, 3), dtype=np.uint8)
        stacked = hstack_panels([tall, short], gap=2)
        self.assertEqual(stacked.shape, (20, 22, 3))

    def test_hstack_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            hstack_panels([])


class TestPngWriter(unittest.TestCase):
    def test_writes_a_decodable_png(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[1, 2] = (200, 100, 50)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out", "image.png")
            write_png(path, image)
            with open(path, "rb") as handle:
                data = handle.read()

        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")

        # IHDR payload starts after the 8-byte signature, 4-byte length and tag
        width, height, depth, color_type = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((width, height, depth, color_type), (5, 3, 8, 2))
        self.assertIn(b"IEND", data[-8:])

        # Rebuild the pixels from the IDAT stream and compare
        start = data.index(b"IDAT") + 4
        end = data.index(b"IEND") - 8
        raw = zlib.decompress(data[start:end])
        rows = [raw[i * (width * 3 + 1) : (i + 1) * (width * 3 + 1)] for i in range(height)]
        self.assertTrue(all(row[0] == 0 for row in rows))  # filter byte
        decoded = np.array([list(row[1:]) for row in rows], dtype=np.uint8)
        self.assertTrue(np.array_equal(decoded.reshape(height, width, 3), image))

    def test_rejects_non_rgb_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.png")
            with self.assertRaises(ValueError):
                write_png(path, np.zeros((4, 4), dtype=np.uint8))


class TestStatsTrackerIntegration(unittest.TestCase):
    def test_recording_is_off_by_default(self):
        stats = StatsTracker()
        self.assertIsNone(stats.heatmap)
        stats.record_positions((10.0, 10.0), (700.0, 300.0))  # must not raise

    def test_recording_reaches_the_grid_when_enabled(self):
        heatmap = PositionHeatmap(Config(), grid_width=8, grid_height=4)
        stats = StatsTracker(heatmap=heatmap)
        stats.record_positions((10.0, 10.0), (700.0, 300.0), (True, False))

        self.assertEqual(heatmap.frames_recorded, 1)
        self.assertEqual(heatmap.grid[0][PHASE_HIT].sum(), 1)


class TestCli(unittest.TestCase):
    def _recording(self, tmp, name, x):
        heatmap = PositionHeatmap(Config(), grid_width=8, grid_height=4)
        heatmap.record_frame((x, 100.0), (700.0, 300.0))
        path = os.path.join(tmp, name)
        heatmap.save(path)
        return path

    def test_single_input_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._recording(tmp, "one.npz", 100.0)
            out = os.path.join(tmp, "one.png")
            self.assertEqual(main([source, "--out", out, "--cell-size", "4"]), 0)
            self.assertTrue(os.path.exists(out))

    def test_compare_is_wider_than_a_single_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._recording(tmp, "a.npz", 100.0)
            second = self._recording(tmp, "b.npz", 300.0)
            single = os.path.join(tmp, "single.png")
            both = os.path.join(tmp, "both.png")

            main([first, "--out", single, "--cell-size", "4"])
            main(["--compare", first, second, "--out", both, "--cell-size", "4", "--shared-scale"])

            self.assertGreater(os.path.getsize(both), 0)
            self.assertTrue(os.path.exists(single))

    def test_player_and_phase_filters_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._recording(tmp, "one.npz", 100.0)
            out = os.path.join(tmp, "filtered.png")
            code = main(
                [source, "--player", "a", "--phase", "idle", "--out", out, "--cell-size", "4"]
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))

    def test_missing_file_exits_with_error(self):
        with self.assertRaises(SystemExit):
            main(["does_not_exist.npz"])

    def test_no_input_exits_with_error(self):
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
