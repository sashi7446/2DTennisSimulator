"""Position heatmaps for 2D Tennis Simulator.

Answers the question the README opens with: does "return to the home
position" emerge on its own?

PNGs are written with zlib rather than matplotlib on purpose - the same
call as audio.py synthesising sound instead of shipping audio files.
Reach for a plotting library here and the project gains a heavyweight
dependency for one picture.

Usage examples: python heatmap.py --help
"""

from __future__ import annotations

import argparse
import os
import struct
import zlib
from typing import Optional, Sequence, Tuple

import numpy as np

from config import Config

PHASE_IDLE = 0
PHASE_HIT = 1
NUM_PHASES = 2

DEFAULT_GRID_WIDTH = 40
DEFAULT_GRID_HEIGHT = 20


class PositionHeatmap:
    """Counts how many frames each player spent in each cell of the court.

    The grid is indexed ``[player][phase][row][column]``. Hit frames are
    kept apart from waiting frames because only the waiting positions are
    chosen by the agent - contact positions are forced by the ball, so
    merging them would blur the behaviour being looked for.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        grid_width: int = DEFAULT_GRID_WIDTH,
        grid_height: int = DEFAULT_GRID_HEIGHT,
    ):
        if grid_width < 1 or grid_height < 1:
            raise ValueError("grid dimensions must be positive")

        self.config = config or Config()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = np.zeros((2, NUM_PHASES, grid_height, grid_width), dtype=np.int64)
        self.frames_recorded = 0

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        """Field coordinates to (row, column), clamped to the grid."""
        if not np.isfinite(x) or not np.isfinite(y):
            return (0, 0)
        col = int(x / self.config.field_width * self.grid_width)
        row = int(y / self.config.field_height * self.grid_height)
        col = max(0, min(self.grid_width - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def record(self, player_id: int, x: float, y: float, hit: bool = False) -> None:
        """Add one frame of presence for a player."""
        row, col = self._cell(x, y)
        self.grid[player_id][PHASE_HIT if hit else PHASE_IDLE][row][col] += 1

    def record_frame(
        self,
        a_pos: Tuple[float, float],
        b_pos: Tuple[float, float],
        hits: Tuple[bool, bool] = (False, False),
    ) -> None:
        """Add one frame for both players."""
        self.record(0, a_pos[0], a_pos[1], hits[0])
        self.record(1, b_pos[0], b_pos[1], hits[1])
        self.frames_recorded += 1

    def layer(self, player_id: Optional[int] = None, phase: Optional[int] = None) -> np.ndarray:
        """A 2D count grid, summed over whichever axes were left open."""
        data = self.grid
        if player_id is not None:
            data = data[player_id : player_id + 1]
        if phase is not None:
            data = data[:, phase : phase + 1]
        return data.sum(axis=(0, 1))

    def save(self, path: str) -> None:
        """Write the grid to a .npz alongside the settings that produced it."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path,
            grid=self.grid,
            frames_recorded=np.int64(self.frames_recorded),
            field_width=np.int64(self.config.field_width),
            field_height=np.int64(self.config.field_height),
        )

    @classmethod
    def load(cls, path: str) -> PositionHeatmap:
        """Read back a grid written by :meth:`save`."""
        with np.load(path) as data:
            grid = data["grid"]
            config = Config(
                field_width=int(data["field_width"]),
                field_height=int(data["field_height"]),
            )
            heatmap = cls(config, grid_width=grid.shape[3], grid_height=grid.shape[2])
            heatmap.grid = grid.astype(np.int64)
            heatmap.frames_recorded = int(data["frames_recorded"])
        return heatmap


# --- Rendering ------------------------------------------------------------

_COLOR_STOPS: Sequence[Tuple[float, Tuple[int, int, int]]] = (
    (0.00, (12, 28, 16)),
    (0.20, (18, 66, 74)),
    (0.45, (32, 120, 96)),
    (0.70, (190, 180, 60)),
    (0.90, (240, 120, 40)),
    (1.00, (255, 240, 210)),
)


# Skips the dark end: this palette is blitted onto the green court, where
# anything reading as shadow disappears into the grass.
LIVE_COLOR_STOPS: Sequence[Tuple[float, Tuple[int, int, int]]] = (
    (0.00, (60, 110, 255)),
    (0.35, (0, 225, 205)),
    (0.70, (255, 225, 70)),
    (1.00, (255, 70, 40)),
)


def colorize(
    normalized: np.ndarray,
    color_stops: Optional[Sequence[Tuple[float, Tuple[int, int, int]]]] = None,
) -> np.ndarray:
    """Map values in 0..1 onto a palette. Returns HxWx3 uint8."""
    values = np.clip(normalized, 0.0, 1.0)
    palette = color_stops or _COLOR_STOPS
    stops = np.array([s[0] for s in palette], dtype=np.float64)
    colors = np.array([s[1] for s in palette], dtype=np.float64)
    rgb = np.stack(
        [np.interp(values, stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def render_grid(
    heatmap: PositionHeatmap,
    player_id: Optional[int] = None,
    phase: Optional[int] = None,
    cell_size: int = 20,
    peak: Optional[float] = None,
    draw_court: bool = True,
) -> np.ndarray:
    """Render one grid to an HxWx3 uint8 image.

    ``peak`` fixes the value that maps to the top of the palette. Pass the
    same peak to two renders to compare them honestly; leave it None and
    each image is normalised by its own maximum, which compares shapes
    rather than volumes.
    """
    counts = heatmap.layer(player_id, phase).astype(np.float64)
    top = peak if peak is not None else counts.max()
    normalized = counts / top if top > 0 else counts

    image = colorize(normalized)
    image = np.repeat(np.repeat(image, cell_size, axis=0), cell_size, axis=1)

    if draw_court:
        _draw_court_lines(image, heatmap.config)
    return image


def _draw_court_lines(image: np.ndarray, config: Config) -> None:
    """Outline the in-areas and the centre line, in place."""
    height, width = image.shape[0], image.shape[1]
    scale_x = width / config.field_width
    scale_y = height / config.field_height
    line = np.array([235, 235, 235], dtype=np.uint8)

    for area_x, area_y in config.get_area_positions():
        left = int(area_x * scale_x)
        right = int((area_x + config.area_width) * scale_x)
        top = int(area_y * scale_y)
        bottom = int((area_y + config.area_height) * scale_y)
        left, right = max(0, left), min(width - 1, right)
        top, bottom = max(0, top), min(height - 1, bottom)
        image[top, left:right] = line
        image[bottom, left:right] = line
        image[top:bottom, left] = line
        image[top:bottom, right] = line

    centre = min(width - 1, int(config.field_width / 2 * scale_x))
    image[:, centre] = line


def hstack_panels(panels: Sequence[np.ndarray], gap: int = 12) -> np.ndarray:
    """Place images side by side with a neutral divider between them."""
    if not panels:
        raise ValueError("no panels to stack")
    height = max(panel.shape[0] for panel in panels)
    divider = np.full((height, gap, 3), 40, dtype=np.uint8)

    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.full((height - panel.shape[0], panel.shape[1], 3), 40, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)

    out = padded[0]
    for panel in padded[1:]:
        out = np.hstack([out, divider, panel])
    return out


def write_png(path: str, image: np.ndarray) -> None:
    """Write an HxWx3 uint8 array as a PNG, standard library only."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB array")

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    height, width = image.shape[0], image.shape[1]
    data = image.astype(np.uint8)
    # Each scanline is prefixed with a filter byte; 0 means "no filter"
    raw = np.hstack(
        [np.zeros((height, 1), dtype=np.uint8), data.reshape(height, width * 3)]
    ).tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)


# --- CLI ------------------------------------------------------------------

_PHASE_CHOICES = {"all": None, "idle": PHASE_IDLE, "hit": PHASE_HIT}
_PLAYER_CHOICES = {"both": None, "a": 0, "b": 1}


def _describe(path: str, heatmap: PositionHeatmap) -> str:
    return (
        f"{os.path.basename(path)}: {heatmap.frames_recorded} frames, "
        f"{heatmap.grid_width}x{heatmap.grid_height} grid"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render recorded position heatmaps to a PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One recording
  python heatmap.py heatmaps/chase.npz --out docs/heatmap.png

  # Side by side, sharing one colour scale so volumes are comparable
  python heatmap.py --compare heatmaps/gen_001.npz heatmaps/gen_050.npz \\
      --out docs/growth.png --shared-scale

  # Only where player A waits between shots
  python heatmap.py heatmaps/chase.npz --player a --phase idle --out a_idle.png
""",
    )
    parser.add_argument("inputs", nargs="*", metavar="NPZ", help="Recorded .npz file(s)")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="NPZ",
        default=None,
        help="Render these recordings side by side in one image",
    )
    parser.add_argument("--out", default="heatmap.png", metavar="PATH", help="Output PNG path")
    parser.add_argument(
        "--player", choices=sorted(_PLAYER_CHOICES), default="both", help="Which player to render"
    )
    parser.add_argument(
        "--phase",
        choices=sorted(_PHASE_CHOICES),
        default="all",
        help="all, idle (waiting frames) or hit (frames a shot was struck)",
    )
    parser.add_argument("--cell-size", type=int, default=20, metavar="PX", help="Pixels per cell")
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help="Normalise every panel by the same peak instead of its own",
    )
    args = parser.parse_args(argv)

    paths = list(args.compare or args.inputs)
    if not paths:
        parser.error("give at least one .npz file (or use --compare)")

    heatmaps = []
    for path in paths:
        if not os.path.exists(path):
            parser.error(f"no such recording: {path}")
        heatmaps.append(PositionHeatmap.load(path))

    player = _PLAYER_CHOICES[args.player]
    phase = _PHASE_CHOICES[args.phase]

    peak: Optional[float] = None
    if args.shared_scale:
        peak = max(float(h.layer(player, phase).max()) for h in heatmaps)

    panels = [render_grid(h, player, phase, cell_size=args.cell_size, peak=peak) for h in heatmaps]
    write_png(args.out, hstack_panels(panels) if len(panels) > 1 else panels[0])

    print(f"Wrote {args.out}")
    for index, (path, heatmap) in enumerate(zip(paths, heatmaps)):
        position = f"panel {index + 1} (left to right)" if len(paths) > 1 else "single panel"
        print(f"  {position}: {_describe(path, heatmap)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
