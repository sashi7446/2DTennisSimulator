#!/usr/bin/env python3
"""Record matches into a replay file for the web viewer (docs/index.html).

The viewer is a static page, so the replay is written as a plain JS
assignment instead of JSON. That way it loads over file:// too, with no
local server and no fetch/CORS setup.

    python record_replay.py --agent-a chase --agent-b smart --points 20
"""

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List

from config import Config
from game import Game
from main import create_agent

DEFAULT_OUTPUT = "docs/replay.js"
VIEWER_PAGE = "docs/index.html"

# Frames kept per point. Two defensive agents may never end a point, and at
# the engine budget of 5000 that produced a 4 MB replay for a phone to load.
DEFAULT_MAX_STEPS = 1800


def record_point(game: Game, agent_a: Any, agent_b: Any, max_steps: int) -> Dict[str, Any]:
    """Play one point and return its recorded frames.

    Each frame is a flat list to keep the file small:
        [ball_x, ball_y, ball_is_in, a_x, a_y, b_x, b_y, hit]
    where hit is 0 (none), 1 (A hit) or 2 (B hit).
    """
    agent_a.reset()
    agent_b.reset()

    frames: List[List[float]] = []
    winner = -1
    reason = "timeout"
    rally = 0

    for _ in range(max_steps):
        obs = game.get_observation()
        result = game.step(agent_a.act(obs), agent_b.act(obs))

        if result.hit_occurred[0]:
            hit = 1
        elif result.hit_occurred[1]:
            hit = 2
        else:
            hit = 0

        ball = game.ball
        frames.append(
            [
                round(ball.x, 1) if ball else 0.0,
                round(ball.y, 1) if ball else 0.0,
                1 if ball and ball.is_in else 0,
                round(game.player_a.x, 1),
                round(game.player_a.y, 1),
                round(game.player_b.x, 1),
                round(game.player_b.y, 1),
                hit,
            ]
        )

        if result.point_result:
            winner = result.point_result.winner
            reason = result.point_result.reason
        rally = game.rally_count

        if game.is_game_over:
            break

    return {"w": winner, "reason": reason, "rally": rally, "f": frames}


def record(
    agent_a_type: str,
    agent_b_type: str,
    num_points: int,
    config: Config,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Record a series of points into a replay dict."""
    agent_a = create_agent(agent_a_type, 0, config)
    agent_b = create_agent(agent_b_type, 1, config)

    points = []
    for i in range(num_points):
        game = Game(config)
        point = record_point(game, agent_a, agent_b, max_steps)
        points.append(point)
        outcome = "AB"[point["w"]] + " wins" if point["w"] >= 0 else "stalled"
        print(
            f"Point {i + 1}/{num_points}: {outcome} "
            f"({point['reason']}, rally {point['rally']}, {len(point['f'])} frames)"
        )

    area_a, area_b = config.get_area_positions()
    return {
        "names": [agent_a.config.name, agent_b.config.name],
        "field": [config.field_width, config.field_height],
        "areas": [
            [area_a[0], area_a[1], config.area_width, config.area_height],
            [area_b[0], area_b[1], config.area_width, config.area_height],
        ],
        "radii": [config.ball_radius, config.player_radius, config.reach_distance],
        "meta": {
            "ball_speed": config.ball_speed,
            "player_speed": config.player_speed,
            "reach": config.reach_distance,
        },
        "fps": config.fps,
        "points": points,
    }


def stamp_viewer(replay_path: str, page_path: str = VIEWER_PAGE, src: str = "") -> bool:
    """Point the viewer at a versioned replay URL.

    GitHub Pages and phone browsers happily serve a cached replay.js after a
    new recording is published, which looks like the recording never ran.
    Hashing the file into the query string gives each recording its own URL.

    `src` is the script src as the page spells it, which is not always the
    file's basename: the matchup index sits in a subdirectory.

    Returns True when the page changed.
    """
    if not os.path.exists(page_path):
        return False

    with open(replay_path, "rb") as f:
        version = hashlib.sha256(f.read()).hexdigest()[:8]

    with open(page_path, encoding="utf-8") as f:
        page = f.read()

    src = src or os.path.basename(replay_path)
    updated = re.sub(
        rf'<script src="{re.escape(src)}(\?v=[0-9a-f]+)?"></script>',
        f'<script src="{src}?v={version}"></script>',
        page,
        count=1,
    )
    if updated == page:
        return False

    with open(page_path, "w", encoding="utf-8") as f:
        f.write(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a replay for the web viewer.")
    parser.add_argument("--agent-a", default="chase", help="Agent type for player A")
    parser.add_argument("--agent-b", default="smart", help="Agent type for player B")
    parser.add_argument("--points", type=int, default=20, help="Number of points to record")
    parser.add_argument("--speed", type=float, default=None, help="Ball speed override")
    parser.add_argument("--player-speed", type=float, default=None, help="Player speed override")
    parser.add_argument("--reach", type=float, default=None, help="Player reach distance override")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Frames to record per point (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output path ({DEFAULT_OUTPUT})")
    args = parser.parse_args()

    config = Config()
    if args.speed is not None:
        config.ball_speed = args.speed
    if args.player_speed is not None:
        config.player_speed = args.player_speed
    if args.reach is not None:
        config.reach_distance = args.reach

    replay = record(args.agent_a, args.agent_b, args.points, config, args.max_steps)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.REPLAY = ")
        json.dump(replay, f, separators=(",", ":"))
        f.write(";\n")

    if stamp_viewer(args.out):
        print(f"Updated {VIEWER_PAGE} to load the new replay URL")

    total_frames = sum(len(p["f"]) for p in replay["points"])
    size_kb = os.path.getsize(args.out) // 1024
    print(f"\nWrote {args.out}: {len(replay['points'])} points, {total_frames} frames, {size_kb} KB")

    stalled = sum(1 for p in replay["points"] if p["w"] < 0)
    if stalled:
        print(
            f"WARNING: {stalled}/{len(replay['points'])} points never ended. "
            "These agents cannot finish a point at these settings, so the replay "
            "is mostly an endless rally."
        )
    if size_kb > 1024:
        print(f"WARNING: {size_kb} KB is a heavy download for a phone.")


if __name__ == "__main__":
    main()
