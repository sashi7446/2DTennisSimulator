#!/usr/bin/env python3
"""Record every agent pairing once, for the matchup grid in docs/index.html.

The point of this file is a survey, not a tournament: ten points per card is
enough to see what a matchup feels like, and 28 cards have to stay light
enough that a phone can flick through them.

Cards already on disk are kept as they are, so re-running this after adding
one agent records only the new cards.

    python record_matrix.py                      # every pairing, reusing what exists
    python record_matrix.py --agents chase smart # just these
    python record_matrix.py --force              # record everything again
"""

import argparse
import hashlib
import itertools
import json
import os
import time
from typing import Any, Dict, List

from config import Config
from record_replay import record, stamp_viewer

# Rule-based agents only: neural and transformer start untrained, so their
# cards would show two players wandering off and tell you nothing.
DEFAULT_AGENTS = ["chase", "smart", "baseliner", "positional", "intercept", "solver", "random"]

OUTPUT_DIR = "docs/replays"
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.js")
VIEWER_PAGE = "docs/index.html"

# Enough points to see which agent actually has the upper hand, while the
# per-card file stays small enough to load on a tap.
DEFAULT_POINTS = 10

# Half the single-replay budget. A stalled card is worth seeing for a few
# seconds; it is not worth 60 KB of endless rally on a phone connection.
MAX_STEPS = 900


def card_id(agent_a: str, agent_b: str) -> str:
    return f"{agent_a}__{agent_b}"


def pairings(agents: List[str]) -> List[tuple]:
    """Every unordered pairing, mirror matches included."""
    return [(a, b) for a, b in itertools.combinations_with_replacement(agents, 2)]


def write_card(path: str, replay: Dict[str, Any]) -> str:
    """Write one card and return its content hash, used for cache busting."""
    payload = json.dumps(replay, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.REPLAY_CARD = ")
        f.write(payload)
        f.write(";\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def load_index() -> Dict[str, Any]:
    """Read the existing index so untouched cards keep their metadata."""
    if not os.path.exists(INDEX_PATH):
        return {}

    with open(INDEX_PATH, encoding="utf-8") as f:
        text = f.read()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        index = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return {card["id"]: card for card in index.get("cards", [])}


def write_index(agents: List[str], cards: List[Dict[str, Any]]) -> None:
    index = {
        # Bump when the frame layout changes, so the viewer can refuse to
        # draw cards it would misread rather than drawing them wrong.
        "version": 1,
        "agents": agents,
        "cards": cards,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write("window.REPLAY_INDEX = ")
        json.dump(index, f, separators=(",", ":"))
        f.write(";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record every agent pairing for the grid.")
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS, help="Agents to pair up")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="Points per card")
    parser.add_argument("--force", action="store_true", help="Re-record cards that already exist")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Frames per point")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    known = load_index()
    config = Config()

    cards: List[Dict[str, Any]] = []
    recorded = reused = 0

    for agent_a, agent_b in pairings(args.agents):
        cid = card_id(agent_a, agent_b)
        filename = f"{cid}.js"
        path = os.path.join(OUTPUT_DIR, filename)

        if not args.force and cid in known and os.path.exists(path):
            cards.append(known[cid])
            reused += 1
            print(f"{cid}: reusing the existing recording")
            continue

        print(f"{cid}: recording {args.points} point(s)")
        replay = record(agent_a, agent_b, args.points, config, args.max_steps)
        version = write_card(path, replay)
        recorded += 1

        cards.append(
            {
                "id": cid,
                "a": agent_a,
                "b": agent_b,
                "file": filename,
                "v": version,
                "names": replay["names"],
                "points": len(replay["points"]),
                "kb": max(1, os.path.getsize(path) // 1024),
                "stalled": sum(1 for p in replay["points"] if p["w"] < 0),
                "recorded": time.strftime("%Y-%m-%d"),
            }
        )

    write_index(args.agents, cards)
    if stamp_viewer(INDEX_PATH, VIEWER_PAGE, src="replays/index.js"):
        print(f"Updated {VIEWER_PAGE} to load the new index URL")

    total_kb = sum(card["kb"] for card in cards)
    print(f"\n{len(cards)} cards ({recorded} recorded, {reused} reused), {total_kb} KB total")

    stalled = [card["id"] for card in cards if card.get("stalled")]
    if stalled:
        print(f"Cards with a point that never ended: {', '.join(stalled)}")


if __name__ == "__main__":
    main()
