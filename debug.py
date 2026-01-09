"""
Debug logging system for 2D Tennis Simulator.

Provides detailed logging of game events to help identify
what's working correctly and what's not.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


class EventType(Enum):
    """Types of debug events."""

    # Ball events
    BALL_CREATED = "ball_created"
    BALL_POSITION_UPDATE = "ball_position_update"
    BALL_IS_IN_ON = "ball_is_in_on"
    BALL_IS_IN_OFF = "ball_is_in_off"
    BALL_WALL_COLLISION = "ball_wall_collision"

    # Player events
    PLAYER_MOVE = "player_move"
    PLAYER_HIT_ATTEMPT = "player_hit_attempt"
    PLAYER_HIT_SUCCESS = "player_hit_success"
    PLAYER_HIT_FAIL = "player_hit_fail"

    # Game events
    GAME_START = "game_start"
    GAME_RESET = "game_reset"
    POINT_START = "point_start"
    POINT_END = "point_end"
    GAME_OVER = "game_over"

    # Validation events
    VALIDATION_ERROR = "validation_error"
    VALIDATION_WARNING = "validation_warning"


@dataclass
class DebugEvent:
    """A single debug event."""

    frame: int
    event_type: EventType
    data: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "type": self.event_type.value,
            "data": self.data,
            "message": self.message,
        }


class DebugLogger:
    """
    Logger for tracking game state and events.

    Use this to:
    - Track all state changes
    - Identify when/where things go wrong
    - Verify game logic is working correctly
    """

    def __init__(self, enabled: bool = True, max_events: int = 10000):
        self.enabled = enabled
        self.max_events = max_events
        self.events: List[DebugEvent] = []
        self.frame = 0
        self.print_live = False  # Print events as they happen

    def log(self, event_type: EventType, data: Dict[str, Any], message: str = ""):
        """Log a debug event."""
        if not self.enabled:
            return

        event = DebugEvent(
            frame=self.frame,
            event_type=event_type,
            data=data,
            message=message,
        )
        self.events.append(event)

        if self.print_live:
            self._print_event(event)

        # Limit stored events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def _print_event(self, event: DebugEvent):
        """Print an event to console."""
        print(f"[{event.frame:05d}] {event.event_type.value}: {event.message}")
        if event.data:
            for key, value in event.data.items():
                print(f"        {key}: {value}")

    def next_frame(self):
        """Advance to next frame."""
        self.frame += 1

    def reset(self):
        """Clear all events and reset frame counter."""
        self.events = []
        self.frame = 0

    def get_events_by_type(self, event_type: EventType) -> List[DebugEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_in_range(self, start_frame: int, end_frame: int) -> List[DebugEvent]:
        """Get events within a frame range."""
        return [e for e in self.events if start_frame <= e.frame <= end_frame]

    def get_last_n_events(self, n: int) -> List[DebugEvent]:
        """Get the last n events."""
        return self.events[-n:]

    def print_summary(self):
        """Print a summary of logged events."""
        print("\n" + "=" * 60)
        print("DEBUG LOG SUMMARY")
        print("=" * 60)
        print(f"Total frames: {self.frame}")
        print(f"Total events: {len(self.events)}")

        # Count by type
        counts: Dict[str, int] = {}
        for event in self.events:
            type_name = event.event_type.value
            counts[type_name] = counts.get(type_name, 0) + 1

        print("\nEvents by type:")
        for type_name, count in sorted(counts.items()):
            print(f"  {type_name}: {count}")

        # Check for validation errors
        errors = self.get_events_by_type(EventType.VALIDATION_ERROR)
        warnings = self.get_events_by_type(EventType.VALIDATION_WARNING)

        if errors:
            print(f"\n⚠ VALIDATION ERRORS: {len(errors)}")
            for event in errors[:5]:  # Show first 5
                print(f"  Frame {event.frame}: {event.message}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        if warnings:
            print(f"\n⚠ VALIDATION WARNINGS: {len(warnings)}")
            for event in warnings[:5]:
                print(f"  Frame {event.frame}: {event.message}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more")

        print("=" * 60)

    def print_frame_log(self, frame: int):
        """Print all events for a specific frame."""
        events = [e for e in self.events if e.frame == frame]
        print(f"\n--- Frame {frame} ({len(events)} events) ---")
        for event in events:
            self._print_event(event)

    def export_json(self, filepath: str):
        """Export all events to JSON file."""
        data = {
            "total_frames": self.frame,
            "events": [e.to_dict() for e in self.events],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(self.events)} events to {filepath}")

    def find_state_transitions(self, event_type: EventType) -> List[Dict]:
        """
        Find all state transitions for a specific event type.
        Useful for tracking is_in changes, etc.
        """
        events = self.get_events_by_type(event_type)
        return [
            {
                "frame": e.frame,
                "data": e.data,
                "message": e.message,
            }
            for e in events
        ]


class GameValidator:
    """
    Validates game state to catch bugs early.

    Checks for:
    - Invalid positions (out of bounds)
    - Invalid state transitions
    - Logic errors
    """

    def __init__(self, logger: DebugLogger, config):
        self.logger = logger
        self.config = config
        self.last_is_in = False
        self.last_hit_by = None

    def validate_ball_position(self, ball) -> bool:
        """Check if ball position is valid."""
        is_valid = True

        # Check bounds (with some tolerance for wall collision detection)
        tolerance = ball.radius + 1
        if ball.x < -tolerance or ball.x > self.config.field_width + tolerance:
            self.logger.log(
                EventType.VALIDATION_ERROR,
                {"x": ball.x, "y": ball.y, "field_width": self.config.field_width},
                f"Ball x position out of bounds: {ball.x}",
            )
            is_valid = False

        if ball.y < -tolerance or ball.y > self.config.field_height + tolerance:
            self.logger.log(
                EventType.VALIDATION_ERROR,
                {"x": ball.x, "y": ball.y, "field_height": self.config.field_height},
                f"Ball y position out of bounds: {ball.y}",
            )
            is_valid = False

        return is_valid

    def validate_player_position(self, player) -> bool:
        """Check if player position is valid."""
        is_valid = True

        if player.x < player.radius or player.x > self.config.field_width - player.radius:
            self.logger.log(
                EventType.VALIDATION_ERROR,
                {"player_id": player.player_id, "x": player.x},
                f"Player {player.player_id} x position out of bounds",
            )
            is_valid = False

        if player.y < player.radius or player.y > self.config.field_height - player.radius:
            self.logger.log(
                EventType.VALIDATION_ERROR,
                {"player_id": player.player_id, "y": player.y},
                f"Player {player.player_id} y position out of bounds",
            )
            is_valid = False

        return is_valid

    def validate_is_in_transition(self, ball, was_in_area: bool):
        """Validate is_in state transitions."""
        # is_in should only turn ON when entering an in-area
        if not self.last_is_in and ball.is_in and not was_in_area:
            self.logger.log(
                EventType.VALIDATION_WARNING,
                {
                    "x": ball.x,
                    "y": ball.y,
                    "is_in": ball.is_in,
                    "was_in_area": was_in_area,
                },
                "is_in turned ON without entering in-area",
            )

        self.last_is_in = ball.is_in

    def validate_hit(self, player, ball, success: bool):
        """Validate hit attempt logic."""
        distance = ball.distance_to(player.x, player.y)

        if success:
            # Hit succeeded - verify conditions were met
            if distance > player.reach_distance:
                self.logger.log(
                    EventType.VALIDATION_ERROR,
                    {
                        "player_id": player.player_id,
                        "distance": distance,
                        "reach": player.reach_distance,
                    },
                    f"Hit succeeded but distance {distance:.1f} > reach {player.reach_distance}",
                )
        else:
            # Hit failed - log why
            reasons = []
            if distance > player.reach_distance:
                reasons.append(f"too far ({distance:.1f} > {player.reach_distance})")
            if not ball.is_in:
                reasons.append("is_in is OFF")

            self.logger.log(
                EventType.PLAYER_HIT_FAIL,
                {
                    "player_id": player.player_id,
                    "distance": distance,
                    "is_in": ball.is_in,
                },
                f"Hit failed: {', '.join(reasons)}",
            )


# Global logger instance (optional)
_global_logger: Optional[DebugLogger] = None


def get_logger() -> DebugLogger:
    """Get or create the global debug logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DebugLogger()
    return _global_logger


def set_logger(logger: DebugLogger):
    """Set the global debug logger."""
    global _global_logger
    _global_logger = logger
