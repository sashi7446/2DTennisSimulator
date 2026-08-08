"""Tests for StatsTracker."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_tracker import StatsTracker  # noqa: E402


class TestRallyTracking:
    """Rally history, records and the record flash window."""

    def test_end_episode_records_rally(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=5, reason="in")
        assert stats.rally_history == [5]
        assert stats.longest_rally == 5
        assert stats.last_point_reason == "in"

    def test_first_point_is_a_record(self):
        stats = StatsTracker()
        assert stats.end_episode(0, rally=3) is True

    def test_longer_rally_sets_new_record(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=3)
        assert stats.end_episode(1, rally=7) is True
        assert stats.longest_rally == 7

    def test_shorter_rally_is_not_a_record(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=7)
        assert stats.end_episode(1, rally=2) is False
        assert stats.longest_rally == 7

    def test_tying_the_record_is_not_a_record(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=4)
        assert stats.end_episode(1, rally=4) is False

    def test_zero_rally_point_is_not_a_record(self):
        """A service winner has no rally, so it must not claim the record."""
        stats = StatsTracker()
        assert stats.end_episode(0, rally=0) is False
        assert stats.longest_rally == 0

    def test_rally_history_is_trimmed(self):
        stats = StatsTracker(max_history=5)
        for i in range(10):
            stats.end_episode(0, rally=i)
        assert len(stats.rally_history) == 5
        assert stats.rally_history == [5, 6, 7, 8, 9]

    def test_average_rally(self):
        stats = StatsTracker()
        for rally in (2, 4, 6):
            stats.end_episode(0, rally=rally)
        assert stats.average_rally == 4.0

    def test_average_rally_empty(self):
        assert StatsTracker().average_rally == 0.0

    def test_end_episode_defaults_are_backward_compatible(self):
        """Older callers pass only the winner."""
        stats = StatsTracker()
        assert stats.end_episode(0) is False
        assert stats.episode_count == 1
        assert stats.total_wins == [1, 0]


class TestRecordFlash:
    """record_is_fresh drives the on-screen record announcement."""

    def test_not_fresh_without_a_record(self):
        assert StatsTracker().record_is_fresh() is False

    def test_fresh_right_after_a_record(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=3)
        assert stats.record_is_fresh() is True

    def test_expires_after_the_window(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=3)
        for _ in range(120):
            stats.next_frame()
        assert stats.record_is_fresh() is False

    def test_still_fresh_inside_the_window(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=3)
        for _ in range(119):
            stats.next_frame()
        assert stats.record_is_fresh() is True

    def test_non_record_does_not_refresh_the_flash(self):
        stats = StatsTracker()
        stats.end_episode(0, rally=9)
        for _ in range(120):
            stats.next_frame()
        stats.end_episode(1, rally=1)
        assert stats.record_is_fresh() is False


class TestHitEvents:
    """Recent hits drive the impact-ring effect."""

    def test_no_hits_initially(self):
        assert StatsTracker().recent_hits() == []

    def test_fresh_hit_has_zero_age(self):
        stats = StatsTracker()
        stats.log_hit(100.0, 200.0, 0)
        assert stats.recent_hits() == [(0.0, 100.0, 200.0, 0)]

    def test_age_grows_with_frames(self):
        stats = StatsTracker()
        stats.log_hit(10.0, 20.0, 1)
        for _ in range(5):
            stats.next_frame()
        age, x, y, player_id = stats.recent_hits(window_frames=10)[0]
        assert age == 0.5
        assert (x, y, player_id) == (10.0, 20.0, 1)

    def test_hit_expires_outside_window(self):
        stats = StatsTracker()
        stats.log_hit(10.0, 20.0, 0)
        for _ in range(10):
            stats.next_frame()
        assert stats.recent_hits(window_frames=10) == []

    def test_only_recent_hits_are_returned(self):
        stats = StatsTracker()
        stats.log_hit(1.0, 1.0, 0)
        for _ in range(20):
            stats.next_frame()
        stats.log_hit(2.0, 2.0, 1)
        hits = stats.recent_hits(window_frames=10)
        assert len(hits) == 1
        assert hits[0][1] == 2.0

    def test_hit_buffer_is_bounded(self):
        stats = StatsTracker()
        for i in range(20):
            stats.log_hit(float(i), 0.0, 0)
        assert len(stats.hit_events) == 8


class TestEventLog:
    """The play-by-play feed."""

    def test_most_recent_event_is_first(self):
        stats = StatsTracker()
        stats.log_event("first")
        stats.log_event("second")
        assert list(stats.event_log) == ["second", "first"]

    def test_log_is_bounded(self):
        stats = StatsTracker()
        for i in range(20):
            stats.log_event(f"event {i}")
        assert len(stats.event_log) == 8

    def test_message_is_not_rewritten(self):
        """The feed is read by humans - no frame prefix noise."""
        stats = StatsTracker()
        stats.next_frame()
        stats.log_event("POINT A  (in)")
        assert stats.event_log[0] == "POINT A  (in)"
