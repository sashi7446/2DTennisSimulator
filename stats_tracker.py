"""Statistics tracking for 2D Tennis Simulator."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    episode_num: int
    winner: int  # 0 or 1
    total_reward_a: float
    total_reward_b: float
    steps: int


class StatsTracker:
    """Tracks game statistics across episodes.

    Separated from rendering to allow headless training
    with the same statistics collection.
    """

    def __init__(self, moving_avg_window: int = 20, max_history: int = 200):
        self.moving_avg_window = moving_avg_window
        self.max_history = max_history

        # Per-episode totals
        self.total_wins = [0, 0]
        self.episode_count = 0

        # Current episode tracking
        self.current_reward_a = 0.0
        self.current_reward_b = 0.0

        # Per-step cumulative rewards (for graphs)
        self.cumulative_rewards_a: List[float] = []
        self.cumulative_rewards_b: List[float] = []

        # Per-episode final rewards (for moving average)
        self.episode_rewards_a: List[float] = []
        self.episode_rewards_b: List[float] = []

        # Event log
        self.event_log: Deque[str] = deque(maxlen=8)
        self.frame_count = 0

    def add_reward(self, reward_a: float, reward_b: float) -> None:
        """Record rewards for current step."""
        self.current_reward_a += reward_a
        self.current_reward_b += reward_b
        self.cumulative_rewards_a.append(self.current_reward_a)
        self.cumulative_rewards_b.append(self.current_reward_b)

    def log_event(self, message: str) -> None:
        """Add event to log."""
        self.event_log.appendleft(f"F{self.frame_count}: {message}")

    def next_frame(self) -> None:
        """Advance frame counter."""
        self.frame_count += 1

    def end_episode(self, winner: int) -> None:
        """Finalize episode statistics."""
        self.episode_count += 1
        self.total_wins[winner] += 1

        self.episode_rewards_a.append(self.current_reward_a)
        self.episode_rewards_b.append(self.current_reward_b)

        # Trim history
        if len(self.episode_rewards_a) > self.max_history:
            self.episode_rewards_a = self.episode_rewards_a[-self.max_history :]
            self.episode_rewards_b = self.episode_rewards_b[-self.max_history :]

        # Reset for next episode
        self.current_reward_a = 0.0
        self.current_reward_b = 0.0
        self.cumulative_rewards_a = []
        self.cumulative_rewards_b = []

    def get_moving_averages(self, rewards: List[float]) -> List[float]:
        """Calculate moving average over window."""
        if not rewards:
            return []
        result = []
        for i in range(len(rewards)):
            start = max(0, i - self.moving_avg_window + 1)
            window = rewards[start : i + 1]
            result.append(sum(window) / len(window))
        return result

    @property
    def win_rate_a(self) -> float:
        """Win rate for player A."""
        total = sum(self.total_wins)
        return self.total_wins[0] / total if total > 0 else 0.5

    @property
    def win_rate_b(self) -> float:
        """Win rate for player B."""
        return 1.0 - self.win_rate_a
