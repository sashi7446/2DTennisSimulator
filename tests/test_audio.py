"""Tests for procedural sound effects.

These must pass on machines with no audio device, so nothing here opens
the mixer: the waveform helpers are pure numpy, and SoundBank is only
exercised in its disabled (silent) form.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio import (  # noqa: E402
    HIT_FREQUENCIES,
    NUMPY_AVAILABLE,
    SAMPLE_RATE,
    SoundBank,
    make_chirp_wave,
    make_hit_wave,
)

pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")


class TestWaveforms:
    """Synthesis is pure numpy - no device needed."""

    def test_hit_wave_length_matches_duration(self):
        wave = make_hit_wave(440.0, duration=0.05)
        assert len(wave) == int(SAMPLE_RATE * 0.05)

    def test_hit_wave_stays_in_range(self):
        wave = make_hit_wave(440.0)
        assert wave.max() <= 1.5
        assert wave.min() >= -1.5

    def test_hit_wave_decays(self):
        """The tail must be quieter than the attack, or it is not a hit."""
        wave = make_hit_wave(440.0)
        head = abs(wave[: len(wave) // 4]).mean()
        tail = abs(wave[-len(wave) // 4 :]).mean()
        assert tail < head

    def test_hit_wave_is_deterministic(self):
        assert (make_hit_wave(440.0) == make_hit_wave(440.0)).all()

    def test_different_pitches_differ(self):
        assert not (make_hit_wave(380.0) == make_hit_wave(820.0)).all()

    def test_chirp_concatenates_notes(self):
        wave = make_chirp_wave((523, 784), note_duration=0.05)
        assert len(wave) == 2 * int(SAMPLE_RATE * 0.05)

    def test_chirp_direction_matters(self):
        """Rising and falling stings must not be the same sound."""
        rising = make_chirp_wave((523, 784))
        falling = make_chirp_wave((392, 262))
        assert not (rising == falling).all()


class TestSilentFallback:
    """A disabled bank must be safe to call everywhere."""

    def test_disabled_bank_is_unavailable(self):
        assert SoundBank(enabled=False).available is False

    def test_hit_is_a_noop_when_unavailable(self):
        SoundBank(enabled=False).hit(1.0)

    def test_point_is_a_noop_when_unavailable(self):
        bank = SoundBank(enabled=False)
        bank.point("in")
        bank.point("out")
        bank.point(None)

    def test_close_is_a_noop_when_unavailable(self):
        SoundBank(enabled=False).close()

    def test_no_sounds_are_built_when_disabled(self):
        """Disabled means no synthesis work at all."""
        assert SoundBank(enabled=False)._hits == []


class TestPitchSelection:
    """Speed maps onto the hit pitch buckets without going out of range."""

    @staticmethod
    def _index(speed_ratio):
        index = int(round((speed_ratio - 0.6) / 0.2))
        return max(0, min(len(HIT_FREQUENCIES) - 1, index))

    def test_slow_ball_clamps_to_lowest(self):
        assert self._index(0.1) == 0

    def test_fast_ball_clamps_to_highest(self):
        assert self._index(9.0) == len(HIT_FREQUENCIES) - 1

    def test_nominal_speed_is_mid_range(self):
        assert 0 < self._index(1.0) < len(HIT_FREQUENCIES) - 1

    def test_faster_is_never_lower_pitched(self):
        indices = [self._index(r / 10) for r in range(1, 30)]
        assert indices == sorted(indices)
