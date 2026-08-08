"""Procedural sound effects for 2D Tennis Simulator.

Every sound is synthesised with numpy at startup, so there are no audio
assets to ship. The whole module degrades to silence when numpy, pygame or
an audio device is missing - headless training must never touch the mixer.
"""

from typing import List, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

SAMPLE_RATE = 22050
BIT_DEPTH = -16
CHANNELS = 2
BUFFER_SIZE = 512  # Small buffer keeps hits tight against the visuals

HIT_FREQUENCIES = (380.0, 470.0, 560.0, 680.0, 820.0)


def _to_stereo(wave: "np.ndarray", gain: float = 0.6) -> "np.ndarray":
    """Convert a mono float waveform in [-1, 1] to 16-bit stereo samples."""
    clipped = np.clip(wave, -1.0, 1.0)
    mono = (clipped * gain * 32767).astype(np.int16)
    return np.repeat(mono[:, None], CHANNELS, axis=1)


def make_hit_wave(frequency: float, duration: float = 0.07) -> "np.ndarray":
    """A short percussive 'pock' - noise transient over a decaying tone."""
    rng = np.random.default_rng(int(frequency))
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE

    body = np.sin(2 * np.pi * frequency * t) * 0.7
    body += np.sin(2 * np.pi * frequency * 2 * t) * 0.25
    transient = rng.uniform(-1.0, 1.0, n) * np.exp(-t * 400) * 0.35

    return (body + transient) * np.exp(-t * 55)


def make_chirp_wave(frequencies: tuple, note_duration: float = 0.09) -> "np.ndarray":
    """Two notes in sequence - rising or falling depending on the order."""
    n = int(SAMPLE_RATE * note_duration)
    t = np.arange(n) / SAMPLE_RATE
    envelope = np.exp(-t * 14) * np.minimum(1.0, t * 200)

    notes = [np.sin(2 * np.pi * freq * t) * envelope for freq in frequencies]
    return np.concatenate(notes)


class SoundBank:
    """Holds the synthesised effects and plays them on request.

    Always safe to call: when the mixer is unavailable every method is a
    no-op and `available` stays False.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.45):
        self.available = False
        self._hits: List = []
        self._point_in = None
        self._point_out = None

        if not enabled or not NUMPY_AVAILABLE or not PYGAME_AVAILABLE:
            return

        try:
            pygame.mixer.pre_init(SAMPLE_RATE, BIT_DEPTH, CHANNELS, BUFFER_SIZE)
            pygame.mixer.init(SAMPLE_RATE, BIT_DEPTH, CHANNELS, BUFFER_SIZE)
        except pygame.error:
            # No audio device (common in containers and over SSH) - stay silent
            return

        try:
            self._hits = [
                pygame.sndarray.make_sound(_to_stereo(make_hit_wave(f))) for f in HIT_FREQUENCIES
            ]
            # Falling for "out" so an error is audible without looking
            self._point_in = pygame.sndarray.make_sound(_to_stereo(make_chirp_wave((523, 784))))
            self._point_out = pygame.sndarray.make_sound(_to_stereo(make_chirp_wave((392, 262))))
        except (pygame.error, ValueError):
            pygame.mixer.quit()
            return

        for sound in [*self._hits, self._point_in, self._point_out]:
            sound.set_volume(volume)

        self.available = True

    def hit(self, speed_ratio: float = 1.0) -> None:
        """Play a hit, pitched up for faster shots."""
        if not self.available:
            return
        index = int(round((speed_ratio - 0.6) / 0.2))
        index = max(0, min(len(self._hits) - 1, index))
        self._hits[index].play()

    def point(self, reason: Optional[str] = None) -> None:
        """Play the point-decided sting. 'out' gets the falling version."""
        if not self.available:
            return
        sound = self._point_out if reason == "out" else self._point_in
        if sound is not None:
            sound.play()

    def close(self) -> None:
        """Release the audio device."""
        if self.available:
            pygame.mixer.quit()
            self.available = False
