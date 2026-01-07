"""Input handling and UI state management for 2D Tennis Simulator."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@dataclass
class InputState:
    """Current state of input/UI controls."""
    quit_requested: bool = False
    paused: bool = False
    step_requested: bool = False
    save_requested: bool = False
    # Debug display toggles
    show_trajectory: bool = True
    show_distances: bool = False
    show_state_panel: bool = True
    show_graphs: bool = True


class InputHandler:
    """Handles pygame events and manages UI state.

    Separates input processing from rendering. The game loop queries
    this handler for state, rather than asking the renderer.
    """

    def __init__(self, debug_mode: bool = False):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required: pip install pygame")
        self.debug_mode = debug_mode
        self.state = InputState()
        self._key_bindings: Dict[int, Callable[[], None]] = self._setup_bindings()

    def _setup_bindings(self) -> Dict[int, Callable[[], None]]:
        """Configure key bindings."""
        bindings = {
            pygame.K_ESCAPE: lambda: setattr(self.state, 'quit_requested', True),
            pygame.K_s: lambda: setattr(self.state, 'save_requested', True),
        }
        if self.debug_mode:
            bindings.update({
                pygame.K_SPACE: lambda: setattr(self.state, 'paused', not self.state.paused),
                pygame.K_n: lambda: setattr(self.state, 'step_requested', True),
                pygame.K_t: lambda: setattr(self.state, 'show_trajectory', not self.state.show_trajectory),
                pygame.K_d: lambda: setattr(self.state, 'show_distances', not self.state.show_distances),
                pygame.K_p: lambda: setattr(self.state, 'show_state_panel', not self.state.show_state_panel),
                pygame.K_g: lambda: setattr(self.state, 'show_graphs', not self.state.show_graphs),
            })
        return bindings

    def process_events(self) -> None:
        """Process all pending pygame events and update state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in self._key_bindings:
                    self._key_bindings[event.key]()

    def should_step(self) -> bool:
        """Determine if game should advance this frame.

        Returns True unless paused (without step request).
        Consumes the step_requested flag if set.
        """
        if not self.state.paused:
            return True
        if self.state.step_requested:
            self.state.step_requested = False
            return True
        return False

    def consume_save_request(self) -> bool:
        """Check and consume save request flag."""
        if self.state.save_requested:
            self.state.save_requested = False
            return True
        return False

    @property
    def running(self) -> bool:
        """True if game should continue running."""
        return not self.state.quit_requested
