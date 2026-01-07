"""Pygame renderer for 2D Tennis Simulator.

Renderers are passive - they only draw what they're given.
They do NOT control game flow or manage input.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Deque, Protocol, TYPE_CHECKING, Any
from collections import deque
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None  # type: ignore

if TYPE_CHECKING:
    from pygame import Surface

from config import Config
from game import Game, GameState

# Colors
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, LIGHT_GREEN = (34, 139, 34), (144, 238, 144)
YELLOW, RED, BLUE = (255, 255, 0), (255, 100, 100), (100, 100, 255)
GRAY, ORANGE, CYAN = (128, 128, 128), (255, 165, 0), (0, 255, 255)


class Overlay(Protocol):
    """Protocol for debug overlay components."""
    def update(self, game: Game) -> None: ...
    def draw(self, screen: Any, config: Config, field_to_screen) -> None: ...


class GameRenderer:
    """Pure renderer - draws game state to screen. No game logic."""

    def __init__(self, config: Optional[Config] = None):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required: pip install pygame")

        self.config = config or Config()
        self.padding, self.ui_height = 50, 60
        self.window_width = self.config.field_width + 2 * self.padding
        self.window_height = self.config.field_height + 2 * self.padding + self.ui_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2D Tennis Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def field_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert field coordinates to screen coordinates."""
        x = 0.0 if (math.isnan(x) or math.isinf(x)) else x
        y = 0.0 if (math.isnan(y) or math.isinf(y)) else y
        return (int(x + self.padding), int(y + self.padding + self.ui_height))

    def _draw_field(self, game: Game) -> None:
        self.screen.fill(GRAY)
        field_rect = pygame.Rect(self.padding, self.padding + self.ui_height,
                                  self.config.field_width, self.config.field_height)
        pygame.draw.rect(self.screen, GREEN, field_rect)

        for area in [game.field.area_a, game.field.area_b]:
            rect = pygame.Rect(*self.field_to_screen(area.x, area.y), int(area.width), int(area.height))
            pygame.draw.rect(self.screen, LIGHT_GREEN, rect)
            pygame.draw.rect(self.screen, WHITE, rect, 2)

        center_x = self.field_to_screen(self.config.field_width / 2, 0)[0]
        pygame.draw.line(self.screen, WHITE, (center_x, self.padding + self.ui_height),
                         (center_x, self.padding + self.ui_height + self.config.field_height), 1)
        pygame.draw.rect(self.screen, WHITE, field_rect, 3)

    def _draw_ball(self, game: Game) -> None:
        if game.ball is None:
            return
        pos = self.field_to_screen(game.ball.x, game.ball.y)
        color = YELLOW if game.ball.in_flag else (200, 200, 0)
        pygame.draw.circle(self.screen, color, pos, game.ball.radius)

    def _draw_players(self, game: Game) -> None:
        for player, color, reach_color in [
            (game.player_a, RED, (255, 150, 150)),
            (game.player_b, BLUE, (150, 150, 255))
        ]:
            pos = self.field_to_screen(player.x, player.y)
            pygame.draw.circle(self.screen, color, pos, int(player.radius))
            pygame.draw.circle(self.screen, WHITE, pos, int(player.radius), 2)
            pygame.draw.circle(self.screen, reach_color, pos, int(player.reach_distance), 1)

    def _draw_ui(self, game: Game, total_wins: Optional[Tuple[int, int]] = None) -> None:
        if total_wins:
            score = self.font.render(f"Total Wins - A: {total_wins[0]}  B: {total_wins[1]}", True, WHITE)
        else:
            score = self.font.render(f"Player A: {game.scores[0]}  -  Player B: {game.scores[1]}", True, WHITE)
        self.screen.blit(score, score.get_rect(center=(self.window_width // 2, 25)))

        in_flag = game.ball and game.ball.in_flag
        self.screen.blit(self.small_font.render(f"Ball: {'IN' if in_flag else 'OUT'}", True, YELLOW if in_flag else GRAY), (10, 45))
        self.screen.blit(self.small_font.render(f"Rally: {game.rally_count}", True, WHITE), (self.window_width - 100, 45))

        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            text = self.font.render(f"Game Over! Player {winner} wins!", True, WHITE)
            rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            pygame.draw.rect(self.screen, BLACK, rect.inflate(20, 10))
            self.screen.blit(text, rect)

    def render(self, game: Game, total_wins: Optional[Tuple[int, int]] = None) -> None:
        """Render game state to screen."""
        self._draw_field(game)
        self._draw_ball(game)
        self._draw_players(game)
        self._draw_ui(game, total_wins)
        pygame.display.flip()

    def tick(self, fps: Optional[int] = None) -> None:
        """Control frame rate."""
        self.clock.tick(fps or self.config.fps)

    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.quit()


class TrajectoryOverlay:
    """Draws ball trajectory trail."""

    def __init__(self, trail_length: int = 60):
        self.trail: Deque[Tuple[float, float, bool]] = deque(maxlen=trail_length)

    def update(self, game: Game) -> None:
        if game.ball:
            x, y = game.ball.x, game.ball.y
            if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                self.trail.append((x, y, game.ball.in_flag))

    def draw(self, screen: "Surface", config: Config, field_to_screen) -> None:
        if len(self.trail) < 2:
            return
        trail = list(self.trail)
        for i in range(1, len(trail)):
            alpha = int(255 * i / len(trail))
            color = (alpha, alpha, 0) if trail[i][2] else (alpha // 2, alpha // 2, 0)
            pygame.draw.line(screen, color,
                             field_to_screen(trail[i-1][0], trail[i-1][1]),
                             field_to_screen(trail[i][0], trail[i][1]), 2)
            if trail[i][2] != trail[i-1][2]:
                pygame.draw.circle(screen, ORANGE if trail[i][2] else RED,
                                   field_to_screen(trail[i][0], trail[i][1]), 6, 2)


class DistanceOverlay:
    """Draws distance lines between players and ball."""

    def draw(self, screen: "Surface", game: Game, field_to_screen, debug_font) -> None:
        if game.ball is None:
            return
        ball_pos = field_to_screen(game.ball.x, game.ball.y)
        for player, color in [(game.player_a, RED), (game.player_b, BLUE)]:
            ppos = field_to_screen(player.x, player.y)
            pygame.draw.line(screen, color, ppos, ball_pos, 1)
            can_hit = player.can_hit(game.ball)
            text = f"{player.distance_to_ball(game.ball):.1f}" + (" [HIT]" if can_hit else "")
            screen.blit(debug_font.render(text, True, CYAN if can_hit else GRAY),
                        ((ppos[0] + ball_pos[0]) // 2, (ppos[1] + ball_pos[1]) // 2))


class StatePanelOverlay:
    """Draws game state information panel."""

    def draw(self, screen: "Surface", game: Game, frame_count: int,
             x: int, y: int, debug_font) -> None:
        pw, ph = 180, 200
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))

        cy = y + 5
        def line(t, c=WHITE):
            nonlocal cy
            screen.blit(debug_font.render(t, True, c), (x + 5, cy))
            cy += 16

        line(f"Frame: {frame_count}", CYAN)
        line(f"Steps: {game.total_steps}")
        line(f"State: {game.state.value}")
        line("---")
        if game.ball:
            line("Ball:", YELLOW)
            line(f"  Pos: ({game.ball.x:.1f}, {game.ball.y:.1f})")
            line(f"  Vel: ({game.ball.vx:.1f}, {game.ball.vy:.1f})")
            line(f"  In-Flag: {game.ball.in_flag}", YELLOW if game.ball.in_flag else GRAY)
            lh = game.ball.last_hit_by
            line(f"  Last Hit: {['A','B'][lh] if lh is not None else 'None'}")
        line("---")
        line("Players:")
        line(f"  A: ({game.player_a.x:.0f}, {game.player_a.y:.0f})", RED)
        line(f"  B: ({game.player_b.x:.0f}, {game.player_b.y:.0f})", BLUE)


class RewardGraphOverlay:
    """Draws reward graphs."""

    def __init__(self, max_points: int = 200):
        self.max_points = max_points

    def draw(self, screen: "Surface", cumulative_a: List[float], cumulative_b: List[float],
             moving_avg_a: List[float], moving_avg_b: List[float],
             episode_count: int, x: int, y: int, debug_font) -> None:
        gw, gh, m = 180, 80, 5

        self._draw_graph(screen, x, y, gw, gh,
                         cumulative_a[-self.max_points:], cumulative_b[-self.max_points:],
                         "Cumulative (Episode)", debug_font)
        self._draw_graph(screen, x + gw + m, y, gw, gh,
                         moving_avg_a, moving_avg_b,
                         "Reward (20-ep MA)", debug_font)

        ly = y + gh + 5
        screen.blit(debug_font.render("■ Player A", True, RED), (x, ly))
        screen.blit(debug_font.render("■ Player B", True, BLUE), (x + 80, ly))
        screen.blit(debug_font.render(f"Episodes: {episode_count}", True, WHITE), (x + gw + m, ly))

    def _draw_graph(self, screen, x, y, width, height,
                    data_a: List[float], data_b: List[float], title: str, debug_font) -> None:
        surf = pygame.Surface((width, height))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))
        pygame.draw.rect(screen, WHITE, (x, y, width, height), 1)
        screen.blit(debug_font.render(title, True, WHITE), (x + 5, y + 2))

        gx, gy, gw, gh = x + 5, y + 18, width - 10, height - 23
        if len(data_a) < 2 and len(data_b) < 2:
            screen.blit(debug_font.render("No data", True, GRAY), (x + width // 2 - 20, y + height // 2))
            return

        all_data = data_a + data_b
        if not all_data:
            return
        min_v, max_v = min(all_data), max(all_data)
        rng = max_v - min_v or 1
        min_v, max_v = min_v - rng * 0.1, max_v + rng * 0.1
        rng = max_v - min_v

        if min_v < 0 < max_v:
            zy = gy + gh - int(-min_v / rng * gh)
            pygame.draw.line(screen, GRAY, (gx, zy), (gx + gw, zy), 1)

        def draw_series(data, color):
            if len(data) < 2:
                return
            pts = [(gx + int(i / (len(data) - 1) * gw),
                    max(gy, min(gy + gh, gy + gh - int((v - min_v) / rng * gh))))
                   for i, v in enumerate(data)]
            pygame.draw.lines(screen, color, False, pts, 2)

        draw_series(data_a, RED)
        draw_series(data_b, BLUE)
        if data_a:
            screen.blit(debug_font.render(f"A:{data_a[-1]:.2f}", True, RED), (x + width - 60, y + 2))
        if data_b:
            screen.blit(debug_font.render(f"B:{data_b[-1]:.2f}", True, BLUE), (x + width - 60, y + 12))


class DebugRenderer(GameRenderer):
    """Extended renderer with debug overlays.

    Uses composition for overlay components but inherits base rendering.
    Still passive - takes display flags from InputState.
    """

    def __init__(self, config: Optional[Config] = None, trail_length: int = 60):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required: pip install pygame")

        self.config = config or Config()
        self.padding, self.ui_height = 50, 60
        self.debug_canvas_height = 300
        self.window_width = self.config.field_width + 2 * self.padding
        self.window_height = self.config.field_height + 2 * self.padding + self.ui_height + self.debug_canvas_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2D Tennis Simulator [DEBUG MODE]")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.debug_font = pygame.font.Font(None, 18)

        # Overlay components
        self.trajectory = TrajectoryOverlay(trail_length)
        self.distance = DistanceOverlay()
        self.state_panel = StatePanelOverlay()
        self.reward_graph = RewardGraphOverlay()

    def render(self, game: Game, total_wins: Optional[Tuple[int, int]] = None,
               stats=None, input_state=None, frame_count: int = 0) -> None:
        """Render game with debug overlays.

        Args:
            game: Current game state
            total_wins: (wins_a, wins_b) tuple
            stats: StatsTracker instance for graphs
            input_state: InputState for display toggles
            frame_count: Current frame number
        """
        # Base rendering
        self._draw_field(game)

        # Trajectory (before ball so trail is behind)
        if input_state and input_state.show_trajectory:
            self.trajectory.draw(self.screen, self.config, self.field_to_screen)

        self._draw_ball(game)
        self._draw_players(game)

        # Distance overlay
        if input_state and input_state.show_distances:
            self.distance.draw(self.screen, game, self.field_to_screen, self.debug_font)

        # State panel
        if input_state and input_state.show_state_panel:
            self.state_panel.draw(self.screen, game, frame_count,
                                  10, self.padding + self.ui_height + 10, self.debug_font)

        # Event log
        if stats and stats.event_log:
            self._draw_event_log(stats.event_log)

        # Reward graphs
        if input_state and input_state.show_graphs and stats:
            gx = self.padding + 200
            gy = self.padding + self.ui_height + self.config.field_height - 165 - 10
            self.reward_graph.draw(
                self.screen,
                stats.cumulative_rewards_a, stats.cumulative_rewards_b,
                stats.get_moving_averages(stats.episode_rewards_a),
                stats.get_moving_averages(stats.episode_rewards_b),
                stats.episode_count, gx, gy, self.debug_font
            )

        # Controls help
        self._draw_controls_help()

        # UI
        self._draw_ui(game, total_wins)

        # Paused indicator
        if input_state and input_state.paused:
            text = self.font.render("PAUSED", True, ORANGE)
            self.screen.blit(text, text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50)))

        pygame.display.flip()

    def update(self, game: Game) -> None:
        """Update overlay state (call each frame when game advances)."""
        self.trajectory.update(game)

    def _draw_event_log(self, event_log: Deque[str]) -> None:
        x, y = self.window_width - 200, self.padding + self.ui_height + 10
        for i, ev in enumerate(event_log):
            alpha = int(255 * (1 - i / len(event_log)))
            self.screen.blit(self.debug_font.render(ev, True, (alpha, alpha, alpha)), (x, y + i * 14))

    def _draw_controls_help(self) -> None:
        helps = ["Debug Controls:", "T - Trajectory", "D - Distances", "P - State panel",
                 "G - Graphs", "SPACE - Pause", "N - Step"]
        x, y = self.window_width - 150, self.window_height - 120
        for i, t in enumerate(helps):
            self.screen.blit(self.debug_font.render(t, True, CYAN if i == 0 else GRAY), (x, y + i * 14))

    def _draw_ui(self, game: Game, total_wins: Optional[Tuple[int, int]] = None) -> None:
        if total_wins:
            score = self.font.render(f"Total Wins - A: {total_wins[0]}  B: {total_wins[1]}", True, WHITE)
        else:
            score = self.font.render(f"Player A: {game.scores[0]}  -  Player B: {game.scores[1]}", True, WHITE)
        self.screen.blit(score, score.get_rect(center=(self.window_width // 2, 25)))

        in_flag = game.ball and game.ball.in_flag
        self.screen.blit(self.small_font.render(f"Ball: {'IN' if in_flag else 'OUT'}", True, YELLOW if in_flag else GRAY), (10, 45))
        self.screen.blit(self.small_font.render(f"Rally: {game.rally_count}", True, WHITE), (self.window_width - 100, 45))

        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            text = self.font.render(f"Episode Over! Player {winner} wins!", True, WHITE)
            self.screen.blit(text, text.get_rect(center=(self.window_width // 2, self.window_height - 30)))


# Backward compatibility aliases
Renderer = GameRenderer


def run_demo_game() -> None:
    """Demo game with random agents."""
    import random
    from input_handler import InputHandler
    from stats_tracker import StatsTracker

    config = Config()
    game = Game(config)
    renderer = DebugRenderer(config)
    input_handler = InputHandler(debug_mode=True)
    stats = StatsTracker()
    frame_count = 0

    while input_handler.running and not game.is_game_over:
        input_handler.process_events()

        if input_handler.should_step():
            game.step((random.randint(0, 16), random.uniform(0, 360)),
                      (random.randint(0, 16), random.uniform(0, 360)))
            renderer.update(game)
            stats.next_frame()
            frame_count += 1

        renderer.render(game, stats=stats, input_state=input_handler.state, frame_count=frame_count)
        renderer.tick()

    if game.is_game_over:
        for _ in range(180):
            input_handler.process_events()
            if not input_handler.running:
                break
            renderer.render(game, stats=stats, input_state=input_handler.state, frame_count=frame_count)
            renderer.tick()

    renderer.close()


if __name__ == "__main__":
    run_demo_game()
