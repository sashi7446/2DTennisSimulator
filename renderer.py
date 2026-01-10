"""Pygame renderer for 2D Tennis Simulator.

Renderers are passive - they only draw what they're given.
They do NOT control game flow or manage input.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Protocol, Tuple

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
        field_rect = pygame.Rect(
            self.padding,
            self.padding + self.ui_height,
            self.config.field_width,
            self.config.field_height,
        )
        pygame.draw.rect(self.screen, GREEN, field_rect)

        for area in [game.field.area_a, game.field.area_b]:
            rect = pygame.Rect(
                *self.field_to_screen(area.x, area.y), int(area.width), int(area.height)
            )
            pygame.draw.rect(self.screen, LIGHT_GREEN, rect)
            pygame.draw.rect(self.screen, WHITE, rect, 2)

        center_x = self.field_to_screen(self.config.field_width / 2, 0)[0]
        pygame.draw.line(
            self.screen,
            WHITE,
            (center_x, self.padding + self.ui_height),
            (center_x, self.padding + self.ui_height + self.config.field_height),
            1,
        )
        pygame.draw.rect(self.screen, WHITE, field_rect, 3)

    def _draw_ball(self, game: Game) -> None:
        if game.ball is None:
            return
        pos = self.field_to_screen(game.ball.x, game.ball.y)
        color = YELLOW if game.ball.is_in else (200, 200, 0)
        pygame.draw.circle(self.screen, color, pos, game.ball.radius)

    def _draw_players(self, game: Game) -> None:
        for player, color, reach_color in [
            (game.player_a, RED, (255, 150, 150)),
            (game.player_b, BLUE, (150, 150, 255)),
        ]:
            pos = self.field_to_screen(player.x, player.y)
            pygame.draw.circle(self.screen, color, pos, int(player.radius))
            pygame.draw.circle(self.screen, WHITE, pos, int(player.radius), 2)
            pygame.draw.circle(self.screen, reach_color, pos, int(player.reach_distance), 1)

    def _draw_ui(self, game: Game, total_wins: Optional[Tuple[int, int]] = None) -> None:
        if total_wins:
            score = self.font.render(
                f"Total Wins - A: {total_wins[0]}  B: {total_wins[1]}", True, WHITE
            )
        else:
            score = self.font.render(
                f"Player A: {game.scores[0]}  -  Player B: {game.scores[1]}", True, WHITE
            )
        self.screen.blit(score, score.get_rect(center=(self.window_width // 2, 25)))

        is_in = game.ball and game.ball.is_in
        self.screen.blit(
            self.small_font.render(
                f"Ball: {'IN' if is_in else 'OUT'}", True, YELLOW if is_in else GRAY
            ),
            (10, 45),
        )
        self.screen.blit(
            self.small_font.render(f"Rally: {game.rally_count}", True, WHITE),
            (self.window_width - 100, 45),
        )

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
                self.trail.append((x, y, game.ball.is_in))

    def draw(self, screen: Surface, config: Config, field_to_screen) -> None:
        if len(self.trail) < 2:
            return
        trail = list(self.trail)
        for i in range(1, len(trail)):
            alpha = int(255 * i / len(trail))
            color = (alpha, alpha, 0) if trail[i][2] else (alpha // 2, alpha // 2, 0)
            pygame.draw.line(
                screen,
                color,
                field_to_screen(trail[i - 1][0], trail[i - 1][1]),
                field_to_screen(trail[i][0], trail[i][1]),
                2,
            )
            if trail[i][2] != trail[i - 1][2]:
                pygame.draw.circle(
                    screen,
                    ORANGE if trail[i][2] else RED,
                    field_to_screen(trail[i][0], trail[i][1]),
                    6,
                    2,
                )


class DistanceOverlay:
    """Draws distance lines between players and ball."""

    def draw(self, screen: Surface, game: Game, field_to_screen, debug_font) -> None:
        if game.ball is None:
            return
        ball_pos = field_to_screen(game.ball.x, game.ball.y)
        for player, color in [(game.player_a, RED), (game.player_b, BLUE)]:
            ppos = field_to_screen(player.x, player.y)
            pygame.draw.line(screen, color, ppos, ball_pos, 1)
            can_hit = player.can_hit(game.ball)
            text = f"{player.distance_to_ball(game.ball):.1f}" + (" [HIT]" if can_hit else "")
            screen.blit(
                debug_font.render(text, True, CYAN if can_hit else GRAY),
                ((ppos[0] + ball_pos[0]) // 2, (ppos[1] + ball_pos[1]) // 2),
            )


class StatePanelOverlay:
    """Draws game engine internal state (not visible to agents)."""

    def draw(
        self, screen: Surface, game: Game, frame_count: int, x: int, y: int, debug_font
    ) -> None:
        pw, ph = 160, 115
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))

        cy = y + 5

        def line(t, c=WHITE):
            nonlocal cy
            screen.blit(debug_font.render(t, True, c), (x + 5, cy))
            cy += 14

        line("Engine State", CYAN)
        line(f"Frame: {frame_count}")
        line(f"Steps: {game.total_steps}")
        line(f"State: {game.state.value}")
        if game.ball:
            line(f"is_in: {game.ball.is_in}", YELLOW if game.ball.is_in else GRAY)
            lh = game.ball.last_hit_by
            line(f"Last Hit: {['A','B'][lh] if lh is not None else 'None'}")


class RewardGraphOverlay:
    """Draws reward graphs."""

    def __init__(self, max_points: int = 200):
        self.max_points = max_points

    def draw(
        self,
        screen: Surface,
        cumulative_a: List[float],
        cumulative_b: List[float],
        moving_avg_a: List[float],
        moving_avg_b: List[float],
        episode_count: int,
        x: int,
        y: int,
        debug_font,
    ) -> None:
        gw, gh, m = 180, 80, 5

        self._draw_graph(
            screen,
            x,
            y,
            gw,
            gh,
            cumulative_a[-self.max_points :],
            cumulative_b[-self.max_points :],
            "Cumulative (Episode)",
            debug_font,
        )
        self._draw_graph(
            screen,
            x + gw + m,
            y,
            gw,
            gh,
            moving_avg_a,
            moving_avg_b,
            "Reward (20-ep MA)",
            debug_font,
        )

        ly = y + gh + 5
        screen.blit(debug_font.render("■ Player A", True, RED), (x, ly))
        screen.blit(debug_font.render("■ Player B", True, BLUE), (x + 80, ly))
        screen.blit(debug_font.render(f"Episodes: {episode_count}", True, WHITE), (x + gw + m, ly))

    def _draw_graph(
        self,
        screen,
        x,
        y,
        width,
        height,
        data_a: List[float],
        data_b: List[float],
        title: str,
        debug_font,
    ) -> None:
        surf = pygame.Surface((width, height))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))
        pygame.draw.rect(screen, WHITE, (x, y, width, height), 1)
        screen.blit(debug_font.render(title, True, WHITE), (x + 5, y + 2))

        gx, gy, gw, gh = x + 5, y + 18, width - 10, height - 23
        if len(data_a) < 2 and len(data_b) < 2:
            screen.blit(
                debug_font.render("No data", True, GRAY), (x + width // 2 - 20, y + height // 2)
            )
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
            pts = [
                (
                    gx + int(i / (len(data) - 1) * gw),
                    max(gy, min(gy + gh, gy + gh - int((v - min_v) / rng * gh))),
                )
                for i, v in enumerate(data)
            ]
            pygame.draw.lines(screen, color, False, pts, 2)

        draw_series(data_a, RED)
        draw_series(data_b, BLUE)
        if data_a:
            screen.blit(
                debug_font.render(f"A:{data_a[-1]:.2f}", True, RED), (x + width - 60, y + 2)
            )
        if data_b:
            screen.blit(
                debug_font.render(f"B:{data_b[-1]:.2f}", True, BLUE), (x + width - 60, y + 12)
            )


class ObservationOverlay:
    """Draws raw observation data passed to agents."""

    def draw(
        self, screen: Surface, obs: Optional[Dict[str, Any]], x: int, y: int, debug_font
    ) -> None:
        if obs is None:
            return
        pw, ph = 200, 125
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))

        cy = y + 3

        def line(t, c=WHITE):
            nonlocal cy
            screen.blit(debug_font.render(t, True, c), (x + 5, cy))
            cy += 13

        line("Observation (raw)", CYAN)
        line(f"ball: ({obs['ball_x']:.1f}, {obs['ball_y']:.1f})")
        line(f"ball_v: ({obs['ball_vx']:.1f}, {obs['ball_vy']:.1f})")
        line(f"is_in: {obs['ball_is_in']}")
        line(f"A: ({obs['player_a_x']:.0f}, {obs['player_a_y']:.0f})", RED)
        line(f"B: ({obs['player_b_x']:.0f}, {obs['player_b_y']:.0f})", BLUE)
        line(f"score: {obs['score_a']}-{obs['score_b']} rally: {obs['rally_count']}")
        line(f"field: {obs['field_width']}x{obs['field_height']}")

    def draw_minimap(
        self, screen: Surface, obs: Optional[Dict[str, Any]], x: int, y: int, debug_font
    ) -> None:
        """Draw minimap visualization of obs data only."""
        if obs is None:
            return

        # Fixed scale: same for x and y to preserve aspect ratio
        scale = 0.2  # 1 obs unit = 0.2 pixels
        fw, fh = obs["field_width"], obs["field_height"]
        mw, mh = int(fw * scale), int(fh * scale)

        # Background
        surf = pygame.Surface((mw + 4, mh + 20))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x - 2, y - 2))

        # Label
        screen.blit(debug_font.render("Obs Minimap", True, CYAN), (x, y))

        # Field area (offset for label)
        my = y + 16
        pygame.draw.rect(screen, (20, 60, 20), (x, my, mw, mh))
        pygame.draw.rect(screen, WHITE, (x, my, mw, mh), 1)

        # Center line
        pygame.draw.line(screen, GRAY, (x + mw // 2, my), (x + mw // 2, my + mh), 1)

        # Ball position
        bx = x + int(obs["ball_x"] * scale)
        by = my + int(obs["ball_y"] * scale)
        ball_color = YELLOW if obs["ball_is_in"] else (100, 100, 0)
        pygame.draw.circle(screen, ball_color, (bx, by), 3)

        # Ball velocity vector
        vx, vy = obs["ball_vx"], obs["ball_vy"]
        if abs(vx) > 0.1 or abs(vy) > 0.1:
            # Scale velocity for visibility (multiply by scale and a factor)
            arrow_scale = scale * 15
            end_x = bx + int(vx * arrow_scale)
            end_y = by + int(vy * arrow_scale)
            pygame.draw.line(screen, ball_color, (bx, by), (end_x, end_y), 1)

        # Player A
        ax = x + int(obs["player_a_x"] * scale)
        ay = my + int(obs["player_a_y"] * scale)
        pygame.draw.circle(screen, RED, (ax, ay), 4)

        # Player B
        bpx = x + int(obs["player_b_x"] * scale)
        bpy = my + int(obs["player_b_y"] * scale)
        pygame.draw.circle(screen, BLUE, (bpx, bpy), 4)


class ActionOverlay:
    """Draws action data - both text panel and visual controller display."""

    def draw_text(
        self,
        screen: Surface,
        action_a: Optional[Tuple[int, float]],
        action_b: Optional[Tuple[int, float]],
        x: int,
        y: int,
        debug_font,
    ) -> None:
        """Draw text-based action display for bottom panel."""
        pw, ph = 150, 60
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))

        cy = y + 3

        def line(t, c=WHITE):
            nonlocal cy
            screen.blit(debug_font.render(t, True, c), (x + 5, cy))
            cy += 13

        line("Actions (raw)", CYAN)
        if action_a:
            line(f"A: mv={action_a[0]} ang={action_a[1]:.1f}", RED)
        else:
            line("A: None", GRAY)
        if action_b:
            line(f"B: mv={action_b[0]} ang={action_b[1]:.1f}", BLUE)
        else:
            line("B: None", GRAY)

    def draw_controller(
        self,
        screen: Surface,
        action: Optional[Tuple[int, float]],
        x: int,
        y: int,
        color: Tuple[int, int, int],
        label: str,
        debug_font,
    ) -> None:
        """Draw controller-style visual input display."""
        # Background
        pw, ph = 70, 120
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(180)
        surf.fill(BLACK)
        screen.blit(surf, (x - pw // 2, y - 10))

        # Label
        screen.blit(debug_font.render(label, True, color), (x - 5, y - 5))

        # --- Movement stick (octagon with ball) ---
        stick_y = y + 35
        stick_radius = 25
        ball_radius = 6

        # Draw octagon guide
        octagon_pts = []
        for i in range(8):
            angle = math.radians(i * 45 - 90)  # Start from top
            ox = x + int(stick_radius * math.cos(angle))
            oy = stick_y + int(stick_radius * math.sin(angle))
            octagon_pts.append((ox, oy))
        pygame.draw.polygon(screen, GRAY, octagon_pts, 1)

        # Draw center cross
        pygame.draw.line(screen, GRAY, (x - 8, stick_y), (x + 8, stick_y), 1)
        pygame.draw.line(screen, GRAY, (x, stick_y - 8), (x, stick_y + 8), 1)

        # Draw ball position based on movement direction
        if action is not None:
            move_dir = action[0]
            if move_dir < 16:
                # Direction 0-15: 22.5 degree increments, 0 = right
                angle = math.radians(move_dir * 22.5)
                bx = x + int((stick_radius - 5) * math.cos(angle))
                by = stick_y + int((stick_radius - 5) * math.sin(angle))
            else:
                # 16 = stay (center)
                bx, by = x, stick_y
            pygame.draw.circle(screen, color, (bx, by), ball_radius)
        else:
            # No action - gray center
            pygame.draw.circle(screen, GRAY, (x, stick_y), ball_radius)

        # --- Hit angle arrow ---
        arrow_y = y + 85
        arrow_length = 20

        # Draw base circle
        pygame.draw.circle(screen, GRAY, (x, arrow_y), 15, 1)

        if action is not None:
            hit_angle = action[1]
            # Convert angle to radians (0 = right, counterclockwise)
            angle_rad = math.radians(hit_angle)
            end_x = x + int(arrow_length * math.cos(angle_rad))
            end_y = arrow_y - int(arrow_length * math.sin(angle_rad))  # Y inverted

            # Draw arrow line
            pygame.draw.line(screen, color, (x, arrow_y), (end_x, end_y), 2)

            # Draw arrowhead
            head_angle1 = angle_rad + math.radians(150)
            head_angle2 = angle_rad - math.radians(150)
            head_len = 6
            h1x = end_x + int(head_len * math.cos(head_angle1))
            h1y = end_y - int(head_len * math.sin(head_angle1))
            h2x = end_x + int(head_len * math.cos(head_angle2))
            h2y = end_y - int(head_len * math.sin(head_angle2))
            pygame.draw.line(screen, color, (end_x, end_y), (h1x, h1y), 2)
            pygame.draw.line(screen, color, (end_x, end_y), (h2x, h2y), 2)


class RewardOverlay:
    """Draws raw reward values."""

    def draw(
        self, screen: Surface, rewards: Optional[Tuple[float, float]], x: int, y: int, debug_font
    ) -> None:
        pw, ph = 140, 55
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        screen.blit(surf, (x, y))

        cy = y + 3

        def line(t, c=WHITE):
            nonlocal cy
            screen.blit(debug_font.render(t, True, c), (x + 5, cy))
            cy += 13

        line("Rewards (raw)", CYAN)
        if rewards:
            line(f"A: {rewards[0]:.4f}", RED)
            line(f"B: {rewards[1]:.4f}", BLUE)
        else:
            line("None", GRAY)


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
        self.debug_panel_height = 120  # Bottom panel for debug info
        self.window_width = self.config.field_width + 2 * self.padding
        self.window_height = (
            self.config.field_height + 2 * self.padding + self.ui_height + self.debug_panel_height
        )

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
        self.observation_overlay = ObservationOverlay()
        self.action_overlay = ActionOverlay()
        self.reward_overlay = RewardOverlay()

    def render(
        self,
        game: Game,
        total_wins: Optional[Tuple[int, int]] = None,
        stats=None,
        input_state=None,
        frame_count: int = 0,
        actual_fps: float = 0.0,
        obs: Optional[Dict[str, Any]] = None,
        actions: Optional[Tuple[Tuple[int, float], Tuple[int, float]]] = None,
        rewards: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Render game with debug overlays.

        Args:
            game: Current game state
            total_wins: (wins_a, wins_b) tuple
            stats: StatsTracker instance for graphs
            input_state: InputState for display toggles
            frame_count: Current frame number
            actual_fps: Measured FPS for display
            obs: Raw observation dict passed to agents
            actions: (action_a, action_b) raw action tuples from agents
            rewards: (reward_a, reward_b) raw reward values
        """
        # Base rendering
        self._draw_field(game)

        # Trajectory (before ball so trail is behind)
        if input_state and input_state.show_trajectory:
            self.trajectory.draw(self.screen, self.config, self.field_to_screen)

        self._draw_ball(game)
        self._draw_players(game)

        # Controller input displays (left and right edges)
        if actions:
            field_center_y = self.padding + self.ui_height + self.config.field_height // 2
            # Player A - left side
            self.action_overlay.draw_controller(
                self.screen,
                actions[0],
                self.padding - 10,
                field_center_y,
                RED,
                "A",
                self.debug_font,
            )
            # Player B - right side
            self.action_overlay.draw_controller(
                self.screen,
                actions[1],
                self.window_width - self.padding + 10,
                field_center_y,
                BLUE,
                "B",
                self.debug_font,
            )

        # Distance overlay
        if input_state and input_state.show_distances:
            self.distance.draw(self.screen, game, self.field_to_screen, self.debug_font)

        # State panel
        if input_state and input_state.show_state_panel:
            self.state_panel.draw(
                self.screen,
                game,
                frame_count,
                10,
                self.padding + self.ui_height + 10,
                self.debug_font,
            )

        # Event log
        if stats and stats.event_log:
            self._draw_event_log(stats.event_log)

        # Debug panel Y position (below field)
        debug_panel_y = self.padding + self.ui_height + self.config.field_height + self.padding + 5

        # Reward graphs (bottom panel, left side)
        if input_state and input_state.show_graphs and stats:
            gx = 10
            gy = debug_panel_y
            self.reward_graph.draw(
                self.screen,
                stats.cumulative_rewards_a,
                stats.cumulative_rewards_b,
                stats.get_moving_averages(stats.episode_rewards_a),
                stats.get_moving_averages(stats.episode_rewards_b),
                stats.episode_count,
                gx,
                gy,
                self.debug_font,
            )

        # Raw data overlays (bottom panel: obs, action, reward side by side)
        if input_state and input_state.show_state_panel:
            self.observation_overlay.draw(self.screen, obs, 380, debug_panel_y, self.debug_font)
            self.action_overlay.draw_text(
                self.screen,
                actions[0] if actions else None,
                actions[1] if actions else None,
                590,
                debug_panel_y,
                self.debug_font,
            )
            self.reward_overlay.draw(self.screen, rewards, 750, debug_panel_y, self.debug_font)

        # Obs minimap (top-right of field area)
        if input_state and input_state.show_state_panel:
            minimap_x = self.window_width - self.padding - 170
            minimap_y = self.padding + self.ui_height + 10
            self.observation_overlay.draw_minimap(
                self.screen, obs, minimap_x, minimap_y, self.debug_font
            )

        # Controls help
        self._draw_controls_help()

        # Speed and FPS indicator
        if input_state:
            self._draw_speed_indicator(input_state, actual_fps)

        # UI
        self._draw_ui(game, total_wins)

        # Paused indicator
        if input_state and input_state.paused:
            text = self.font.render("PAUSED", True, ORANGE)
            self.screen.blit(
                text, text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
            )

        pygame.display.flip()

    def update(self, game: Game) -> None:
        """Update overlay state (call each frame when game advances)."""
        self.trajectory.update(game)

    def _draw_event_log(self, event_log: Deque[str]) -> None:
        x, y = self.window_width - 200, self.padding + self.ui_height + 10
        for i, ev in enumerate(event_log):
            alpha = int(255 * (1 - i / len(event_log)))
            self.screen.blit(
                self.debug_font.render(ev, True, (alpha, alpha, alpha)), (x, y + i * 14)
            )

    def _draw_controls_help(self) -> None:
        helps = [
            "Controls:",
            "1-4: Speed",
            "R: Reset",
            "F: FPS",
            "T/D/P/G: Overlays",
            "SPACE: Pause",
            "N: Step",
        ]
        x, y = self.window_width - 120, self.window_height - 115
        for i, t in enumerate(helps):
            self.screen.blit(
                self.debug_font.render(t, True, CYAN if i == 0 else GRAY), (x, y + i * 14)
            )

    def _draw_speed_indicator(self, input_state, actual_fps: float) -> None:
        """Draw speed level and FPS indicator."""
        speed_names = ["1x", "2x", "4x", "MAX"]
        speed_level = input_state.speed_level
        speed_text = speed_names[speed_level]

        # Speed indicator (top-left area, below ball status)
        color = CYAN if speed_level == 0 else ORANGE if speed_level < 3 else RED
        self.screen.blit(
            self.small_font.render(f"Speed: {speed_text}", True, color),
            (self.window_width // 2 - 50, 45),
        )

        # FPS display (if enabled)
        if input_state.show_fps:
            fps_color = GREEN if actual_fps >= 55 else ORANGE if actual_fps >= 30 else RED
            self.screen.blit(
                self.small_font.render(f"FPS: {actual_fps:.0f}", True, fps_color),
                (self.window_width // 2 + 50, 45),
            )

    def _draw_ui(self, game: Game, total_wins: Optional[Tuple[int, int]] = None) -> None:
        if total_wins:
            score = self.font.render(
                f"Total Wins - A: {total_wins[0]}  B: {total_wins[1]}", True, WHITE
            )
        else:
            score = self.font.render(
                f"Player A: {game.scores[0]}  -  Player B: {game.scores[1]}", True, WHITE
            )
        self.screen.blit(score, score.get_rect(center=(self.window_width // 2, 25)))

        is_in = game.ball and game.ball.is_in
        self.screen.blit(
            self.small_font.render(
                f"Ball: {'IN' if is_in else 'OUT'}", True, YELLOW if is_in else GRAY
            ),
            (10, 45),
        )
        self.screen.blit(
            self.small_font.render(f"Rally: {game.rally_count}", True, WHITE),
            (self.window_width - 100, 45),
        )

        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            text = self.font.render(f"Episode Over! Player {winner} wins!", True, WHITE)
            self.screen.blit(
                text, text.get_rect(center=(self.window_width // 2, self.window_height - 30))
            )


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
            game.step(
                (random.randint(0, 16), random.uniform(0, 360)),
                (random.randint(0, 16), random.uniform(0, 360)),
            )
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
            renderer.render(
                game, stats=stats, input_state=input_handler.state, frame_count=frame_count
            )
            renderer.tick()

    renderer.close()


if __name__ == "__main__":
    run_demo_game()
