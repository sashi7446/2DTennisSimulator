"""Pygame renderer for 2D Tennis Simulator."""

from typing import Optional, Tuple, List, Deque
from collections import deque
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import Config
from game import Game, GameState


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)  # Court green
LIGHT_GREEN = (144, 238, 144)  # In-area green
YELLOW = (255, 255, 0)  # Ball
RED = (255, 100, 100)  # Player A
BLUE = (100, 100, 255)  # Player B
GRAY = (128, 128, 128)  # Wall
DARK_GREEN = (0, 100, 0)  # Lines
ORANGE = (255, 165, 0)  # Debug highlight
CYAN = (0, 255, 255)  # Debug info


class Renderer:
    """
    Basic game renderer - shows single episode state.

    This renderer displays the current episode only:
    - The field with walls and in-areas
    - Players and ball
    - Episode score (always 0-1, since 1 episode = 1 point)
    - In-flag indicator

    Note: For cumulative multi-episode tracking, use DebugRenderer.
    """

    def __init__(self, config: Optional[Config] = None):
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )

        self.config = config or Config()

        # Calculate window size with padding for UI
        self.padding = 50
        self.ui_height = 60
        self.window_width = self.config.field_width + 2 * self.padding
        self.window_height = self.config.field_height + 2 * self.padding + self.ui_height

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2D Tennis Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self._initialized = True

    def _field_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert field coordinates to screen coordinates."""
        import math
        # Handle NaN/Inf values gracefully
        if math.isnan(x) or math.isinf(x):
            x = 0.0
        if math.isnan(y) or math.isinf(y):
            y = 0.0
        screen_x = int(x + self.padding)
        screen_y = int(y + self.padding + self.ui_height)
        return (screen_x, screen_y)

    def _draw_field(self, game: Game) -> None:
        """Draw the field, walls, and in-areas."""
        # Draw background
        self.screen.fill(GRAY)

        # Draw court area (inside walls)
        field_rect = pygame.Rect(
            self.padding,
            self.padding + self.ui_height,
            self.config.field_width,
            self.config.field_height,
        )
        pygame.draw.rect(self.screen, GREEN, field_rect)

        # Draw in-areas
        area_a = game.field.area_a
        area_b = game.field.area_b

        area_a_rect = pygame.Rect(
            *self._field_to_screen(area_a.x, area_a.y),
            int(area_a.width),
            int(area_a.height),
        )
        area_b_rect = pygame.Rect(
            *self._field_to_screen(area_b.x, area_b.y),
            int(area_b.width),
            int(area_b.height),
        )

        pygame.draw.rect(self.screen, LIGHT_GREEN, area_a_rect)
        pygame.draw.rect(self.screen, LIGHT_GREEN, area_b_rect)

        # Draw area borders
        pygame.draw.rect(self.screen, WHITE, area_a_rect, 2)
        pygame.draw.rect(self.screen, WHITE, area_b_rect, 2)

        # Draw center line
        center_x, _ = self._field_to_screen(self.config.field_width / 2, 0)
        pygame.draw.line(
            self.screen,
            WHITE,
            (center_x, self.padding + self.ui_height),
            (center_x, self.padding + self.ui_height + self.config.field_height),
            1,
        )

        # Draw wall border
        pygame.draw.rect(self.screen, WHITE, field_rect, 3)

    def _draw_ball(self, game: Game) -> None:
        """Draw the ball with in-flag indicator."""
        if game.ball is None:
            return

        ball_screen = self._field_to_screen(game.ball.x, game.ball.y)

        # Ball color changes based on in-flag
        ball_color = YELLOW if game.ball.in_flag else (200, 200, 0)

        # Draw ball
        pygame.draw.circle(
            self.screen, ball_color, ball_screen, game.ball.radius
        )

    def _draw_players(self, game: Game) -> None:
        """Draw both players."""
        # Player A (red)
        player_a_screen = self._field_to_screen(game.player_a.x, game.player_a.y)
        pygame.draw.circle(
            self.screen, RED, player_a_screen, int(game.player_a.radius)
        )
        pygame.draw.circle(
            self.screen, WHITE, player_a_screen, int(game.player_a.radius), 2
        )

        # Draw reach indicator for player A
        pygame.draw.circle(
            self.screen,
            (255, 150, 150),
            player_a_screen,
            int(game.player_a.reach_distance),
            1,
        )

        # Player B (blue)
        player_b_screen = self._field_to_screen(game.player_b.x, game.player_b.y)
        pygame.draw.circle(
            self.screen, BLUE, player_b_screen, int(game.player_b.radius)
        )
        pygame.draw.circle(
            self.screen, WHITE, player_b_screen, int(game.player_b.radius), 2
        )

        # Draw reach indicator for player B
        pygame.draw.circle(
            self.screen,
            (150, 150, 255),
            player_b_screen,
            int(game.player_b.reach_distance),
            1,
        )

    def _draw_ui(self, game: Game) -> None:
        """Draw the score and game state UI (episode-level only)."""
        # Episode score display (always 0-1 since 1 episode = 1 point)
        score_text = f"Player A: {game.scores[0]}  -  Player B: {game.scores[1]}"
        score_surface = self.font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(
            center=(self.window_width // 2, 25)
        )
        self.screen.blit(score_surface, score_rect)

        # In-flag indicator
        in_flag_text = "IN" if (game.ball and game.ball.in_flag) else "OUT"
        in_flag_color = YELLOW if (game.ball and game.ball.in_flag) else GRAY
        in_flag_surface = self.small_font.render(
            f"Ball: {in_flag_text}", True, in_flag_color
        )
        self.screen.blit(in_flag_surface, (10, 45))

        # Rally count
        rally_text = f"Rally: {game.rally_count}"
        rally_surface = self.small_font.render(rally_text, True, WHITE)
        self.screen.blit(rally_surface, (self.window_width - 100, 45))

        # Game state
        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            winner_text = f"Game Over! Player {winner} wins!"
            winner_surface = self.font.render(winner_text, True, WHITE)
            winner_rect = winner_surface.get_rect(
                center=(self.window_width // 2, self.window_height // 2)
            )
            # Draw background box
            pygame.draw.rect(
                self.screen,
                BLACK,
                winner_rect.inflate(20, 10),
            )
            self.screen.blit(winner_surface, winner_rect)

    def render(self, game: Game) -> None:
        """
        Render the current game state.

        Args:
            game: The game to render
        """
        self._draw_field(game)
        self._draw_ball(game)
        self._draw_players(game)
        self._draw_ui(game)

        pygame.display.flip()

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            False if window should close, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def tick(self, fps: Optional[int] = None) -> None:
        """Limit frame rate."""
        self.clock.tick(fps or self.config.fps)

    def close(self) -> None:
        """Close the renderer and pygame."""
        pygame.quit()
        self._initialized = False


class DebugRenderer(Renderer):
    """
    Training/Debug renderer - shows cumulative multi-episode statistics.

    Extends the basic Renderer with training-focused features:
    - **Cumulative wins across all episodes** (not just current episode)
    - Ball trajectory (past positions)
    - In-flag state changes
    - Hit detection zones
    - Detailed state information
    - Frame-by-frame data
    - Reward graphs (cumulative and moving average)

    Use this renderer when:
    - Training agents and need to track progress over many episodes
    - Debugging game mechanics
    - Analyzing agent behavior over time
    """

    def __init__(self, config: Optional[Config] = None, trail_length: int = 60):
        # Don't call super().__init__() yet - we need to override window size first
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )

        self.config = config or Config()

        # Calculate window size with padding for UI AND debug canvas
        self.padding = 50
        self.ui_height = 60
        self.debug_canvas_height = 300  # Additional debug canvas at the bottom
        self.window_width = self.config.field_width + 2 * self.padding
        self.window_height = (
            self.config.field_height + 2 * self.padding + self.ui_height + self.debug_canvas_height
        )

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2D Tennis Simulator [DEBUG MODE]")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self._initialized = True

        # Ball trajectory trail
        self.ball_trail: Deque[Tuple[float, float, bool]] = deque(maxlen=trail_length)

        # Event log for on-screen display
        self.event_log: Deque[str] = deque(maxlen=8)

        # State tracking for change detection
        self.last_in_flag = False
        self.last_scores = [0, 0]
        self.frame_count = 0

        # Track cumulative wins across episodes
        self.total_wins = [0, 0]  # [Player A, Player B]

        # Agent-view debug tracking
        self.agent_a_state = None  # Latest state seen by agent A
        self.agent_b_state = None  # Latest state seen by agent B
        self.agent_a_action = None  # Latest action from agent A
        self.agent_b_action = None  # Latest action from agent B
        self.agent_a_reward = None  # Latest reward for agent A
        self.agent_b_reward = None  # Latest reward for agent B

        # Debug panel settings
        self.debug_font = pygame.font.Font(None, 18)
        self.show_trajectory = True
        self.show_distances = False  # Disabled by default (toggle with D key)
        self.show_state_panel = True
        self.show_graphs = True
        self.paused = False
        self.step_mode = False

        # Reward tracking
        self.cumulative_rewards_a: List[float] = []  # Current episode
        self.cumulative_rewards_b: List[float] = []  # Current episode
        self.current_cumulative_a: float = 0.0
        self.current_cumulative_b: float = 0.0

        # Episode history for moving average (stores total reward per episode)
        self.episode_rewards_a: List[float] = []
        self.episode_rewards_b: List[float] = []
        self.moving_avg_window: int = 20  # 20-episode moving average

        # Graph settings
        self.graph_max_points: int = 200  # Max points to show in cumulative graph

    def _draw_ball_trajectory(self, game: Game) -> None:
        """Draw the ball's recent trajectory."""
        if not self.show_trajectory or len(self.ball_trail) < 2:
            return

        # Draw trail with fading effect
        trail_list = list(self.ball_trail)
        for i in range(1, len(trail_list)):
            prev_x, prev_y, prev_in_flag = trail_list[i - 1]
            curr_x, curr_y, curr_in_flag = trail_list[i]

            # Fade based on age
            alpha = int(255 * (i / len(trail_list)))

            # Color based on in_flag state
            if curr_in_flag:
                color = (alpha, alpha, 0)  # Yellow when in
            else:
                color = (alpha // 2, alpha // 2, 0)  # Dim when out

            start = self._field_to_screen(prev_x, prev_y)
            end = self._field_to_screen(curr_x, curr_y)
            pygame.draw.line(self.screen, color, start, end, 2)

        # Mark in_flag state change points
        for i, (x, y, in_flag) in enumerate(trail_list):
            if i > 0:
                _, _, prev_in_flag = trail_list[i - 1]
                if in_flag != prev_in_flag:
                    pos = self._field_to_screen(x, y)
                    color = ORANGE if in_flag else RED
                    pygame.draw.circle(self.screen, color, pos, 6, 2)

    def _draw_distances(self, game: Game) -> None:
        """Draw distance lines from players to ball."""
        if not self.show_distances or game.ball is None:
            return

        ball_pos = self._field_to_screen(game.ball.x, game.ball.y)

        for player, color in [(game.player_a, RED), (game.player_b, BLUE)]:
            player_pos = self._field_to_screen(player.x, player.y)
            distance = player.distance_to_ball(game.ball)

            # Draw line to ball
            pygame.draw.line(self.screen, color, player_pos, ball_pos, 1)

            # Draw distance text
            mid_x = (player_pos[0] + ball_pos[0]) // 2
            mid_y = (player_pos[1] + ball_pos[1]) // 2

            can_hit = player.can_hit(game.ball)
            dist_color = CYAN if can_hit else GRAY

            dist_text = f"{distance:.1f}"
            if can_hit:
                dist_text += " [HIT]"
            dist_surface = self.debug_font.render(dist_text, True, dist_color)
            self.screen.blit(dist_surface, (mid_x, mid_y))

    def _draw_state_panel(self, game: Game) -> None:
        """Draw detailed state information panel."""
        if not self.show_state_panel:
            return

        # Panel background
        panel_x = 10
        panel_y = self.padding + self.ui_height + 10
        panel_width = 180
        panel_height = 200

        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill(BLACK)
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # Draw state information
        y_offset = panel_y + 5
        line_height = 16

        def draw_line(text: str, color=WHITE):
            nonlocal y_offset
            surface = self.debug_font.render(text, True, color)
            self.screen.blit(surface, (panel_x + 5, y_offset))
            y_offset += line_height

        draw_line(f"Frame: {self.frame_count}", CYAN)
        draw_line(f"Steps: {game.total_steps}")
        draw_line(f"State: {game.state.value}")
        draw_line("---")

        if game.ball:
            draw_line("Ball:", YELLOW)
            draw_line(f"  Pos: ({game.ball.x:.1f}, {game.ball.y:.1f})")
            draw_line(f"  Vel: ({game.ball.vx:.1f}, {game.ball.vy:.1f})")
            in_flag_color = YELLOW if game.ball.in_flag else GRAY
            draw_line(f"  In-Flag: {game.ball.in_flag}", in_flag_color)
            last_hit = game.ball.last_hit_by
            draw_line(f"  Last Hit: {['A', 'B'][last_hit] if last_hit is not None else 'None'}")

        draw_line("---")
        draw_line("Players:")
        draw_line(f"  A: ({game.player_a.x:.0f}, {game.player_a.y:.0f})", RED)
        draw_line(f"  B: ({game.player_b.x:.0f}, {game.player_b.y:.0f})", BLUE)

    def _draw_event_log(self) -> None:
        """Draw recent events log."""
        if not self.event_log:
            return

        x = self.window_width - 200
        y = self.padding + self.ui_height + 10

        for i, event in enumerate(self.event_log):
            alpha = int(255 * (1 - i / len(self.event_log)))
            color = (alpha, alpha, alpha)
            surface = self.debug_font.render(event, True, color)
            self.screen.blit(surface, (x, y + i * 14))

    def _draw_controls_help(self) -> None:
        """Draw keyboard controls help."""
        help_text = [
            "Debug Controls:",
            "T - Toggle trajectory",
            "D - Toggle distances",
            "P - Toggle state panel",
            "G - Toggle graphs",
            "SPACE - Pause/Resume",
            "N - Step (when paused)",
        ]

        x = self.window_width - 150
        y = self.window_height - 120

        for i, text in enumerate(help_text):
            color = CYAN if i == 0 else GRAY
            surface = self.debug_font.render(text, True, color)
            self.screen.blit(surface, (x, y + i * 14))

    def _draw_graph(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data_a: List[float],
        data_b: List[float],
        title: str,
        show_zero_line: bool = True,
    ) -> None:
        """Draw a line graph with two data series."""
        # Background
        graph_surface = pygame.Surface((width, height))
        graph_surface.set_alpha(200)
        graph_surface.fill(BLACK)
        self.screen.blit(graph_surface, (x, y))

        # Border
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height), 1)

        # Title
        title_surface = self.debug_font.render(title, True, WHITE)
        self.screen.blit(title_surface, (x + 5, y + 2))

        # Graph area (with padding for title)
        graph_x = x + 5
        graph_y = y + 18
        graph_w = width - 10
        graph_h = height - 23

        if len(data_a) < 2 and len(data_b) < 2:
            # Not enough data
            no_data = self.debug_font.render("No data", True, GRAY)
            self.screen.blit(no_data, (x + width // 2 - 20, y + height // 2))
            return

        # Find min/max for scaling
        all_data = data_a + data_b
        if not all_data:
            return
        min_val = min(all_data)
        max_val = max(all_data)

        # Add some padding to range
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        min_val -= range_val * 0.1
        max_val += range_val * 0.1
        range_val = max_val - min_val

        # Draw zero line if in range
        if show_zero_line and min_val < 0 < max_val:
            zero_y = graph_y + graph_h - int((0 - min_val) / range_val * graph_h)
            pygame.draw.line(
                self.screen, GRAY, (graph_x, zero_y), (graph_x + graph_w, zero_y), 1
            )

        def draw_line_series(data: List[float], color: Tuple[int, int, int]) -> None:
            if len(data) < 2:
                return
            points = []
            for i, val in enumerate(data):
                px = graph_x + int(i / (len(data) - 1) * graph_w)
                py = graph_y + graph_h - int((val - min_val) / range_val * graph_h)
                py = max(graph_y, min(graph_y + graph_h, py))
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)

        # Draw both series
        draw_line_series(data_a, RED)
        draw_line_series(data_b, BLUE)

        # Draw current values
        if data_a:
            val_a = self.debug_font.render(f"A:{data_a[-1]:.2f}", True, RED)
            self.screen.blit(val_a, (x + width - 60, y + 2))
        if data_b:
            val_b = self.debug_font.render(f"B:{data_b[-1]:.2f}", True, BLUE)
            self.screen.blit(val_b, (x + width - 60, y + 12))

    def _draw_reward_graphs(self) -> None:
        """Draw 2 reward graphs: cumulative (current episode) and moving average."""
        if not self.show_graphs:
            return

        # Graph dimensions
        graph_width = 180
        graph_height = 80
        margin = 5

        # Position graphs at the bottom of the field area
        base_x = self.padding + 200
        base_y = self.padding + self.ui_height + self.config.field_height - graph_height * 2 - margin * 2

        # Left graph: Cumulative rewards (current episode) - shows reward flow in real-time
        self._draw_graph(
            base_x,
            base_y,
            graph_width,
            graph_height,
            self.cumulative_rewards_a[-self.graph_max_points:],
            self.cumulative_rewards_b[-self.graph_max_points:],
            "Cumulative (Episode)",
        )

        # Right graph: Episode rewards with moving average - shows all episodes
        self._draw_graph(
            base_x + graph_width + margin,
            base_y,
            graph_width,
            graph_height,
            self._get_moving_averages(self.episode_rewards_a),
            self._get_moving_averages(self.episode_rewards_b),
            f"Reward ({self.moving_avg_window}-ep MA)",
        )

        # Labels for Player A and B
        label_y = base_y + graph_height + 5
        label_a = self.debug_font.render("■ Player A", True, RED)
        label_b = self.debug_font.render("■ Player B", True, BLUE)
        self.screen.blit(label_a, (base_x, label_y))
        self.screen.blit(label_b, (base_x + 80, label_y))

        # Episode count
        ep_text = f"Episodes: {len(self.episode_rewards_a)}"
        ep_surface = self.debug_font.render(ep_text, True, WHITE)
        self.screen.blit(ep_surface, (base_x + graph_width + margin, label_y))

    def _get_moving_averages(self, rewards: List[float]) -> List[float]:
        """Calculate moving average over episodes."""
        if len(rewards) < 1:
            return []

        result = []
        for i in range(len(rewards)):
            start = max(0, i - self.moving_avg_window + 1)
            window = rewards[start : i + 1]
            result.append(sum(window) / len(window))
        return result

    def add_reward(self, reward_a: float, reward_b: float) -> None:
        """Add rewards from the current step."""
        self.current_cumulative_a += reward_a
        self.current_cumulative_b += reward_b
        self.cumulative_rewards_a.append(self.current_cumulative_a)
        self.cumulative_rewards_b.append(self.current_cumulative_b)

    def end_episode(self, winner: Optional[int] = None) -> None:
        """
        Call when an episode (game) ends to record totals.

        Args:
            winner: 0 for Player A, 1 for Player B, None if no winner
        """
        if self.cumulative_rewards_a or self.cumulative_rewards_b:
            self.episode_rewards_a.append(self.current_cumulative_a)
            self.episode_rewards_b.append(self.current_cumulative_b)

        # Track cumulative wins
        if winner is not None:
            self.total_wins[winner] += 1

        # Reset for next episode
        self.current_cumulative_a = 0.0
        self.current_cumulative_b = 0.0
        self.cumulative_rewards_a = []
        self.cumulative_rewards_b = []

    def reset_episode_rewards(self) -> None:
        """Reset cumulative rewards for a new episode (but keep history)."""
        self.current_cumulative_a = 0.0
        self.current_cumulative_b = 0.0
        self.cumulative_rewards_a = []
        self.cumulative_rewards_b = []

    def _check_state_changes(self, game: Game) -> None:
        """Check for state changes and log them."""
        if game.ball:
            # In-flag change
            if game.ball.in_flag != self.last_in_flag:
                if game.ball.in_flag:
                    self.event_log.appendleft(f"F{self.frame_count}: IN-FLAG ON")
                else:
                    self.event_log.appendleft(f"F{self.frame_count}: IN-FLAG OFF (hit)")
                self.last_in_flag = game.ball.in_flag

        # Score change
        if game.scores != self.last_scores:
            diff_a = game.scores[0] - self.last_scores[0]
            diff_b = game.scores[1] - self.last_scores[1]
            if diff_a > 0:
                self.event_log.appendleft(f"F{self.frame_count}: Point to A")
            if diff_b > 0:
                self.event_log.appendleft(f"F{self.frame_count}: Point to B")
            self.last_scores = game.scores.copy()

    def update(self, game: Game) -> None:
        """Update debug state (call before render)."""
        import math
        self.frame_count += 1

        # Record ball position for trail (skip NaN values)
        if game.ball:
            x, y = game.ball.x, game.ball.y
            if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                self.ball_trail.append((x, y, game.ball.in_flag))

        # Check for state changes
        self._check_state_changes(game)

    def _draw_ui(self, game: Game) -> None:
        """Draw UI with cumulative wins (overrides base Renderer)."""
        # Cumulative wins display (total across all episodes)
        score_text = f"Total Wins - Player A: {self.total_wins[0]}  Player B: {self.total_wins[1]}"
        score_surface = self.font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(
            center=(self.window_width // 2, 25)
        )
        self.screen.blit(score_surface, score_rect)

        # In-flag indicator
        in_flag_text = "IN" if (game.ball and game.ball.in_flag) else "OUT"
        in_flag_color = YELLOW if (game.ball and game.ball.in_flag) else GRAY
        in_flag_surface = self.small_font.render(
            f"Ball: {in_flag_text}", True, in_flag_color
        )
        self.screen.blit(in_flag_surface, (10, 45))

        # Rally count
        rally_text = f"Rally: {game.rally_count}"
        rally_surface = self.small_font.render(rally_text, True, WHITE)
        self.screen.blit(rally_surface, (self.window_width - 100, 45))

        # Game state
        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            winner_text = f"Episode Over! Player {winner} wins!"
            winner_surface = self.font.render(winner_text, True, WHITE)
            winner_rect = winner_surface.get_rect(
                center=(self.window_width // 2, self.window_height - 30)
            )
            self.screen.blit(winner_surface, winner_rect)

    def render(self, game: Game) -> None:
        """Render with debug overlays."""
        # Base rendering
        self._draw_field(game)

        # Debug overlays (behind ball/players)
        self._draw_ball_trajectory(game)

        # Ball and players
        self._draw_ball(game)
        self._draw_players(game)

        # More debug overlays (on top)
        self._draw_distances(game)
        self._draw_state_panel(game)
        self._draw_event_log()
        self._draw_reward_graphs()
        self._draw_controls_help()

        # UI
        self._draw_ui(game)

        # Pause indicator
        if self.paused:
            pause_surface = self.font.render("PAUSED", True, ORANGE)
            pause_rect = pause_surface.get_rect(
                center=(self.window_width // 2, self.window_height // 2 - 50)
            )
            self.screen.blit(pause_surface, pause_rect)

        pygame.display.flip()

    def handle_events(self) -> bool:
        """Handle events including debug controls."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_t:
                    self.show_trajectory = not self.show_trajectory
                elif event.key == pygame.K_d:
                    self.show_distances = not self.show_distances
                elif event.key == pygame.K_p:
                    self.show_state_panel = not self.show_state_panel
                elif event.key == pygame.K_g:
                    self.show_graphs = not self.show_graphs
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n:
                    self.step_mode = True
        return True

    def should_step(self) -> bool:
        """Check if game should advance (for pause/step mode)."""
        if not self.paused:
            return True
        if self.step_mode:
            self.step_mode = False
            return True
        return False


def run_demo_game() -> None:
    """Run a demo game with random AI players."""
    import random

    config = Config()
    game = Game(config)
    renderer = Renderer(config)

    running = True
    while running and not game.is_game_over:
        # Handle events
        running = renderer.handle_events()

        # Random actions for demo
        action_a = (
            random.randint(0, 16),  # Random movement
            random.uniform(0, 360),  # Random hit angle
        )
        action_b = (
            random.randint(0, 16),
            random.uniform(0, 360),
        )

        # Step game
        result = game.step(action_a, action_b)

        # Render
        renderer.render(game)
        renderer.tick()

    # Show final state for a moment
    if game.is_game_over:
        for _ in range(180):  # 3 seconds at 60fps
            renderer.handle_events()
            renderer.render(game)
            renderer.tick()

    renderer.close()


if __name__ == "__main__":
    run_demo_game()
