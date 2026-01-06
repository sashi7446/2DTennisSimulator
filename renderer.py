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
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, LIGHT_GREEN = (34, 139, 34), (144, 238, 144)
YELLOW, RED, BLUE = (255, 255, 0), (255, 100, 100), (100, 100, 255)
GRAY, ORANGE, CYAN = (128, 128, 128), (255, 165, 0), (0, 255, 255)


class Renderer:
    """Basic renderer for single episode display."""

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
        self._initialized = True

    def _field_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        x = 0.0 if (math.isnan(x) or math.isinf(x)) else x
        y = 0.0 if (math.isnan(y) or math.isinf(y)) else y
        return (int(x + self.padding), int(y + self.padding + self.ui_height))

    def _draw_field(self, game: Game) -> None:
        self.screen.fill(GRAY)
        field_rect = pygame.Rect(self.padding, self.padding + self.ui_height,
                                  self.config.field_width, self.config.field_height)
        pygame.draw.rect(self.screen, GREEN, field_rect)

        for area in [game.field.area_a, game.field.area_b]:
            rect = pygame.Rect(*self._field_to_screen(area.x, area.y), int(area.width), int(area.height))
            pygame.draw.rect(self.screen, LIGHT_GREEN, rect)
            pygame.draw.rect(self.screen, WHITE, rect, 2)

        center_x = self._field_to_screen(self.config.field_width / 2, 0)[0]
        pygame.draw.line(self.screen, WHITE, (center_x, self.padding + self.ui_height),
                         (center_x, self.padding + self.ui_height + self.config.field_height), 1)
        pygame.draw.rect(self.screen, WHITE, field_rect, 3)

    def _draw_ball(self, game: Game) -> None:
        if game.ball is None:
            return
        pos = self._field_to_screen(game.ball.x, game.ball.y)
        color = YELLOW if game.ball.in_flag else (200, 200, 0)
        pygame.draw.circle(self.screen, color, pos, game.ball.radius)

    def _draw_players(self, game: Game) -> None:
        for player, color, reach_color in [
            (game.player_a, RED, (255, 150, 150)),
            (game.player_b, BLUE, (150, 150, 255))
        ]:
            pos = self._field_to_screen(player.x, player.y)
            pygame.draw.circle(self.screen, color, pos, int(player.radius))
            pygame.draw.circle(self.screen, WHITE, pos, int(player.radius), 2)
            pygame.draw.circle(self.screen, reach_color, pos, int(player.reach_distance), 1)

    def _draw_ui(self, game: Game) -> None:
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

    def render(self, game: Game) -> None:
        self._draw_field(game)
        self._draw_ball(game)
        self._draw_players(game)
        self._draw_ui(game)
        pygame.display.flip()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True

    def tick(self, fps: Optional[int] = None) -> None:
        self.clock.tick(fps or self.config.fps)

    def close(self) -> None:
        pygame.quit()
        self._initialized = False


class DebugRenderer(Renderer):
    """Training/debug renderer with separate debug canvas below game field."""

    def __init__(self, config: Optional[Config] = None, trail_length: int = 60):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required: pip install pygame")

        self.config = config or Config()
        self.padding, self.ui_height = 50, 60
        self.debug_canvas_height = 300
        self.window_width = self.config.field_width + 2 * self.padding
        self.game_area_height = self.config.field_height + 2 * self.padding + self.ui_height
        self.window_height = self.game_area_height + self.debug_canvas_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2D Tennis Simulator [DEBUG MODE]")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.debug_font = pygame.font.Font(None, 18)
        self._initialized = True

        # Separate surfaces: game and debug
        self.game_surface = pygame.Surface((self.window_width, self.game_area_height))
        self.debug_surface = pygame.Surface((self.window_width, self.debug_canvas_height))

        self.ball_trail: Deque[Tuple[float, float, bool]] = deque(maxlen=trail_length)
        self.event_log: Deque[str] = deque(maxlen=8)
        self.last_in_flag, self.last_scores, self.frame_count = False, [0, 0], 0
        self.total_wins = [0, 0]

        self.show_trajectory, self.show_distances = True, False
        self.show_state_panel, self.show_graphs = True, True
        self.paused, self.step_mode = False, False

        self.cumulative_rewards_a: List[float] = []
        self.cumulative_rewards_b: List[float] = []
        self.current_cumulative_a, self.current_cumulative_b = 0.0, 0.0
        self.episode_rewards_a: List[float] = []
        self.episode_rewards_b: List[float] = []
        self.moving_avg_window, self.graph_max_points = 20, 200

        # Agent debug state (set via step_debug)
        self.agent_obs: Optional[dict] = None
        self.agent_reward: float = 0.0
        self.agent_action: Optional[Tuple[int, float]] = None
        self.agent_player_id: int = 0

    def _draw_ball_trajectory(self, game: Game) -> None:
        if not self.show_trajectory or len(self.ball_trail) < 2:
            return
        trail = list(self.ball_trail)
        for i in range(1, len(trail)):
            alpha = int(255 * i / len(trail))
            color = (alpha, alpha, 0) if trail[i][2] else (alpha // 2, alpha // 2, 0)
            pygame.draw.line(self.screen, color,
                             self._field_to_screen(trail[i-1][0], trail[i-1][1]),
                             self._field_to_screen(trail[i][0], trail[i][1]), 2)
            if i > 0 and trail[i][2] != trail[i-1][2]:
                pygame.draw.circle(self.screen, ORANGE if trail[i][2] else RED,
                                   self._field_to_screen(trail[i][0], trail[i][1]), 6, 2)

    def _draw_distances(self, game: Game) -> None:
        if not self.show_distances or game.ball is None:
            return
        ball_pos = self._field_to_screen(game.ball.x, game.ball.y)
        for player, color in [(game.player_a, RED), (game.player_b, BLUE)]:
            ppos = self._field_to_screen(player.x, player.y)
            pygame.draw.line(self.screen, color, ppos, ball_pos, 1)
            can_hit = player.can_hit(game.ball)
            text = f"{player.distance_to_ball(game.ball):.1f}" + (" [HIT]" if can_hit else "")
            self.screen.blit(self.debug_font.render(text, True, CYAN if can_hit else GRAY),
                             ((ppos[0] + ball_pos[0]) // 2, (ppos[1] + ball_pos[1]) // 2))

    def _draw_state_panel(self, game: Game) -> None:
        if not self.show_state_panel:
            return
        px, py, pw, ph = 10, self.padding + self.ui_height + 10, 180, 200
        surf = pygame.Surface((pw, ph))
        surf.set_alpha(200)
        surf.fill(BLACK)
        self.screen.blit(surf, (px, py))

        y = py + 5
        def line(t, c=WHITE):
            nonlocal y
            self.screen.blit(self.debug_font.render(t, True, c), (px + 5, y))
            y += 16

        line(f"Frame: {self.frame_count}", CYAN)
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

    def _get_moving_averages(self, rewards: List[float]) -> List[float]:
        if not rewards:
            return []
        return [sum(rewards[max(0, i - self.moving_avg_window + 1):i + 1]) / len(rewards[max(0, i - self.moving_avg_window + 1):i + 1])
                for i in range(len(rewards))]

    def add_reward(self, reward_a: float, reward_b: float) -> None:
        self.current_cumulative_a += reward_a
        self.current_cumulative_b += reward_b
        self.cumulative_rewards_a.append(self.current_cumulative_a)
        self.cumulative_rewards_b.append(self.current_cumulative_b)

    def end_episode(self, winner: Optional[int] = None) -> None:
        if self.cumulative_rewards_a or self.cumulative_rewards_b:
            self.episode_rewards_a.append(self.current_cumulative_a)
            self.episode_rewards_b.append(self.current_cumulative_b)
        if winner is not None:
            self.total_wins[winner] += 1
        self.current_cumulative_a = self.current_cumulative_b = 0.0
        self.cumulative_rewards_a, self.cumulative_rewards_b = [], []

    def reset_episode_rewards(self) -> None:
        self.current_cumulative_a = self.current_cumulative_b = 0.0
        self.cumulative_rewards_a, self.cumulative_rewards_b = [], []

    def _check_state_changes(self, game: Game) -> None:
        if game.ball and game.ball.in_flag != self.last_in_flag:
            self.event_log.appendleft(f"F{self.frame_count}: IN-FLAG {'ON' if game.ball.in_flag else 'OFF (hit)'}")
            self.last_in_flag = game.ball.in_flag
        if game.scores != self.last_scores:
            if game.scores[0] > self.last_scores[0]:
                self.event_log.appendleft(f"F{self.frame_count}: Point to A")
            if game.scores[1] > self.last_scores[1]:
                self.event_log.appendleft(f"F{self.frame_count}: Point to B")
            self.last_scores = game.scores.copy()

    def update(self, game: Game) -> None:
        self.frame_count += 1
        if game.ball:
            x, y = game.ball.x, game.ball.y
            if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                self.ball_trail.append((x, y, game.ball.in_flag))
        self._check_state_changes(game)

    def _draw_ui(self, game: Game) -> None:
        score = self.font.render(f"Total Wins - Player A: {self.total_wins[0]}  Player B: {self.total_wins[1]}", True, WHITE)
        self.screen.blit(score, score.get_rect(center=(self.window_width // 2, 25)))

        in_flag = game.ball and game.ball.in_flag
        self.screen.blit(self.small_font.render(f"Ball: {'IN' if in_flag else 'OUT'}", True, YELLOW if in_flag else GRAY), (10, 45))
        self.screen.blit(self.small_font.render(f"Rally: {game.rally_count}", True, WHITE), (self.window_width - 100, 45))

        if game.state == GameState.GAME_OVER:
            winner = "A" if game.scores[0] > game.scores[1] else "B"
            text = self.font.render(f"Episode Over! Player {winner} wins!", True, WHITE)
            self.screen.blit(text, text.get_rect(center=(self.window_width // 2, self.window_height - 30)))

    def step_debug(self, observation: dict, reward: float, action: Tuple[int, float], player_id: int = 0) -> None:
        """Store agent's observation, reward, and action for debug visualization."""
        self.agent_obs = observation
        self.agent_reward = reward
        self.agent_action = action
        self.agent_player_id = player_id

    def _draw_agent_debug_point(self) -> None:
        """Draw agent's observed position on game surface (separate from engine rendering)."""
        if self.agent_obs is None:
            return
        prefix = "player_a" if self.agent_player_id == 0 else "player_b"
        x_key, y_key = f"{prefix}_x", f"{prefix}_y"
        if x_key not in self.agent_obs or y_key not in self.agent_obs:
            return
        x, y = self.agent_obs[x_key], self.agent_obs[y_key]
        pos = self._field_to_screen(x, y)
        # Draw diamond marker for agent's observed position
        color = CYAN if self.agent_player_id == 0 else ORANGE
        size = 8
        pts = [(pos[0], pos[1] - size), (pos[0] + size, pos[1]),
               (pos[0], pos[1] + size), (pos[0] - size, pos[1])]
        pygame.draw.polygon(self.game_surface, color, pts, 2)

    def _draw_debug_canvas(self) -> None:
        """Draw debug info on the separate debug surface below game field."""
        self.debug_surface.fill((30, 30, 30))

        # Title
        self.debug_surface.blit(self.debug_font.render("=== DEBUG CANVAS ===", True, CYAN), (10, 5))

        # Agent observation display
        if self.agent_obs is not None:
            y = 25
            self.debug_surface.blit(self.debug_font.render(f"Agent {self.agent_player_id} View:", True, WHITE), (10, y))
            y += 16
            self.debug_surface.blit(self.debug_font.render(f"Reward: {self.agent_reward:.3f}", True, YELLOW), (10, y))
            y += 16
            if self.agent_action:
                self.debug_surface.blit(self.debug_font.render(
                    f"Action: move={self.agent_action[0]}, hit_angle={self.agent_action[1]:.1f}", True, WHITE), (10, y))
            y += 20

            # Show key observation values
            obs = self.agent_obs
            prefix = "player_a" if self.agent_player_id == 0 else "player_b"
            opp_prefix = "player_b" if self.agent_player_id == 0 else "player_a"
            self.debug_surface.blit(self.debug_font.render(
                f"My pos: ({obs.get(f'{prefix}_x', 0):.1f}, {obs.get(f'{prefix}_y', 0):.1f})", True, CYAN), (10, y))
            y += 14
            self.debug_surface.blit(self.debug_font.render(
                f"Opp pos: ({obs.get(f'{opp_prefix}_x', 0):.1f}, {obs.get(f'{opp_prefix}_y', 0):.1f})", True, GRAY), (10, y))
            y += 14
            self.debug_surface.blit(self.debug_font.render(
                f"Ball: ({obs.get('ball_x', 0):.1f}, {obs.get('ball_y', 0):.1f})", True, YELLOW), (10, y))

        # Event log on debug surface
        x = 200
        self.debug_surface.blit(self.debug_font.render("Events:", True, WHITE), (x, 25))
        for i, ev in enumerate(self.event_log):
            alpha = int(255 * (1 - i / max(1, len(self.event_log))))
            self.debug_surface.blit(self.debug_font.render(ev, True, (alpha, alpha, alpha)), (x, 41 + i * 14))

        # Reward graphs on debug surface
        if self.show_graphs:
            gw, gh = 180, 80
            self._draw_graph_on_surface(self.debug_surface, 400, 20, gw, gh,
                                        self.cumulative_rewards_a[-self.graph_max_points:],
                                        self.cumulative_rewards_b[-self.graph_max_points:], "Cumulative (Episode)")
            self._draw_graph_on_surface(self.debug_surface, 400 + gw + 10, 20, gw, gh,
                                        self._get_moving_averages(self.episode_rewards_a),
                                        self._get_moving_averages(self.episode_rewards_b), f"Reward ({self.moving_avg_window}-ep MA)")
            ly = 105
            self.debug_surface.blit(self.debug_font.render("■ A", True, RED), (400, ly))
            self.debug_surface.blit(self.debug_font.render("■ B", True, BLUE), (430, ly))
            self.debug_surface.blit(self.debug_font.render(f"Ep: {len(self.episode_rewards_a)}", True, WHITE), (470, ly))

        # Controls help on debug surface
        helps = ["T-Traj", "D-Dist", "P-Panel", "G-Graph", "SPC-Pause", "N-Step"]
        hx = self.window_width - 300
        self.debug_surface.blit(self.debug_font.render("Controls: " + " | ".join(helps), True, GRAY), (hx, self.debug_canvas_height - 20))

    def _draw_graph_on_surface(self, surface: pygame.Surface, x: int, y: int, width: int, height: int,
                                data_a: List[float], data_b: List[float], title: str) -> None:
        """Draw graph on specified surface."""
        pygame.draw.rect(surface, BLACK, (x, y, width, height))
        pygame.draw.rect(surface, WHITE, (x, y, width, height), 1)
        surface.blit(self.debug_font.render(title, True, WHITE), (x + 5, y + 2))

        gx, gy, gw, gh = x + 5, y + 18, width - 10, height - 23
        if len(data_a) < 2 and len(data_b) < 2:
            surface.blit(self.debug_font.render("No data", True, GRAY), (x + width // 2 - 20, y + height // 2))
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
            pygame.draw.line(surface, GRAY, (gx, zy), (gx + gw, zy), 1)

        def draw_series(data, color):
            if len(data) < 2:
                return
            pts = [(gx + int(i / (len(data) - 1) * gw),
                    max(gy, min(gy + gh, gy + gh - int((v - min_v) / rng * gh))))
                   for i, v in enumerate(data)]
            pygame.draw.lines(surface, color, False, pts, 2)

        draw_series(data_a, RED)
        draw_series(data_b, BLUE)
        if data_a:
            surface.blit(self.debug_font.render(f"A:{data_a[-1]:.1f}", True, RED), (x + width - 50, y + 2))
        if data_b:
            surface.blit(self.debug_font.render(f"B:{data_b[-1]:.1f}", True, BLUE), (x + width - 50, y + 12))

    def render(self, game: Game) -> None:
        # Draw game on game_surface
        old_screen = self.screen
        self.screen = self.game_surface
        self._draw_field(game)
        self._draw_ball_trajectory(game)
        self._draw_ball(game)
        self._draw_players(game)
        self._draw_agent_debug_point()
        self._draw_distances(game)
        self._draw_state_panel(game)
        self._draw_ui(game)
        if self.paused:
            text = self.font.render("PAUSED", True, ORANGE)
            self.game_surface.blit(text, text.get_rect(center=(self.window_width // 2, self.game_area_height // 2)))
        self.screen = old_screen

        # Draw debug canvas
        self._draw_debug_canvas()

        # Blit both surfaces to main screen
        self.screen.blit(self.game_surface, (0, 0))
        self.screen.blit(self.debug_surface, (0, self.game_area_height))
        pygame.display.flip()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN:
                key_actions = {pygame.K_t: 'show_trajectory', pygame.K_d: 'show_distances',
                               pygame.K_p: 'show_state_panel', pygame.K_g: 'show_graphs', pygame.K_SPACE: 'paused'}
                if event.key in key_actions:
                    setattr(self, key_actions[event.key], not getattr(self, key_actions[event.key]))
                elif event.key == pygame.K_n:
                    self.step_mode = True
        return True

    def should_step(self) -> bool:
        if not self.paused:
            return True
        if self.step_mode:
            self.step_mode = False
            return True
        return False


def run_demo_game() -> None:
    import random
    config, game, renderer = Config(), Game(Config()), Renderer(Config())
    while renderer.handle_events() and not game.is_game_over:
        game.step((random.randint(0, 16), random.uniform(0, 360)), (random.randint(0, 16), random.uniform(0, 360)))
        renderer.render(game)
        renderer.tick()
    if game.is_game_over:
        for _ in range(180):
            renderer.handle_events()
            renderer.render(game)
            renderer.tick()
    renderer.close()


if __name__ == "__main__":
    run_demo_game()
