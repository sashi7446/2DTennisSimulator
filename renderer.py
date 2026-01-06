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
    """Training/debug renderer with trajectory, reward graphs, and cumulative stats."""

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
        self._initialized = True

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

    def _draw_event_log(self) -> None:
        if not self.event_log:
            return
        x, y = self.window_width - 200, self.padding + self.ui_height + 10
        for i, ev in enumerate(self.event_log):
            alpha = int(255 * (1 - i / len(self.event_log)))
            self.screen.blit(self.debug_font.render(ev, True, (alpha, alpha, alpha)), (x, y + i * 14))

    def _draw_controls_help(self) -> None:
        helps = ["Debug Controls:", "T - Trajectory", "D - Distances", "P - State panel",
                 "G - Graphs", "SPACE - Pause", "N - Step"]
        x, y = self.window_width - 150, self.window_height - 120
        for i, t in enumerate(helps):
            self.screen.blit(self.debug_font.render(t, True, CYAN if i == 0 else GRAY), (x, y + i * 14))

    def _draw_graph(self, x: int, y: int, width: int, height: int,
                     data_a: List[float], data_b: List[float], title: str, show_zero_line: bool = True) -> None:
        surf = pygame.Surface((width, height))
        surf.set_alpha(200)
        surf.fill(BLACK)
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height), 1)
        self.screen.blit(self.debug_font.render(title, True, WHITE), (x + 5, y + 2))

        gx, gy, gw, gh = x + 5, y + 18, width - 10, height - 23
        if len(data_a) < 2 and len(data_b) < 2:
            self.screen.blit(self.debug_font.render("No data", True, GRAY), (x + width // 2 - 20, y + height // 2))
            return

        all_data = data_a + data_b
        if not all_data:
            return
        min_v, max_v = min(all_data), max(all_data)
        rng = max_v - min_v or 1
        min_v, max_v = min_v - rng * 0.1, max_v + rng * 0.1
        rng = max_v - min_v

        if show_zero_line and min_v < 0 < max_v:
            zy = gy + gh - int(-min_v / rng * gh)
            pygame.draw.line(self.screen, GRAY, (gx, zy), (gx + gw, zy), 1)

        def draw_series(data, color):
            if len(data) < 2:
                return
            pts = [(gx + int(i / (len(data) - 1) * gw),
                    max(gy, min(gy + gh, gy + gh - int((v - min_v) / rng * gh))))
                   for i, v in enumerate(data)]
            pygame.draw.lines(self.screen, color, False, pts, 2)

        draw_series(data_a, RED)
        draw_series(data_b, BLUE)
        if data_a:
            self.screen.blit(self.debug_font.render(f"A:{data_a[-1]:.2f}", True, RED), (x + width - 60, y + 2))
        if data_b:
            self.screen.blit(self.debug_font.render(f"B:{data_b[-1]:.2f}", True, BLUE), (x + width - 60, y + 12))

    def _draw_reward_graphs(self) -> None:
        if not self.show_graphs:
            return
        gw, gh, m = 180, 80, 5
        bx = self.padding + 200
        by = self.padding + self.ui_height + self.config.field_height - gh * 2 - m * 2

        self._draw_graph(bx, by, gw, gh, self.cumulative_rewards_a[-self.graph_max_points:],
                         self.cumulative_rewards_b[-self.graph_max_points:], "Cumulative (Episode)")
        self._draw_graph(bx + gw + m, by, gw, gh, self._get_moving_averages(self.episode_rewards_a),
                         self._get_moving_averages(self.episode_rewards_b), f"Reward ({self.moving_avg_window}-ep MA)")

        ly = by + gh + 5
        self.screen.blit(self.debug_font.render("■ Player A", True, RED), (bx, ly))
        self.screen.blit(self.debug_font.render("■ Player B", True, BLUE), (bx + 80, ly))
        self.screen.blit(self.debug_font.render(f"Episodes: {len(self.episode_rewards_a)}", True, WHITE), (bx + gw + m, ly))

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

    def render(self, game: Game) -> None:
        self._draw_field(game)
        self._draw_ball_trajectory(game)
        self._draw_ball(game)
        self._draw_players(game)
        self._draw_distances(game)
        self._draw_state_panel(game)
        self._draw_event_log()
        self._draw_reward_graphs()
        self._draw_controls_help()
        self._draw_ui(game)
        if self.paused:
            text = self.font.render("PAUSED", True, ORANGE)
            self.screen.blit(text, text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50)))
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
