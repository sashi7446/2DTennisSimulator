"""Pygame renderer for 2D Tennis Simulator."""

from typing import Optional, Tuple
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


class Renderer:
    """
    Pygame-based renderer for the tennis simulator.

    Provides visual representation of:
    - The field with walls and in-areas
    - Players and ball
    - Score display
    - In-flag indicator
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

        # Draw velocity indicator
        if game.ball.vx != 0 or game.ball.vy != 0:
            speed = game.ball.get_speed()
            scale = 3  # Scale factor for velocity line
            end_x = ball_screen[0] + int(game.ball.vx * scale)
            end_y = ball_screen[1] + int(game.ball.vy * scale)
            pygame.draw.line(self.screen, WHITE, ball_screen, (end_x, end_y), 1)

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
        """Draw the score and game state UI."""
        # Score display
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
