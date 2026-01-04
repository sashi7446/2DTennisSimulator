#!/usr/bin/env python3
"""
2D Tennis Simulator - Main Entry Point

A 2D tennis simulation where AI agents compete against each other.
Designed to observe emergent tactical behaviors like returning to home position.
"""

import argparse
import random
import math
from typing import Optional

from config import Config
from game import Game, GameState


def run_headless_game(
    config: Optional[Config] = None,
    max_points: int = 11,
    verbose: bool = True,
) -> tuple:
    """
    Run a game without visualization.

    Args:
        config: Game configuration
        max_points: Points to win
        verbose: Print game events

    Returns:
        Tuple of (winner, final_scores, total_steps)
    """
    config = config or Config()
    config.points_to_win = max_points
    game = Game(config)

    while not game.is_game_over:
        # Random actions for demo
        action_a = (
            random.randint(0, 16),
            random.uniform(0, 360),
        )
        action_b = (
            random.randint(0, 16),
            random.uniform(0, 360),
        )

        result = game.step(action_a, action_b)

        if result.point_result and verbose:
            pr = result.point_result
            winner_name = "A" if pr.winner == 0 else "B"
            print(
                f"Point to Player {winner_name} ({pr.reason}) - "
                f"Score: {game.scores[0]}-{game.scores[1]}"
            )

    if verbose:
        winner = "A" if game.winner == 0 else "B"
        print(f"\nGame Over! Player {winner} wins!")
        print(f"Final Score: {game.scores[0]}-{game.scores[1]}")
        print(f"Total Steps: {game.total_steps}")

    return game.winner, game.scores, game.total_steps


def run_visual_game(config: Optional[Config] = None) -> None:
    """Run a game with pygame visualization."""
    try:
        from renderer import Renderer
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame with: pip install pygame")
        return

    config = config or Config()
    game = Game(config)
    renderer = Renderer(config)

    running = True
    while running and not game.is_game_over:
        running = renderer.handle_events()

        # Simple chase AI for demo
        obs = game.get_observation()

        # Player A chases ball
        dx_a = obs["ball_x"] - obs["player_a_x"]
        dy_a = obs["ball_y"] - obs["player_a_y"]
        angle_a = math.degrees(math.atan2(dy_a, dx_a))
        if angle_a < 0:
            angle_a += 360
        move_a = int(angle_a / 22.5) % 16

        # Player B chases ball
        dx_b = obs["ball_x"] - obs["player_b_x"]
        dy_b = obs["ball_y"] - obs["player_b_y"]
        angle_b = math.degrees(math.atan2(dy_b, dx_b))
        if angle_b < 0:
            angle_b += 360
        move_b = int(angle_b / 22.5) % 16

        # Hit toward opponent's side
        hit_a = 0 if obs["player_a_x"] < config.field_width / 2 else 180
        hit_b = 180 if obs["player_b_x"] > config.field_width / 2 else 0

        result = game.step((move_a, hit_a), (move_b, hit_b))

        renderer.render(game)
        renderer.tick()

    # Show final state
    if game.is_game_over:
        for _ in range(180):
            if not renderer.handle_events():
                break
            renderer.render(game)
            renderer.tick()

    renderer.close()


def run_keyboard_game(config: Optional[Config] = None) -> None:
    """Run a game with keyboard control for Player A."""
    try:
        import pygame
        from renderer import Renderer
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame with: pip install pygame")
        return

    config = config or Config()
    game = Game(config)
    renderer = Renderer(config)

    print("\n=== Keyboard Controls ===")
    print("Arrow keys / WASD: Move Player A")
    print("Space: Hit ball (auto-aims toward opponent)")
    print("ESC: Quit")
    print("========================\n")

    running = True
    while running and not game.is_game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Determine movement direction for Player A
        dx, dy = 0, 0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1

        if dx != 0 or dy != 0:
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            move_a = int(angle / 22.5) % 16
        else:
            move_a = 16  # Stay

        # Hit toward opponent's side
        hit_a = 0.0  # Aim right

        # Simple AI for Player B
        obs = game.get_observation()
        dx_b = obs["ball_x"] - obs["player_b_x"]
        dy_b = obs["ball_y"] - obs["player_b_y"]
        angle_b = math.degrees(math.atan2(dy_b, dx_b))
        if angle_b < 0:
            angle_b += 360
        move_b = int(angle_b / 22.5) % 16
        hit_b = 180.0  # Aim left

        game.step((move_a, hit_a), (move_b, hit_b))

        renderer.render(game)
        renderer.tick()

    renderer.close()


def main():
    parser = argparse.ArgumentParser(
        description="2D Tennis Simulator for AI agents"
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "keyboard"],
        default="visual",
        help="Game mode (default: visual)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=11,
        help="Points to win (default: 11)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Ball speed (default: 5.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for visual mode (default: 60)",
    )

    args = parser.parse_args()

    # Create config
    config = Config(
        ball_speed=args.speed,
        points_to_win=args.points,
        fps=args.fps,
    )

    if args.mode == "headless":
        run_headless_game(config, args.points)
    elif args.mode == "keyboard":
        run_keyboard_game(config)
    else:
        run_visual_game(config)


if __name__ == "__main__":
    main()
