#!/usr/bin/env python3
"""
2D Tennis Simulator - Main Entry Point

A 2D tennis simulation where AI agents compete against each other.
Designed to observe emergent tactical behaviors like returning to home position.
"""

import argparse
import os
from typing import Optional, Tuple

from config import Config
from game import Game, GameState
from agents import (
    Agent,
    ChaseAgent,
    SmartChaseAgent,
    RandomAgent,
    load_agent,
    NEURAL_AVAILABLE,
)

if NEURAL_AVAILABLE:
    from agents import NeuralAgent


def create_agent(agent_type: str, player_id: int, config: Config, load_path: Optional[str] = None) -> Agent:
    """
    Create or load an agent.

    Args:
        agent_type: Type of agent (chase, smart, random, neural, or path to saved agent)
        player_id: 0 for Player A, 1 for Player B
        config: Game configuration
        load_path: Path to load agent from (optional)

    Returns:
        Configured agent instance
    """
    # Check if it's a path to a saved agent
    if load_path and os.path.exists(load_path):
        agent = load_agent(load_path)
        agent.set_player_id(player_id)
        if hasattr(agent, 'set_field_dimensions'):
            agent.set_field_dimensions(config.field_width, config.field_height)
        print(f"Loaded agent from {load_path}: {agent.config.name}")
        return agent

    # Create new agent by type
    if agent_type == "chase":
        agent = ChaseAgent()
    elif agent_type == "smart":
        agent = SmartChaseAgent()
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "neural":
        if not NEURAL_AVAILABLE:
            print("Warning: NeuralAgent requires numpy. Falling back to ChaseAgent.")
            agent = ChaseAgent()
        else:
            agent = NeuralAgent()
    else:
        # Try to load from path
        if os.path.exists(agent_type):
            agent = load_agent(agent_type)
        else:
            print(f"Unknown agent type: {agent_type}. Using ChaseAgent.")
            agent = ChaseAgent()

    agent.set_player_id(player_id)
    if hasattr(agent, 'set_field_dimensions'):
        agent.set_field_dimensions(config.field_width, config.field_height)

    return agent


def run_visual_game(
    config: Config,
    agent_a: Agent,
    agent_b: Agent,
    debug: bool = False,
    save_dir: Optional[str] = None,
) -> None:
    """Run a game with pygame visualization and agent control."""
    try:
        from renderer import Renderer, DebugRenderer
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame with: pip install pygame")
        return

    if debug:
        renderer = DebugRenderer(config)
        print("\n=== DEBUG MODE ===")
        print("T - Toggle trajectory")
        print("D - Toggle distances")
        print("P - Toggle state panel")
        print("G - Toggle graphs")
        print("SPACE - Pause/Resume")
        print("N - Step (when paused)")
        print("S - Save agents")
        print("==================\n")
    else:
        renderer = Renderer(config)

    # Print agent info
    print(f"\n=== Match ===")
    print(f"Player A: {agent_a.config.name} ({agent_a.config.agent_type})")
    print(f"Player B: {agent_b.config.name} ({agent_b.config.agent_type})")
    print(f"=============\n")

    running = True
    episode_count = 0
    save_requested = False

    while running:
        # Start new game/episode
        game = Game(config)
        episode_count += 1

        # Reset agents for new episode
        agent_a.reset()
        agent_b.reset()

        print(f"Episode {episode_count} started")

        while running and not game.is_game_over:
            # Handle events
            for event in _get_pygame_events():
                if event.type == _get_pygame_const('QUIT'):
                    running = False
                elif event.type == _get_pygame_const('KEYDOWN'):
                    if event.key == _get_pygame_const('K_ESCAPE'):
                        running = False
                    elif event.key == _get_pygame_const('K_s'):
                        save_requested = True

            if not running:
                break

            running = renderer.handle_events()

            # Check if we should step (respects pause in debug mode)
            should_step = True
            if debug and hasattr(renderer, 'should_step'):
                should_step = renderer.should_step()

            if should_step:
                # Get observation
                obs = game.get_observation()

                # Agents choose actions
                action_a = agent_a.act(obs)
                action_b = agent_b.act(obs)

                # Step game
                result = game.step(action_a, action_b)

                # Agents learn from rewards
                agent_a.learn(result.rewards[0], game.is_game_over)
                agent_b.learn(result.rewards[1], game.is_game_over)

                # Update debug renderer
                if debug and hasattr(renderer, 'update'):
                    renderer.update(game)
                if debug and hasattr(renderer, 'add_reward'):
                    renderer.add_reward(result.rewards[0], result.rewards[1])

            renderer.render(game)
            renderer.tick()

        # Episode ended
        if debug and hasattr(renderer, 'end_episode'):
            renderer.end_episode()

        # Print episode result
        if game.is_game_over:
            winner = "A" if game.winner == 0 else "B"
            print(f"Episode {episode_count}: Player {winner} wins! "
                  f"Score: {game.scores[0]}-{game.scores[1]}")

            # Print learning stats if available
            for name, agent in [("A", agent_a), ("B", agent_b)]:
                info = agent.get_info()
                if "episodes_trained" in info:
                    print(f"  {name}: {info.get('episodes_trained', 0)} episodes trained, "
                          f"recent avg: {info.get('recent_avg_reward', 0):.2f}")

        # Save if requested
        if save_requested and save_dir:
            _save_agents(agent_a, agent_b, save_dir, episode_count)
            save_requested = False

        # Show final state briefly, then auto-restart
        if game.is_game_over and running:
            for _ in range(60):  # 1 second at 60fps
                if not renderer.handle_events():
                    running = False
                    break
                renderer.render(game)
                renderer.tick()

    # Final save
    if save_dir:
        _save_agents(agent_a, agent_b, save_dir, episode_count)

    renderer.close()


def _get_pygame_events():
    """Get pygame events (lazy import)."""
    import pygame
    return pygame.event.get()


def _get_pygame_const(name: str):
    """Get pygame constant (lazy import)."""
    import pygame
    return getattr(pygame, name)


def _save_agents(agent_a: Agent, agent_b: Agent, save_dir: str, episode: int) -> None:
    """Save both agents."""
    os.makedirs(save_dir, exist_ok=True)

    path_a = os.path.join(save_dir, f"agent_a_{agent_a.config.agent_type}")
    path_b = os.path.join(save_dir, f"agent_b_{agent_b.config.agent_type}")

    agent_a.save(path_a)
    agent_b.save(path_b)

    print(f"Agents saved to {save_dir} (episode {episode})")


def run_headless_training(
    config: Config,
    agent_a: Agent,
    agent_b: Agent,
    num_episodes: int = 100,
    save_dir: Optional[str] = None,
    save_interval: int = 10,
) -> None:
    """Run training without visualization."""
    print(f"\n=== Headless Training ===")
    print(f"Player A: {agent_a.config.name}")
    print(f"Player B: {agent_b.config.name}")
    print(f"Episodes: {num_episodes}")
    print(f"=========================\n")

    wins = [0, 0]

    for episode in range(1, num_episodes + 1):
        game = Game(config)
        agent_a.reset()
        agent_b.reset()

        while not game.is_game_over:
            obs = game.get_observation()
            action_a = agent_a.act(obs)
            action_b = agent_b.act(obs)
            result = game.step(action_a, action_b)
            agent_a.learn(result.rewards[0], game.is_game_over)
            agent_b.learn(result.rewards[1], game.is_game_over)

        wins[game.winner] += 1

        if episode % 10 == 0:
            print(f"Episode {episode}: Wins A={wins[0]} B={wins[1]} "
                  f"({100*wins[0]/episode:.1f}% vs {100*wins[1]/episode:.1f}%)")

        if save_dir and episode % save_interval == 0:
            _save_agents(agent_a, agent_b, save_dir, episode)

    print(f"\nFinal: A={wins[0]} wins, B={wins[1]} wins")

    if save_dir:
        _save_agents(agent_a, agent_b, save_dir, num_episodes)


def list_agent_types() -> None:
    """Print available agent types."""
    print("\nAvailable agent types:")
    print("  chase   - Simple ball-chasing AI")
    print("  smart   - Improved chase with positioning")
    print("  random  - Random actions (baseline)")
    if NEURAL_AVAILABLE:
        print("  neural  - Learning neural network agent")
    else:
        print("  neural  - (requires numpy)")
    print("\nYou can also specify a path to a saved agent directory.")


def main():
    parser = argparse.ArgumentParser(
        description="2D Tennis Simulator - Watch AI agents learn and compete!"
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "list"],
        default="visual",
        help="Game mode (default: visual)",
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="chase",
        help="Agent type for Player A (chase/smart/random/neural or path)",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="chase",
        help="Agent type for Player B (chase/smart/random/neural or path)",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with state overlay and graphs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save trained agents",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for headless training (default: 100)",
    )

    args = parser.parse_args()

    # List mode
    if args.mode == "list":
        list_agent_types()
        return

    # Create config
    config = Config(
        ball_speed=args.speed,
        points_to_win=args.points,
        fps=args.fps,
    )

    # Create agents
    agent_a = create_agent(args.agent_a, player_id=0, config=config)
    agent_b = create_agent(args.agent_b, player_id=1, config=config)

    # Run game
    if args.mode == "headless":
        run_headless_training(
            config, agent_a, agent_b,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
        )
    else:
        run_visual_game(
            config, agent_a, agent_b,
            debug=args.debug,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
