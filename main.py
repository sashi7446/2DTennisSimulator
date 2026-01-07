#!/usr/bin/env python3
"""
2D Tennis Simulator - Main Entry Point

A 2D tennis simulation where AI agents compete against each other.
Designed to observe emergent tactical behaviors like returning to home position.
"""

import argparse
import os
from typing import Optional

from config import Config
from game import Game
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
    """Create or load an agent."""
    if load_path and os.path.exists(load_path):
        agent = load_agent(load_path)
        agent.set_player_id(player_id)
        if hasattr(agent, 'set_field_dimensions'):
            agent.set_field_dimensions(config.field_width, config.field_height)
        print(f"Loaded agent from {load_path}: {agent.config.name}")
        return agent

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
    """Run a game with pygame visualization and agent control.

    Responsibilities are cleanly separated:
    - InputHandler: processes events, manages pause/step state
    - StatsTracker: tracks rewards, wins, events
    - Renderer: draws game state to screen (passive)
    - Game: runs simulation logic
    """
    try:
        from renderer import GameRenderer, DebugRenderer
        from input_handler import InputHandler
        from stats_tracker import StatsTracker
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame with: pip install pygame")
        return

    # Create components
    input_handler = InputHandler(debug_mode=debug)
    stats = StatsTracker() if debug else None

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
        renderer = GameRenderer(config)

    print(f"\n=== Match ===")
    print(f"Player A: {agent_a.config.name} ({agent_a.config.agent_type})")
    print(f"Player B: {agent_b.config.name} ({agent_b.config.agent_type})")
    print(f"=============\n")

    episode_count = 0
    total_wins = [0, 0]
    frame_count = 0

    while input_handler.running:
        # Start new episode
        game = Game(config)
        episode_count += 1
        agent_a.reset()
        agent_b.reset()

        # Track in_flag changes for event log
        last_in_flag = False

        print(f"Episode {episode_count} started (Total wins: A={total_wins[0]}, B={total_wins[1]})")

        while input_handler.running and not game.is_game_over:
            # Input processing (separate from rendering)
            input_handler.process_events()

            # Save if requested
            if input_handler.consume_save_request() and save_dir:
                _save_agents(agent_a, agent_b, save_dir, episode_count)

            # Game step (controlled by InputHandler, not Renderer)
            if input_handler.should_step():
                obs = game.get_observation()
                action_a = agent_a.act(obs)
                action_b = agent_b.act(obs)
                result = game.step(action_a, action_b)

                agent_a.learn(result.rewards[0], game.is_game_over)
                agent_b.learn(result.rewards[1], game.is_game_over)

                if debug and stats:
                    stats.add_reward(result.rewards[0], result.rewards[1])
                    stats.next_frame()
                    frame_count += 1

                    # Log state changes
                    if game.ball and game.ball.in_flag != last_in_flag:
                        stats.log_event(f"IN-FLAG {'ON' if game.ball.in_flag else 'OFF'}")
                        last_in_flag = game.ball.in_flag

                    renderer.update(game)

            # Render (passive - just draws current state)
            if debug:
                renderer.render(game, total_wins=tuple(total_wins),
                                stats=stats, input_state=input_handler.state,
                                frame_count=frame_count)
            else:
                renderer.render(game, total_wins=tuple(total_wins))

            renderer.tick()

        # Episode end
        if game.is_game_over:
            winner = game.winner
            total_wins[winner] += 1

            if debug and stats:
                stats.end_episode(winner)

            winner_name = "A" if winner == 0 else "B"
            print(f"Episode {episode_count}: Player {winner_name} wins! "
                  f"Total wins: A={total_wins[0]}, B={total_wins[1]}")

            for name, agent in [("A", agent_a), ("B", agent_b)]:
                info = agent.get_info()
                if "episodes_trained" in info:
                    print(f"  {name}: {info.get('episodes_trained', 0)} episodes trained, "
                          f"recent avg: {info.get('recent_avg_reward', 0):.2f}")

        # Show final state briefly
        if game.is_game_over and input_handler.running:
            for _ in range(60):
                input_handler.process_events()
                if not input_handler.running:
                    break
                if debug:
                    renderer.render(game, total_wins=tuple(total_wins),
                                    stats=stats, input_state=input_handler.state,
                                    frame_count=frame_count)
                else:
                    renderer.render(game, total_wins=tuple(total_wins))
                renderer.tick()

    # Final save
    if save_dir:
        _save_agents(agent_a, agent_b, save_dir, episode_count)

    renderer.close()


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
        "--speed",
        type=float,
        default=None,
        help=f"Ball speed (default: {Config().ball_speed})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=f"Frames per second for visual mode (default: {Config().fps})",
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

    if args.mode == "list":
        list_agent_types()
        return

    config_kwargs = {}
    if args.speed is not None:
        config_kwargs['ball_speed'] = args.speed
    if args.fps is not None:
        config_kwargs['fps'] = args.fps

    config = Config(**config_kwargs)

    agent_a = create_agent(args.agent_a, player_id=0, config=config)
    agent_b = create_agent(args.agent_b, player_id=1, config=config)

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
