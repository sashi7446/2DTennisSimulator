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
    from agents import NeuralAgent, TransformerAgent


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
    elif agent_type == "transformer":
        if not NEURAL_AVAILABLE:
            print("Warning: TransformerAgent requires numpy. Falling back to ChaseAgent.")
            agent = ChaseAgent()
        else:
            agent = TransformerAgent()
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
    - InputHandler: processes events, manages pause/step/speed state
    - StatsTracker: tracks rewards, wins, events
    - Renderer: draws game state to screen (passive)
    - Game: runs simulation logic
    """
    import time
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
        print("1-4  - Speed (1x/2x/4x/Max)")
        print("T/D/P/G - Toggle overlays")
        print("F    - Toggle FPS display")
        print("SPACE - Pause, N - Step")
        print("R    - Reset episode")
        print("S    - Save agents")
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
    step_count = 0  # For render skipping

    # FPS measurement
    fps_update_interval = 0.5  # Update FPS every 0.5 seconds
    fps_last_time = time.time()
    fps_frame_count = 0
    actual_fps = 0.0

    while input_handler.running:
        # Start new episode
        game = Game(config)
        episode_count += 1
        agent_a.reset()
        agent_b.reset()
        step_count = 0
        last_in_flag = False
        last_obs = None
        last_actions = None
        last_rewards = None

        # print(f"Episode {episode_count} started")

        while input_handler.running and not game.is_game_over:
            input_handler.process_events()

            # Reset episode if requested
            if input_handler.consume_reset_request():
                break

            if input_handler.consume_save_request() and save_dir:
                _save_agents(agent_a, agent_b, save_dir, episode_count)

            # Game step
            if input_handler.should_step():
                current_obs = game.get_observation()
                current_action_a = agent_a.act(current_obs)
                current_action_b = agent_b.act(current_obs)
                result = game.step(current_action_a, current_action_b)

                agent_a.learn(result.rewards[0], game.is_game_over)
                agent_b.learn(result.rewards[1], game.is_game_over)
                step_count += 1

                # Store for debug rendering
                last_obs = current_obs
                last_actions = (current_action_a, current_action_b)
                last_rewards = result.rewards

                if debug and stats:
                    stats.add_reward(result.rewards[0], result.rewards[1])
                    stats.next_frame()
                    frame_count += 1

                    if game.ball and game.ball.in_flag != last_in_flag:
                        stats.log_event(f"IN-FLAG {'ON' if game.ball.in_flag else 'OFF'}")
                        last_in_flag = game.ball.in_flag

                    renderer.update(game)

            # Render with skip logic for high-speed modes
            render_interval = input_handler.state.render_interval
            should_render = (step_count % render_interval == 0) or input_handler.state.paused

            if should_render:
                # Update FPS measurement
                fps_frame_count += 1
                now = time.time()
                elapsed = now - fps_last_time
                if elapsed >= fps_update_interval:
                    actual_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_last_time = now

                if debug:
                    renderer.render(game, total_wins=tuple(total_wins),
                                    stats=stats, input_state=input_handler.state,
                                    frame_count=frame_count, actual_fps=actual_fps,
                                    obs=last_obs, actions=last_actions, rewards=last_rewards)
                else:
                    renderer.render(game, total_wins=tuple(total_wins))

            # Tick with dynamic FPS (0 = no limit)
            target_fps = input_handler.state.target_fps
            renderer.tick(target_fps if target_fps > 0 else 0)

        # Episode end (skip if reset was requested)
        if game.is_game_over:
            winner = game.winner
            total_wins[winner] += 1

            if debug and stats:
                stats.end_episode(winner)

            winner_name = "A" if winner == 0 else "B"
            # print(f"Episode {episode_count}: Player {winner_name} wins! "
            #       f"Total wins: A={total_wins[0]}, B={total_wins[1]}")

            for name, agent in [("A", agent_a), ("B", agent_b)]:
                info = agent.get_info()
                if "episodes_trained" in info:
                    print(f"  {name}: {info.get('episodes_trained', 0)} episodes trained, "
                          f"recent avg: {info.get('recent_avg_reward', 0):.2f}")

            # Brief pause to show result (skip in unlimited mode)
            if input_handler.state.target_fps != 0:
                for _ in range(60):
                    input_handler.process_events()
                    if not input_handler.running or input_handler.consume_reset_request():
                        break
                    if debug:
                        renderer.render(game, total_wins=tuple(total_wins),
                                        stats=stats, input_state=input_handler.state,
                                        frame_count=frame_count, actual_fps=actual_fps,
                                        obs=last_obs, actions=last_actions, rewards=last_rewards)
                    else:
                        renderer.render(game, total_wins=tuple(total_wins))
                    renderer.tick(60)

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
) -> tuple:
    """Run training without visualization.

    Returns:
        Tuple of (wins, agent_a, agent_b) for use in benchmark mode.
    """
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

    return wins, agent_a, agent_b


def run_benchmark(
    config: Config,
    agent_a: Agent,
    agent_b: Agent,
    train_episodes: int = 300,
    save_dir: Optional[str] = None,
) -> None:
    """Run benchmark: train headless, then visualize in debug mode.

    This mode:
    1. Trains agents for train_episodes in headless mode
    2. Automatically launches debug visualization to observe learned behavior
    """
    print(f"\n{'='*50}")
    print(f"  BENCHMARK MODE")
    print(f"  Training: {train_episodes} episodes (headless)")
    print(f"  Then: Debug visualization")
    print(f"{'='*50}\n")

    # Phase 1: Headless training
    wins, agent_a, agent_b = run_headless_training(
        config, agent_a, agent_b,
        num_episodes=train_episodes,
        save_dir=save_dir,
        save_interval=50,
    )

    # Print training summary
    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  Results: A={wins[0]} wins ({100*wins[0]/train_episodes:.1f}%)")
    print(f"           B={wins[1]} wins ({100*wins[1]/train_episodes:.1f}%)")
    print(f"{'='*50}")
    print(f"\nLaunching debug visualization...")
    print(f"Watch how the trained agents play!\n")

    # Disable learning for visualization (optional: observe pure policy)
    if hasattr(agent_a, 'set_training_mode'):
        agent_a.set_training_mode(False)
    if hasattr(agent_b, 'set_training_mode'):
        agent_b.set_training_mode(False)

    # Phase 2: Debug visualization
    run_visual_game(
        config, agent_a, agent_b,
        debug=True,
        save_dir=save_dir,
    )


def list_agent_types() -> None:
    """Print available agent types."""
    print("\nAvailable agent types:")
    print("  chase   - Simple ball-chasing AI")
    print("  smart   - Improved chase with positioning")
    print("  random  - Random actions (baseline)")
    if NEURAL_AVAILABLE:
        print("  neural      - Learning neural network agent")
        print("  transformer - Advanced Transformer-based agent")
    else:
        print("  neural      - (requires numpy)")
        print("  transformer - (requires numpy)")
    print("\nYou can also specify a path to a saved agent directory.")


def main():
    parser = argparse.ArgumentParser(
        description="2D Tennis Simulator - Watch AI agents learn and compete!"
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "benchmark", "list"],
        default="visual",
        help="Game mode: visual, headless, benchmark (train then debug), list",
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
    elif args.mode == "benchmark":
        run_benchmark(
            config, agent_a, agent_b,
            train_episodes=args.episodes,
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
