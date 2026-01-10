#!/usr/bin/env python3
"""
2D Tennis Simulator - Main Entry Point

A 2D tennis simulation where AI agents compete against each other.
Designed to observe emergent tactical behaviors like returning to home position.
"""

import argparse
import os
from typing import Optional

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from agents import (
    NEURAL_AVAILABLE,
    Agent,
    ChaseAgent,
    RandomAgent,
    SmartChaseAgent,
    load_agent,
)
from config import Config
from game import Game

if NEURAL_AVAILABLE:
    from agents import NeuralAgent, TransformerAgent


def create_agent(
    agent_type: str, player_id: int, config: Config, load_path: Optional[str] = None
) -> Agent:
    """Create or load an agent."""
    if load_path and os.path.exists(load_path):
        agent = load_agent(load_path)
        agent.set_player_id(player_id)
        if hasattr(agent, "set_field_dimensions"):
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
    if hasattr(agent, "set_field_dimensions"):
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
        from input_handler import InputHandler
        from renderer import DebugRenderer, GameRenderer
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

    print("\n=== Match ===")
    print(f"Player A: {agent_a.config.name} ({agent_a.config.agent_type})")
    print(f"Player B: {agent_b.config.name} ({agent_b.config.agent_type})")
    print("=============\n")

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
        last_is_in = False
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

                    if game.ball and game.ball.is_in != last_is_in:
                        stats.log_event(f"is_in {'ON' if game.ball.is_in else 'OFF'}")
                        last_is_in = game.ball.is_in

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
                    renderer.render(
                        game,
                        total_wins=tuple(total_wins),
                        stats=stats,
                        input_state=input_handler.state,
                        frame_count=frame_count,
                        actual_fps=actual_fps,
                        obs=last_obs,
                        actions=last_actions,
                        rewards=last_rewards,
                    )
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

            # winner_name = "A" if winner == 0 else "B"

            for name, agent in [("A", agent_a), ("B", agent_b)]:
                info = agent.get_info()
                if "episodes_trained" in info:
                    print(
                        f"  {name}: {info.get('episodes_trained', 0)} episodes trained, "
                        f"recent avg: {info.get('recent_avg_reward', 0):.2f}"
                    )

            # Brief pause to show result (skip in unlimited mode)
            if input_handler.state.target_fps != 0:
                for _ in range(60):
                    input_handler.process_events()
                    if not input_handler.running or input_handler.consume_reset_request():
                        break
                    if debug:
                        renderer.render(
                            game,
                            total_wins=tuple(total_wins),
                            stats=stats,
                            input_state=input_handler.state,
                            frame_count=frame_count,
                            actual_fps=actual_fps,
                            obs=last_obs,
                            actions=last_actions,
                            rewards=last_rewards,
                        )
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
    print("\n=== Headless Training ===")
    print(f"Player A: {agent_a.config.name}")
    print(f"Player B: {agent_b.config.name}")
    print(f"Episodes: {num_episodes}")
    print("=========================\n")

    wins = [0, 0]

    # Use tqdm progress bar if available
    episode_range = range(1, num_episodes + 1)
    if TQDM_AVAILABLE:
        episode_iterator = tqdm(episode_range, desc="Training", unit="episode")
    else:
        episode_iterator = episode_range

    for episode in episode_iterator:
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

        # Update progress bar description with current stats
        if TQDM_AVAILABLE and episode % 10 == 0:
            win_rate_a = 100 * wins[0] / episode
            win_rate_b = 100 * wins[1] / episode
            episode_iterator.set_postfix(
                {"A": f"{wins[0]} ({win_rate_a:.1f}%)", "B": f"{wins[1]} ({win_rate_b:.1f}%)"}
            )
        elif not TQDM_AVAILABLE and episode % 10 == 0:
            print(
                f"Episode {episode}: Wins A={wins[0]} B={wins[1]} "
                f"({100*wins[0]/episode:.1f}% vs {100*wins[1]/episode:.1f}%)"
            )

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
    print("  BENCHMARK MODE")
    print(f"  Training: {train_episodes} episodes (headless)")
    print("  Then: Debug visualization")
    print(f"{'='*50}\n")

    # Phase 1: Headless training
    wins, agent_a, agent_b = run_headless_training(
        config,
        agent_a,
        agent_b,
        num_episodes=train_episodes,
        save_dir=save_dir,
        save_interval=50,
    )

    # Print training summary
    print(f"\n{'='*50}")
    print("  TRAINING COMPLETE")
    print(f"  Results: A={wins[0]} wins ({100*wins[0]/train_episodes:.1f}%)")
    print(f"           B={wins[1]} wins ({100*wins[1]/train_episodes:.1f}%)")
    print(f"{'='*50}")
    print("\nLaunching debug visualization...")
    print("Watch how the trained agents play!\n")

    # Disable learning for visualization (optional: observe pure policy)
    if hasattr(agent_a, "set_training_mode"):
        agent_a.set_training_mode(False)
    if hasattr(agent_b, "set_training_mode"):
        agent_b.set_training_mode(False)

    # Phase 2: Debug visualization
    run_visual_game(
        config,
        agent_a,
        agent_b,
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
        description="2D Tennis Simulator - Watch AI agents learn and compete!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch two chase agents compete visually
  python main.py --agent-a chase --agent-b chase

  # Train a neural agent against chase AI with debug mode
  python main.py --agent-a neural --agent-b chase --debug

  # Fast headless training for 500 episodes
  python main.py --mode headless --agent-a neural --agent-b chase --episodes 500

  # Benchmark mode: train 300 episodes then visualize
  python main.py --mode benchmark --agent-a neural --agent-b smart --episodes 300

  # List all available agent types
  python main.py --mode list

  # Load a saved agent and compete
  python main.py --agent-a saved_agents/agent_a_neural --agent-b chase

For more information, see README.md
""",
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "headless", "benchmark", "list"],
        default="visual",
        help="Game mode (default: %(default)s)\n"
        "  visual: Watch games with pygame rendering\n"
        "  headless: Fast training without graphics\n"
        "  benchmark: Train headless, then show debug view\n"
        "  list: Display available agent types",
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="chase",
        metavar="TYPE",
        help="Agent type for Player A (default: %(default)s)\n"
        "Built-in types: chase, smart, random, neural, transformer\n"
        "Or provide a path to a saved agent directory",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="chase",
        metavar="TYPE",
        help="Agent type for Player B (default: %(default)s)\n"
        "Built-in types: chase, smart, random, neural, transformer\n"
        "Or provide a path to a saved agent directory",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        metavar="SPEED",
        help=f"Ball speed in pixels/frame (default: {Config().ball_speed})\n"
        "Higher values make the game faster and more challenging",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        metavar="FPS",
        help=f"Target frames per second in visual mode (default: {Config().fps})\n"
        "Use 0 for unlimited FPS (max speed)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in visual mode\n"
        "Shows: state overlay, reward graphs, trajectory prediction\n"
        "Keyboard: D (debug), T (trajectory), P (graphs), G (grid)\n"
        "          1-4 (speed), SPACE (pause), R (reset), S (save)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save trained agents (optional)\n"
        "Agents are saved at regular intervals during training\n"
        "In visual mode, press 'S' to save manually",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        metavar="N",
        help="Number of episodes for headless/benchmark training\n"
        "(default: %(default)s)\n"
        "Each episode is one complete point (serve until wall hit)",
    )

    args = parser.parse_args()

    if args.mode == "list":
        list_agent_types()
        return

    config_kwargs = {}
    if args.speed is not None:
        config_kwargs["ball_speed"] = args.speed
    if args.fps is not None:
        config_kwargs["fps"] = args.fps

    config = Config(**config_kwargs)

    agent_a = create_agent(args.agent_a, player_id=0, config=config)
    agent_b = create_agent(args.agent_b, player_id=1, config=config)

    if args.mode == "headless":
        run_headless_training(
            config,
            agent_a,
            agent_b,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
        )
    elif args.mode == "benchmark":
        run_benchmark(
            config,
            agent_a,
            agent_b,
            train_episodes=args.episodes,
            save_dir=args.save_dir,
        )
    else:
        run_visual_game(
            config,
            agent_a,
            agent_b,
            debug=args.debug,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
