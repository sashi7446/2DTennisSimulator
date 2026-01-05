"""Neural Network Agent with Policy Gradient Learning."""

import math
import json
import pickle
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from agents.base import Agent, AgentConfig


@dataclass
class NeuralAgentConfig(AgentConfig):
    """Configuration for neural network agent."""

    name: str = "NeuralBot"
    agent_type: str = "neural"
    version: str = "1.0"
    description: str = "Neural network agent with policy gradient learning"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 64,
        "learning_rate": 0.001,
        "gamma": 0.99,  # Discount factor
        "entropy_coef": 0.01,  # Entropy bonus for exploration
    })


class NeuralAgent(Agent):
    """
    Neural network agent that learns via policy gradient.

    Network architecture:
    - Input: 11 features (ball pos/vel, player positions, scores, in_flag)
    - Hidden: 64 units with ReLU
    - Output: 17 movement logits + 2 hit angle parameters (mean, std)

    Learning:
    - REINFORCE algorithm with baseline
    - Entropy bonus for exploration
    """

    def __init__(self, config: Optional[NeuralAgentConfig] = None):
        if config is None:
            config = NeuralAgentConfig()
        super().__init__(config)

        # Network parameters
        self.hidden_size = config.parameters.get("hidden_size", 64)
        self.learning_rate = config.parameters.get("learning_rate", 0.001)
        self.gamma = config.parameters.get("gamma", 0.99)
        self.entropy_coef = config.parameters.get("entropy_coef", 0.01)

        # Input: 11 features, Output: 17 move + 2 angle params
        self.input_size = 11
        self.move_output_size = 17
        self.angle_output_size = 2  # mean, log_std

        # Initialize weights
        self._init_weights()

        # Episode buffer for learning
        self.states: List[np.ndarray] = []
        self.actions: List[Tuple[int, float]] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []

        # Training stats
        self.episode_count = 0
        self.total_updates = 0
        self.avg_reward_history: List[float] = []

        # Field dimensions
        self.field_width = 800
        self.field_height = 400

    def _init_weights(self) -> None:
        """Initialize network weights with Xavier initialization."""
        # Hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)

        # Movement head
        self.W_move = np.random.randn(self.hidden_size, self.move_output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b_move = np.zeros(self.move_output_size)

        # Angle head (mean and log_std)
        self.W_angle = np.random.randn(self.hidden_size, self.angle_output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b_angle = np.zeros(self.angle_output_size)

        # Value head (baseline)
        self.W_value = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.b_value = np.zeros(1)

    def _obs_to_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert observation dict to normalized feature vector."""
        # Normalize based on player perspective
        if self.player_id == 0:
            my_x = observation["player_a_x"]
            my_y = observation["player_a_y"]
            opp_x = observation["player_b_x"]
            opp_y = observation["player_b_y"]
            my_score = observation.get("score_a", 0)
            opp_score = observation.get("score_b", 0)
        else:
            my_x = observation["player_b_x"]
            my_y = observation["player_b_y"]
            opp_x = observation["player_a_x"]
            opp_y = observation["player_a_y"]
            my_score = observation.get("score_b", 0)
            opp_score = observation.get("score_a", 0)

        ball_x = observation["ball_x"]
        ball_y = observation["ball_y"]
        ball_vx = observation.get("ball_vx", 0)
        ball_vy = observation.get("ball_vy", 0)
        in_flag = 1.0 if observation.get("ball_in_flag", False) else 0.0

        # Normalize to [-1, 1] or [0, 1]
        features = np.array([
            ball_x / self.field_width - 0.5,
            ball_y / self.field_height - 0.5,
            ball_vx / 10.0,  # Normalize velocity
            ball_vy / 10.0,
            my_x / self.field_width - 0.5,
            my_y / self.field_height - 0.5,
            opp_x / self.field_width - 0.5,
            opp_y / self.field_height - 0.5,
            in_flag,
            my_score / 11.0,
            opp_score / 11.0,
        ], dtype=np.float32)

        return features

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def _forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass through network.

        Returns:
            move_probs: Probability distribution over 17 movement actions
            angle_params: [mean, log_std] for hit angle
            value: State value estimate
        """
        # Hidden layer
        hidden = self._relu(features @ self.W1 + self.b1)

        # Movement head
        move_logits = hidden @ self.W_move + self.b_move
        move_probs = self._softmax(move_logits)

        # Angle head
        angle_params = hidden @ self.W_angle + self.b_angle
        # Scale mean based on player facing direction
        # Player A (id=0) faces right: center at 0°, range ±45°
        # Player B (id=1) faces left: center at 180°, range ±45°
        base_angle = 0.0 if self.player_id == 0 else 180.0
        angle_params = angle_params.copy()  # Avoid modifying original
        angle_params[0] = base_angle + np.tanh(angle_params[0]) * 45.0  # Mean in [base-45, base+45]
        angle_params[1] = np.clip(angle_params[1], -1, 1)  # Log std in [-1, 1] for stability

        # Value head
        value = (hidden @ self.W_value + self.b_value)[0]

        return move_probs, angle_params, value

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Choose action based on current policy."""
        features = self._obs_to_features(observation)
        move_probs, angle_params, value = self._forward(features)

        # Sample movement action
        move_action = np.random.choice(len(move_probs), p=move_probs)

        # Sample hit angle from Gaussian
        angle_mean = angle_params[0]
        angle_std = np.exp(angle_params[1]) + 0.1  # Add minimum std for stability
        hit_angle = np.random.normal(angle_mean, angle_std)
        # No need to wrap - game.py clamps to valid range

        # Calculate log probability for learning
        move_log_prob = np.log(move_probs[move_action] + 1e-10)
        angle_log_prob = -0.5 * ((hit_angle - angle_mean) / angle_std) ** 2 - np.log(angle_std) - 0.5 * np.log(2 * np.pi)
        total_log_prob = move_log_prob + angle_log_prob

        # Store for learning
        self.states.append(features)
        self.actions.append((move_action, hit_angle))
        self.log_probs.append(total_log_prob)

        return (int(move_action), float(hit_angle))

    def learn(self, reward: float, done: bool) -> None:
        """Update based on reward."""
        self.rewards.append(reward)

        if done:
            self._update()
            self.episode_count += 1

    def _update(self) -> None:
        """Perform policy gradient update."""
        if len(self.rewards) == 0:
            return

        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate gradients and update
        states = np.array(self.states)

        for i, (features, (move_action, hit_angle), log_prob, G) in enumerate(
            zip(self.states, self.actions, self.log_probs, returns)
        ):
            # Forward pass
            hidden = self._relu(features @ self.W1 + self.b1)
            move_logits = hidden @ self.W_move + self.b_move
            move_probs = self._softmax(move_logits)
            angle_params_raw = hidden @ self.W_angle + self.b_angle
            base_angle = 0.0 if self.player_id == 0 else 180.0
            angle_mean = base_angle + np.tanh(angle_params_raw[0]) * 45.0
            angle_log_std = np.clip(angle_params_raw[1], -1, 1)
            value = (hidden @ self.W_value + self.b_value)[0]

            # Advantage (clipped for stability)
            advantage = np.clip(G - value, -10, 10)

            # Policy gradient for movement (simplified)
            move_grad = np.zeros(self.move_output_size)
            move_grad[move_action] = 1.0 - move_probs[move_action]
            move_grad *= advantage * self.learning_rate

            # Update movement weights (with gradient clipping)
            move_update = np.outer(hidden, move_grad)
            move_update = np.clip(move_update, -1, 1)
            self.W_move += move_update
            self.b_move += np.clip(move_grad, -1, 1)

            # Update angle weights (simplified gradient with numerical stability)
            angle_std = np.exp(angle_log_std) + 0.1  # Minimum std for stability
            angle_diff = hit_angle - angle_mean
            angle_diff = np.clip(angle_diff, -90, 90)  # Clip difference

            angle_grad = np.zeros(2)
            # Gradient for mean (through tanh)
            tanh_grad = 1.0 - np.tanh(angle_params_raw[0]) ** 2
            angle_grad[0] = (angle_diff / (angle_std ** 2)) * advantage * self.learning_rate * 45.0 * tanh_grad
            # Gradient for log_std
            normalized_sq = (angle_diff / angle_std) ** 2
            angle_grad[1] = (normalized_sq - 1) * advantage * self.learning_rate

            # Clip gradients
            angle_grad = np.clip(angle_grad, -1, 1)
            angle_update = np.outer(hidden, angle_grad)
            angle_update = np.clip(angle_update, -1, 1)
            self.W_angle += angle_update
            self.b_angle += angle_grad

            # Update value weights
            value_grad = np.clip(advantage * self.learning_rate, -1, 1)
            self.W_value += hidden.reshape(-1, 1) * value_grad
            self.b_value += value_grad

        # Record stats
        avg_reward = np.mean(self.rewards)
        self.avg_reward_history.append(avg_reward)
        self.total_updates += 1

        # Clear episode buffer
        self.reset()

    def reset(self) -> None:
        """Reset episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def set_field_dimensions(self, width: float, height: float) -> None:
        """Set field dimensions for normalization."""
        self.field_width = width
        self.field_height = height

    def _save_weights(self, directory: Path) -> None:
        """Save network weights."""
        weights = {
            "W1": self.W1,
            "b1": self.b1,
            "W_move": self.W_move,
            "b_move": self.b_move,
            "W_angle": self.W_angle,
            "b_angle": self.b_angle,
            "W_value": self.W_value,
            "b_value": self.b_value,
            "episode_count": self.episode_count,
            "total_updates": self.total_updates,
            "avg_reward_history": self.avg_reward_history,
        }
        with open(directory / "weights.pkl", "wb") as f:
            pickle.dump(weights, f)

    def _load_weights(self, directory: Path) -> None:
        """Load network weights."""
        weights_path = directory / "weights.pkl"
        if weights_path.exists():
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)

            self.W1 = weights["W1"]
            self.b1 = weights["b1"]
            self.W_move = weights["W_move"]
            self.b_move = weights["b_move"]
            self.W_angle = weights["W_angle"]
            self.b_angle = weights["b_angle"]
            self.W_value = weights["W_value"]
            self.b_value = weights["b_value"]
            self.episode_count = weights.get("episode_count", 0)
            self.total_updates = weights.get("total_updates", 0)
            self.avg_reward_history = weights.get("avg_reward_history", [])

    def get_info(self) -> Dict[str, Any]:
        """Get agent info with training stats."""
        info = super().get_info()
        info.update({
            "strategy": "Policy gradient neural network",
            "episodes_trained": self.episode_count,
            "total_updates": self.total_updates,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
        })
        if self.avg_reward_history:
            info["recent_avg_reward"] = np.mean(self.avg_reward_history[-10:])
        return info
