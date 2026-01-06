"""Neural Network Agent with Policy Gradient Learning."""

import pickle
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from agents.base import Agent, AgentConfig


@dataclass
class NeuralAgentConfig(AgentConfig):
    name: str = "NeuralBot"
    agent_type: str = "neural"
    version: str = "1.0"
    description: str = "Neural network agent with policy gradient"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 64, "learning_rate": 0.001, "gamma": 0.99, "entropy_coef": 0.01
    })


class NeuralAgent(Agent):
    """Policy gradient neural network agent (REINFORCE with baseline)."""

    def __init__(self, config: Optional[NeuralAgentConfig] = None):
        super().__init__(config or NeuralAgentConfig())
        p = self.config.parameters
        self.hidden_size = p.get("hidden_size", 64)
        self.learning_rate = p.get("learning_rate", 0.001)
        self.gamma = p.get("gamma", 0.99)
        self.entropy_coef = p.get("entropy_coef", 0.01)

        self.input_size, self.move_output_size, self.angle_output_size = 11, 17, 2
        self._init_weights()

        self.states: List[np.ndarray] = []
        self.actions: List[Tuple[int, float]] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []

        self.episode_count = 0
        self.total_updates = 0
        self.avg_reward_history: List[float] = []
        self.field_width, self.field_height = 800, 400

    def _init_weights(self) -> None:
        """Xavier initialization for network weights."""
        def xavier(fan_in, fan_out):
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

        self.W1 = xavier(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W_move = xavier(self.hidden_size, self.move_output_size)
        self.b_move = np.zeros(self.move_output_size)
        self.W_angle = xavier(self.hidden_size, self.angle_output_size)
        self.b_angle = np.zeros(self.angle_output_size)
        self.W_value = xavier(self.hidden_size, 1)
        self.b_value = np.zeros(1)

    def _obs_to_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert observation to normalized feature vector from player's perspective."""
        o = observation
        is_a = self.player_id == 0
        my_x, my_y = (o["player_a_x"], o["player_a_y"]) if is_a else (o["player_b_x"], o["player_b_y"])
        opp_x, opp_y = (o["player_b_x"], o["player_b_y"]) if is_a else (o["player_a_x"], o["player_a_y"])
        my_score = o.get("score_a" if is_a else "score_b", 0)
        opp_score = o.get("score_b" if is_a else "score_a", 0)

        return np.array([
            o["ball_x"] / self.field_width - 0.5, o["ball_y"] / self.field_height - 0.5,
            o.get("ball_vx", 0) / 10.0, o.get("ball_vy", 0) / 10.0,
            my_x / self.field_width - 0.5, my_y / self.field_height - 0.5,
            opp_x / self.field_width - 0.5, opp_y / self.field_height - 0.5,
            1.0 if o.get("ball_in_flag", False) else 0.0,
            my_score / 11.0, opp_score / 11.0,
        ], dtype=np.float32)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def _forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward pass. Returns (move_probs, angle_params, value)."""
        hidden = self._relu(features @ self.W1 + self.b1)
        move_probs = self._softmax(hidden @ self.W_move + self.b_move)

        angle_params = (hidden @ self.W_angle + self.b_angle).copy()
        base_angle = 0.0 if self.player_id == 0 else 180.0
        angle_params[0] = base_angle + np.tanh(angle_params[0]) * 45.0
        angle_params[1] = np.clip(angle_params[1], -1, 1)

        value = (hidden @ self.W_value + self.b_value)[0]
        return move_probs, angle_params, value

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """Choose action based on current policy."""
        features = self._obs_to_features(observation)
        move_probs, angle_params, _ = self._forward(features)

        move_action = np.random.choice(len(move_probs), p=move_probs)
        angle_mean, angle_std = angle_params[0], np.exp(angle_params[1]) + 0.1
        hit_angle = np.random.normal(angle_mean, angle_std)

        # Log probability for learning
        move_log_prob = np.log(move_probs[move_action] + 1e-10)
        angle_log_prob = -0.5 * ((hit_angle - angle_mean) / angle_std) ** 2 - np.log(angle_std) - 0.5 * np.log(2 * np.pi)

        self.states.append(features)
        self.actions.append((move_action, hit_angle))
        self.log_probs.append(move_log_prob + angle_log_prob)

        return (int(move_action), float(hit_angle))

    def learn(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        if done:
            self._update()
            self.episode_count += 1

    def _update(self) -> None:
        """Policy gradient update."""
        if not self.rewards:
            return

        # Discounted returns
        returns, G = [], 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        base_angle = 0.0 if self.player_id == 0 else 180.0

        for features, (move_action, hit_angle), _, G in zip(
            self.states, self.actions, self.log_probs, returns
        ):
            hidden = self._relu(features @ self.W1 + self.b1)
            move_probs = self._softmax(hidden @ self.W_move + self.b_move)
            angle_params_raw = hidden @ self.W_angle + self.b_angle
            angle_mean = base_angle + np.tanh(angle_params_raw[0]) * 45.0
            angle_log_std = np.clip(angle_params_raw[1], -1, 1)
            value = (hidden @ self.W_value + self.b_value)[0]

            advantage = np.clip(G - value, -10, 10)

            # Movement gradient
            move_grad = np.zeros(self.move_output_size)
            move_grad[move_action] = (1.0 - move_probs[move_action]) * advantage * self.learning_rate
            self.W_move += np.clip(np.outer(hidden, move_grad), -1, 1)
            self.b_move += np.clip(move_grad, -1, 1)

            # Angle gradient
            angle_std = np.exp(angle_log_std) + 0.1
            angle_diff = np.clip(hit_angle - angle_mean, -90, 90)
            tanh_grad = 1.0 - np.tanh(angle_params_raw[0]) ** 2
            angle_grad = np.array([
                (angle_diff / angle_std**2) * advantage * self.learning_rate * 45.0 * tanh_grad,
                ((angle_diff / angle_std)**2 - 1) * advantage * self.learning_rate
            ])
            angle_grad = np.clip(angle_grad, -1, 1)
            self.W_angle += np.clip(np.outer(hidden, angle_grad), -1, 1)
            self.b_angle += angle_grad

            # Value gradient
            value_grad = np.clip(advantage * self.learning_rate, -1, 1)
            self.W_value += hidden.reshape(-1, 1) * value_grad
            self.b_value += value_grad

        self.avg_reward_history.append(np.mean(self.rewards))
        self.total_updates += 1
        self.reset()

    def reset(self) -> None:
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []

    def set_field_dimensions(self, width: float, height: float) -> None:
        self.field_width, self.field_height = width, height

    def _save_weights(self, directory: Path) -> None:
        with open(directory / "weights.pkl", "wb") as f:
            pickle.dump({
                "W1": self.W1, "b1": self.b1, "W_move": self.W_move, "b_move": self.b_move,
                "W_angle": self.W_angle, "b_angle": self.b_angle,
                "W_value": self.W_value, "b_value": self.b_value,
                "episode_count": self.episode_count, "total_updates": self.total_updates,
                "avg_reward_history": self.avg_reward_history,
            }, f)

    def _load_weights(self, directory: Path) -> None:
        path = directory / "weights.pkl"
        if path.exists():
            with open(path, "rb") as f:
                w = pickle.load(f)
            for k in ["W1", "b1", "W_move", "b_move", "W_angle", "b_angle", "W_value", "b_value"]:
                setattr(self, k, w[k])
            self.episode_count = w.get("episode_count", 0)
            self.total_updates = w.get("total_updates", 0)
            self.avg_reward_history = w.get("avg_reward_history", [])

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "strategy": "Policy gradient", "episodes_trained": self.episode_count,
            "total_updates": self.total_updates, "hidden_size": self.hidden_size,
        })
        if self.avg_reward_history:
            info["recent_avg_reward"] = np.mean(self.avg_reward_history[-10:])
        return info
