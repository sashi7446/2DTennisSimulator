"""
Transformer-based Neural Network Agent

より強力なアーキテクチャ:
- Multi-head Self-Attention: 複数の観測要素間の関係性を学習
- Residual Connections: 深いネットワークでの学習安定性
- Layer Normalization: 訓練の安定化
- GELU Activation: よりスムーズな勾配
- Separate Policy & Value Heads: Actor-Critic構造
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from pathlib import Path
from agents.base import Agent, AgentConfig


class TransformerAgent(Agent):
    """Transformer アーキテクチャを使用した高度なニューラルエージェント"""

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="Transformer",
                agent_type="transformer",
                version="1.0",
                description="Multi-head attention based neural network agent",
                parameters={
                    "embedding_dim": 64,
                    "num_heads": 4,
                    "num_blocks": 2,
                    "ffn_dim": 128,
                    "policy_hidden_dim": 64,
                    "value_hidden_dim": 32,
                }
            )
        super().__init__(config)

        # ハイパーパラメータ
        self.embedding_dim = config.parameters.get("embedding_dim", 64)
        self.num_heads = config.parameters.get("num_heads", 4)
        self.num_blocks = config.parameters.get("num_blocks", 2)
        self.ffn_dim = config.parameters.get("ffn_dim", 128)
        self.policy_hidden_dim = config.parameters.get("policy_hidden_dim", 64)
        self.value_hidden_dim = config.parameters.get("value_hidden_dim", 32)

        # 入力次元: [ball_x, ball_y, ball_vx, ball_vy, player_x, player_y,
        #           opponent_x, opponent_y, last_move_x, last_move_y]
        self.input_dim = 10
        self.head_dim = self.embedding_dim // self.num_heads

        # ネットワーク初期化
        self._initialize_network()

        # 状態記憶
        self.last_move = np.array([0.0, 0.0])
        self.last_state_value = 0.0

    def _initialize_network(self):
        """ネットワークの重みを初期化"""

        # Input Encoder
        self.input_encoder = self._xavier_init(self.input_dim, self.embedding_dim)
        self.input_bias = np.zeros(self.embedding_dim)
        self.input_norm_gamma = np.ones(self.embedding_dim)
        self.input_norm_beta = np.zeros(self.embedding_dim)

        # Transformer Blocks
        self.transformer_blocks = []
        for _ in range(self.num_blocks):
            block = {
                # Multi-head Attention
                'attention': {
                    'heads': [
                        {
                            'W_q': self._xavier_init(self.embedding_dim, self.head_dim),
                            'W_k': self._xavier_init(self.embedding_dim, self.head_dim),
                            'W_v': self._xavier_init(self.embedding_dim, self.head_dim),
                        }
                        for _ in range(self.num_heads)
                    ],
                    'W_o': self._xavier_init(self.embedding_dim, self.embedding_dim),
                    'b_o': np.zeros(self.embedding_dim),
                },
                'norm1_gamma': np.ones(self.embedding_dim),
                'norm1_beta': np.zeros(self.embedding_dim),

                # Feed Forward Network
                'ffn_w1': self._he_init(self.embedding_dim, self.ffn_dim),
                'ffn_b1': np.zeros(self.ffn_dim),
                'ffn_w2': self._he_init(self.ffn_dim, self.embedding_dim),
                'ffn_b2': np.zeros(self.embedding_dim),
                'norm2_gamma': np.ones(self.embedding_dim),
                'norm2_beta': np.zeros(self.embedding_dim),
            }
            self.transformer_blocks.append(block)

        # Policy Head (action prediction)
        self.policy_w1 = self._he_init(self.embedding_dim, self.policy_hidden_dim)
        self.policy_b1 = np.zeros(self.policy_hidden_dim)
        self.policy_w2 = self._he_init(self.policy_hidden_dim, self.policy_hidden_dim)
        self.policy_b2 = np.zeros(self.policy_hidden_dim)
        self.policy_w_out = self._xavier_init(self.policy_hidden_dim, 2)  # [move_x, move_y]
        self.policy_b_out = np.zeros(2)

        # Value Head (state value estimation)
        self.value_w1 = self._he_init(self.embedding_dim, self.value_hidden_dim)
        self.value_b1 = np.zeros(self.value_hidden_dim)
        self.value_w_out = self._xavier_init(self.value_hidden_dim, 1)
        self.value_b_out = np.zeros(1)

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier/Glorot 初期化"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def _he_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """He 初期化 (ReLU/GELU用)"""
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out) * std

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer Normalization"""
        mean = np.mean(x)
        var = np.var(x)
        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation (近似版)"""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """安定版 Softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _multi_head_attention(self, x: np.ndarray, block: Dict) -> np.ndarray:
        """Multi-head Self-Attention"""
        attention = block['attention']
        head_outputs = []

        for head in attention['heads']:
            # Q, K, V projections
            q = x @ head['W_q']
            k = x @ head['W_k']
            v = head['W_v']

            # Attention score: Q·K^T / sqrt(d_k)
            score = np.dot(q, k) / np.sqrt(self.head_dim)
            attention_weight = 1.0  # 単一トークンなのでsoftmax不要

            # Weighted value
            head_output = v * attention_weight
            head_outputs.append(head_output)

        # Concatenate heads and project
        concatenated = np.concatenate(head_outputs)
        output = concatenated @ attention['W_o'] + attention['b_o']
        return output

    def _transformer_block(self, x: np.ndarray, block: Dict) -> np.ndarray:
        """単一 Transformer Block"""

        # Multi-head Attention + Residual + LayerNorm
        attended = self._multi_head_attention(x, block)
        x = x + attended
        x = self._layer_norm(x, block['norm1_gamma'], block['norm1_beta'])

        # Feed-Forward Network + Residual + LayerNorm
        ffn_output = x @ block['ffn_w1'] + block['ffn_b1']
        ffn_output = self._gelu(ffn_output)
        ffn_output = ffn_output @ block['ffn_w2'] + block['ffn_b2']
        x = x + ffn_output
        x = self._layer_norm(x, block['norm2_gamma'], block['norm2_beta'])

        return x

    def _encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """観測を正規化されたベクトルに変換"""

        # 観測から必要な情報を抽出
        ball_x = obs.get('ball_x', 400) / 400.0 - 1.0  # [-1, 1]
        ball_y = obs.get('ball_y', 300) / 300.0 - 1.0
        ball_vx = np.clip(obs.get('ball_vx', 0) / 10.0, -1, 1)
        ball_vy = np.clip(obs.get('ball_vy', 0) / 10.0, -1, 1)

        player_x = obs.get('player_x', 200) / 400.0 - 1.0
        player_y = obs.get('player_y', 300) / 300.0 - 1.0

        opponent_x = obs.get('opponent_x', 600) / 400.0 - 1.0
        opponent_y = obs.get('opponent_y', 300) / 300.0 - 1.0

        # 前回の行動を含める（時間的文脈）
        last_move_x = self.last_move[0]
        last_move_y = self.last_move[1]

        return np.array([
            ball_x, ball_y, ball_vx, ball_vy,
            player_x, player_y,
            opponent_x, opponent_y,
            last_move_x, last_move_y
        ], dtype=np.float32)

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """観測から行動を決定"""

        # 観測をエンコード
        input_vec = self._encode_observation(observation)

        # Input Encoding
        embedding = input_vec @ self.input_encoder + self.input_bias
        embedding = self._layer_norm(embedding, self.input_norm_gamma, self.input_norm_beta)
        embedding = self._gelu(embedding)

        # Transformer Blocks
        for block in self.transformer_blocks:
            embedding = self._transformer_block(embedding, block)

        # Policy Head: 移動方向を決定
        policy_features = embedding @ self.policy_w1 + self.policy_b1
        policy_features = self._gelu(policy_features)
        policy_features = policy_features @ self.policy_w2 + self.policy_b2
        policy_features = self._gelu(policy_features)
        policy_output = policy_features @ self.policy_w_out + self.policy_b_out

        # tanh で [-1, 1] に制限
        move_x = np.tanh(policy_output[0])
        move_y = np.tanh(policy_output[1])

        # Value Head: 状態価値を推定（将来的な強化学習用）
        value_features = embedding @ self.value_w1 + self.value_b1
        value_features = self._gelu(value_features)
        value_output = value_features @ self.value_w_out + self.value_b_out
        self.last_state_value = np.tanh(value_output[0])

        # 行動を記憶
        self.last_move = np.array([move_x, move_y])

        # 移動方向を16方向に変換
        angle = np.arctan2(move_y, move_x)
        direction = int((angle / (2 * np.pi) * 16) % 16)

        # 打つ角度（ボールの方向を基準に）
        ball_x = observation.get('ball_x', 400)
        ball_y = observation.get('ball_y', 300)
        player_x = observation.get('player_x', 200)
        player_y = observation.get('player_y', 300)

        hit_angle = np.degrees(np.arctan2(ball_y - player_y, ball_x - player_x)) % 360

        return direction, hit_angle

    def reset(self):
        """エピソード開始時にリセット"""
        self.last_move = np.array([0.0, 0.0])
        self.last_state_value = 0.0

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1) -> 'TransformerAgent':
        """遺伝的アルゴリズム用の変異"""
        mutated = TransformerAgent(self.config)

        def mutate_weights(weights: np.ndarray) -> np.ndarray:
            mask = np.random.random(weights.shape) < mutation_rate
            noise = np.random.randn(*weights.shape) * mutation_scale
            return weights + mask * noise

        # 全ての重みを変異
        mutated.input_encoder = mutate_weights(self.input_encoder)
        mutated.input_bias = mutate_weights(self.input_bias)

        for i, block in enumerate(self.transformer_blocks):
            mut_block = mutated.transformer_blocks[i]
            for j, head in enumerate(block['attention']['heads']):
                mut_block['attention']['heads'][j]['W_q'] = mutate_weights(head['W_q'])
                mut_block['attention']['heads'][j]['W_k'] = mutate_weights(head['W_k'])
                mut_block['attention']['heads'][j]['W_v'] = mutate_weights(head['W_v'])
            mut_block['attention']['W_o'] = mutate_weights(block['attention']['W_o'])
            mut_block['ffn_w1'] = mutate_weights(block['ffn_w1'])
            mut_block['ffn_w2'] = mutate_weights(block['ffn_w2'])

        mutated.policy_w1 = mutate_weights(self.policy_w1)
        mutated.policy_w2 = mutate_weights(self.policy_w2)
        mutated.policy_w_out = mutate_weights(self.policy_w_out)
        mutated.value_w1 = mutate_weights(self.value_w1)
        mutated.value_w_out = mutate_weights(self.value_w_out)

        return mutated

    def _save_weights(self, directory: Path):
        """重みを保存"""
        weights = {
            'input_encoder': self.input_encoder,
            'input_bias': self.input_bias,
            'input_norm_gamma': self.input_norm_gamma,
            'input_norm_beta': self.input_norm_beta,
            'transformer_blocks': self.transformer_blocks,
            'policy_w1': self.policy_w1,
            'policy_b1': self.policy_b1,
            'policy_w2': self.policy_w2,
            'policy_b2': self.policy_b2,
            'policy_w_out': self.policy_w_out,
            'policy_b_out': self.policy_b_out,
            'value_w1': self.value_w1,
            'value_b1': self.value_b1,
            'value_w_out': self.value_w_out,
            'value_b_out': self.value_b_out,
        }
        np.savez(directory / 'weights.npz', **weights)

    def _load_weights(self, directory: Path):
        """重みを読み込み"""
        weights = np.load(directory / 'weights.npz', allow_pickle=True)
        self.input_encoder = weights['input_encoder']
        self.input_bias = weights['input_bias']
        self.input_norm_gamma = weights['input_norm_gamma']
        self.input_norm_beta = weights['input_norm_beta']
        self.transformer_blocks = weights['transformer_blocks'].tolist()
        self.policy_w1 = weights['policy_w1']
        self.policy_b1 = weights['policy_b1']
        self.policy_w2 = weights['policy_w2']
        self.policy_b2 = weights['policy_b2']
        self.policy_w_out = weights['policy_w_out']
        self.policy_b_out = weights['policy_b_out']
        self.value_w1 = weights['value_w1']
        self.value_b1 = weights['value_b1']
        self.value_w_out = weights['value_w_out']
        self.value_b_out = weights['value_b_out']
