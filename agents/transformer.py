"""
Transformer-based Neural Network Agent

より強力なアーキテクチャ:
- Multi-head Self-Attention: 複数の観測要素間の関係性を学習
- Residual Connections: 深いネットワークでの学習安定性
- Layer Normalization: 訓練の安定化
- GELU Activation: よりスムーズな勾配
- Separate Policy & Value Heads: Actor-Critic構造
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

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
                    "embedding_dim": 160,  # さらに知能拡張 (128 -> 160)
                    "num_heads": 8,
                    "num_blocks": 3,
                    "ffn_dim": 320,
                    "policy_hidden_dim": 160,
                    "value_hidden_dim": 80,
                },
            )
        super().__init__(config)

        # ハイパーパラメータ
        self.embedding_dim = config.parameters.get("embedding_dim", 64)
        self.num_heads = config.parameters.get("num_heads", 4)
        self.num_blocks = config.parameters.get("num_blocks", 2)
        self.ffn_dim = config.parameters.get("ffn_dim", 128)
        self.policy_hidden_dim = config.parameters.get("policy_hidden_dim", 64)
        self.value_hidden_dim = config.parameters.get("value_hidden_dim", 32)

        # 入力次元: [dx, dy, b_vx, b_vy, m_vx, m_vy, odx, ody, dist, angle, last_mx, last_my, rel_vx, rel_vy, pred_y, my_x, my_y]
        self.input_dim = 17
        self.head_dim = self.embedding_dim // self.num_heads

        # ネットワーク初期化
        self.seq_len = 30
        self.pos_embedding = None
        self._initialize_network()

        # 状態記憶
        self.last_pos = None
        self.last_move = np.array([0.0, 0.0])
        self.last_state_value = 0.0
        self.step_counter = 0
        self.temp = 0.8
        self.history = []

        # --- 魔改造 V3.7 Interceptor-Prime v4 ---
        self.step_proximity_sum = 0.0
        self.foot_reward_confirmed = 0.0
        self.jitter_penalty_sum = 0.0
        self.x_drift_penalty_sum = 0.0
        self.min_ball_dist = float("inf")  # ログ用
        self.min_dist_y_err = 0.0  # ログ用
        self.episode_hit_count = 0
        self.cumulative_reward = 0.0
        self.best_score = -float("inf")
        self.elite_pool = []
        self.stagnation_counter = 0
        self.best_weights = None
        # ------------------------------------

    def _initialize_network(self):
        """ネットワークの重みを初期化"""
        # Positional Embedding
        self.pos_embedding = self._xavier_init(self.seq_len, self.embedding_dim)

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
                "attention": {
                    "heads": [
                        {
                            "W_q": self._xavier_init(self.embedding_dim, self.head_dim),
                            "W_k": self._xavier_init(self.embedding_dim, self.head_dim),
                            "W_v": self._xavier_init(self.embedding_dim, self.head_dim),
                        }
                        for _ in range(self.num_heads)
                    ],
                    "W_o": self._xavier_init(self.embedding_dim, self.embedding_dim),
                    "b_o": np.zeros(self.embedding_dim),
                },
                "norm1_gamma": np.ones(self.embedding_dim),
                "norm1_beta": np.zeros(self.embedding_dim),
                # Feed Forward Network
                "ffn_w1": self._he_init(self.embedding_dim, self.ffn_dim),
                "ffn_b1": np.zeros(self.ffn_dim),
                "ffn_w2": self._he_init(self.ffn_dim, self.embedding_dim),
                "ffn_b2": np.zeros(self.embedding_dim),
                "norm2_gamma": np.ones(self.embedding_dim),
                "norm2_beta": np.zeros(self.embedding_dim),
            }
            self.transformer_blocks.append(block)

        # Policy Head (action prediction)
        self.policy_w1 = self._he_init(self.embedding_dim, self.policy_hidden_dim)
        self.policy_b1 = np.zeros(self.policy_hidden_dim)
        self.policy_w2 = self._he_init(self.policy_hidden_dim, self.policy_hidden_dim)
        self.policy_b2 = np.zeros(self.policy_hidden_dim)
        self.policy_w_out = self._xavier_init(
            self.policy_hidden_dim, 3
        )  # [move_x, move_y, hit_angle]
        self.policy_b_out = np.zeros(3)

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

    def _layer_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Layer Normalization (2D/シーケンス対応)"""
        # トークン（行）ごとに平均と分散を計算
        if x.ndim == 1:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)

        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation (近似版)"""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """安定版 Softmax (2D/シーケンス対応: 行方向に適用)"""
        if x.ndim == 1:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            max_val = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - max_val)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _multi_head_attention(self, x: np.ndarray, block: Dict) -> np.ndarray:
        """Multi-head Self-Attention (シーケンス間の関係性を計算)"""
        attention = block["attention"]
        head_outputs = []

        for head in attention["heads"]:
            # Q, K, V projections (Seq, Dim) @ (Dim, Head) -> (Seq, Head)
            q = x @ head["W_q"]
            k = x @ head["W_k"]
            v = x @ head["W_v"]

            # Attention score: (Seq, Head) @ (Head, Seq) -> (Seq, Seq)
            score = (q @ k.T) / np.sqrt(self.head_dim)
            # 全ての過去の瞬間に対して重みを振る
            attention_weights = self._softmax(score)

            # Weighted value: (Seq, Seq) @ (Seq, Head) -> (Seq, Head)
            head_output = attention_weights @ v
            head_outputs.append(head_output)

        # Concatenate heads along axis 1 (Dim): (Seq, Dim)
        concatenated = np.concatenate(head_outputs, axis=1)
        output = concatenated @ attention["W_o"] + attention["b_o"]
        return output

    def _transformer_block(self, x: np.ndarray, block: Dict) -> np.ndarray:
        """単一 Transformer Block"""

        # Multi-head Attention + Residual + LayerNorm
        attended = self._multi_head_attention(x, block)
        x = x + attended
        x = self._layer_norm(x, block["norm1_gamma"], block["norm1_beta"])

        # Feed-Forward Network + Residual + LayerNorm
        ffn_output = x @ block["ffn_w1"] + block["ffn_b1"]
        ffn_output = self._gelu(ffn_output)
        ffn_output = ffn_output @ block["ffn_w2"] + block["ffn_b2"]
        x = x + ffn_output
        x = self._layer_norm(x, block["norm2_gamma"], block["norm2_beta"])

        return x

    def _encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """【V3 Grandmaster】10次元の究極特徴抽出（相手の位置を把握）"""
        is_a = self.player_id == 0
        my_x = obs["player_a_x" if is_a else "player_b_x"]
        my_y = obs["player_a_y" if is_a else "player_b_y"]
        opp_x = obs["player_b_x" if is_a else "player_a_x"]
        opp_y = obs["player_b_y" if is_a else "player_a_y"]
        ball_x = obs["ball_x"]
        ball_y = obs["ball_y"]
        b_vx = obs["ball_vx"]
        b_vy = obs["ball_vy"]

        # 速度算出
        if self.last_pos is None:
            my_vx, my_vy = 0.0, 0.0
        else:
            my_vx = my_x - self.last_pos[0]
            my_vy = my_y - self.last_pos[1]
        self.last_pos = (my_x, my_y)

        # 各種正規化
        dx = (ball_x - my_x) / 800.0
        dy = (ball_y - my_y) / 400.0
        odx = (opp_x - my_x) / 800.0
        ody = (opp_y - my_y) / 400.0
        v_norm = 15.0

        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) / np.pi

        # 相対速度 (Interception用)
        rel_vx = (b_vx - my_vx) / v_norm
        rel_vy = (b_vy - my_vy) / v_norm

        # インターセプト予測 (未来予知：ベースライン上での交点)
        # 設定変更に合わせてターゲットを A:37.5, B:762.5 に修正
        target_x = 37.5 if self.player_id == 0 else 762.5
        if abs(b_vx) > 0.1:
            time_to_reach = (target_x - ball_x) / b_vx
            if time_to_reach > 0:
                pred_y = ball_y + b_vy * time_to_reach
                pred_val = (pred_y - my_y) / 400.0  # Yの相対偏差
            else:
                pred_val = 0.0
        else:
            pred_val = 0.0

        return np.array(
            [
                dx,
                dy,
                b_vx / v_norm,
                b_vy / v_norm,
                my_vx / v_norm,
                my_vy / v_norm,
                odx,
                ody,
                dist,
                angle,
                self.last_move[0],
                self.last_move[1],
                rel_vx,
                rel_vy,
                pred_val,
                my_x / 800.0,
                my_y / 400.0,  # 自己座標を追加して空間認識を強化
            ],
            dtype=np.float32,
        )

    def act(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """観測から行動を決定 (時系列シーケンス処理)"""

        # 1. 観測をエンコードして履歴に保存
        input_vec = self._encode_observation(observation)
        self.history.append(input_vec)
        if len(self.history) > self.seq_len:
            self.history.pop(0)

        # 足りない場合はパディング（初期動作用）
        while len(self.history) < self.seq_len:
            self.history.insert(0, input_vec)

        # (Seq, InputDim) の行列に変換
        tokens = np.array(self.history, dtype=np.float32)

        # 2. Input Projection & Positional Embedding
        # (Seq, InputDim) @ (InputDim, Dim) -> (Seq, Dim)
        embedding = tokens @ self.input_encoder + self.input_bias
        # 時間の情報を足す
        embedding = embedding + self.pos_embedding
        embedding = self._layer_norm(embedding, self.input_norm_gamma, self.input_norm_beta)
        embedding = self._gelu(embedding)

        # 3. Transformer Blocks (Seq, Dim) -> (Seq, Dim)
        for block in self.transformer_blocks:
            embedding = self._transformer_block(embedding, block)

        # 4. 最後のトークン（＝現在の状況）のみを抽出して意思決定に使用
        last_token = embedding[-1]

        # Policy Head: [move_x, move_y, hit_angle_offset]
        policy_features = last_token @ self.policy_w1 + self.policy_b1
        policy_features = self._gelu(policy_features)
        policy_features = policy_features @ self.policy_w2 + self.policy_b2
        policy_features = self._gelu(policy_features)
        policy_output = policy_features @ self.policy_w_out + self.policy_b_out

        move_x = np.tanh(policy_output[0] / self.temp)
        move_y = np.tanh(policy_output[1] / self.temp)
        # 打球角度を自律制御 (-90度〜90度のオフセット)
        angle_offset = np.tanh(policy_output[2] / self.temp) * 90.0

        # Value Head
        value_features = last_token @ self.value_w1 + self.value_b1
        value_features = self._gelu(value_features)
        value_output = value_features @ self.value_w_out + self.value_b_out
        self.last_state_value = np.tanh(value_output[0])

        # 指数関数的・重力報酬 (距離が近いほど爆発的に増える)
        dx, dy = input_vec[0], input_vec[1]
        step_dist = np.sqrt(dx**2 + dy**2)
        self.step_proximity_sum += np.exp(-step_dist * 10.0) * 3.0

        # ジグザグ抑制
        current_move = np.array([move_x, move_y])
        if np.linalg.norm(self.last_move) > 0.1 and np.linalg.norm(current_move) > 0.1:
            dot = np.dot(self.last_move, current_move)
            if dot < 0:
                self.jitter_penalty_sum += 0.8

        # X軸(前後)ドリフト抑制 (ベースライン 37.5 or 762.5 に固定)
        target_x = 37.5 if self.player_id == 0 else 762.5
        my_x = observation["player_a_x" if self.player_id == 0 else "player_b_x"]
        x_diff = abs(my_x - target_x) / 800.0
        self.x_drift_penalty_sum += x_diff * 5.0  # さらに厳罰化

        # 最小接近距離の記録 (ログ精度向上)
        ball_dist = np.sqrt(dx**2 + dy**2) * 800.0  # px単位
        if ball_dist < self.min_ball_dist:
            self.min_ball_dist = ball_dist
            self.min_dist_y_err = abs(input_vec[-3] * 400.0)  # その時のY偏差

        # 行動を記憶
        self.last_move = current_move

        # --- 情緒的な独り言 (500回に1回) ---
        self.step_counter += 1
        if self.step_counter % 500 == 0:
            thoughts = [
                "ボールの軌道が見える...",
                "もっと速く...！",
                "ここだ！",
                "今の動き、悪くないな",
                "次は外さない",
            ]
            print(f"  🤖 (独り言): {np.random.choice(thoughts)}")

        # 移動方向を16方向に変換
        move_angle = np.arctan2(move_y, move_x)
        direction = int((move_angle / (2 * np.pi) * 16) % 16)

        # 打つ角度の決定（ベースとなるネット方向 + 自律オフセット）
        is_a = self.player_id == 0
        base_angle = 0 if is_a else 180  # 相手側コートの方向
        hit_angle = (base_angle + angle_offset) % 360

        return direction, hit_angle

    def learn(self, reward: float, done: bool) -> None:
        """【V3.7 Interceptor-Prime v3】ベースライン・ハンター。ヒット失敗を恐れず、Y軸の偏差を埋める。"""
        # 1. 自分の打球を検知（その瞬間に「成功」を確定）
        if 0.05 < reward < 0.5:
            self.episode_hit_count += 1
            # 触れた瞬間は特大ボーナス
            self.foot_reward_confirmed += 500.0

        if done:
            # --- Prime 報酬体系 v3 ---
            # A. 足（移動）の評価: 接近ボーナスをそのまま使う（不連続な 0.05 掛けを廃止）
            # これにより「惜しい失敗」が正当に評価され、山登りしやすくなる
            foot_score = self.foot_reward_confirmed + (self.step_proximity_sum * 0.1)

            # B. 物理的安定性 (X軸の無駄な動きをより厳しく)
            penalty_score = -(self.jitter_penalty_sum + self.x_drift_penalty_sum)

            total_score = foot_score + penalty_score

            # --- 日本語リアルタイム分析 ---
            if self.episode_hit_count > 0:
                print(f"  🏎️ 【物理極致】インターセプト成功！ スコア: {total_score:.1f}")
            else:
                res = "失点" if self.cumulative_reward < -0.5 else "ラッキー"
                # 最も近かった瞬間の Y 偏差を表示 (0.0px バグの解消)
                print(
                    f"  ❌ 捕球失敗... 最低接近ズレ: {self.min_dist_y_err:.1f}px ({res}) スコア: {total_score:.1f}"
                )

            # --- エリート・進化管理 ---
            self.stagnation_counter += 1
            is_record = False

            if not self.elite_pool or total_score > self.best_score:
                if self.elite_pool:
                    print(
                        f"  🏁 【足AI覚醒】最接近を更新！ ({self.best_score:.1f} -> {total_score:.1f})"
                    )
                self.best_score = total_score
                self.elite_pool.insert(0, (total_score, self._get_all_weights()))
                self.elite_pool = self.elite_pool[:3]
                self.stagnation_counter = 0
                is_record = True

            # 3. 復元と変異 (ヒット率だけでなく、接近そのものを評価)
            if not is_record:
                if self.stagnation_counter > 20:
                    if len(self.elite_pool) > 1 and np.random.random() < 0.4:
                        print("  🧪 接近性能をブレンド...")
                        w1 = self.elite_pool[0][1]
                        w2 = self.elite_pool[1][1]
                        blended = self._blend_weights(w1, w2, ratio=0.5)
                        self._set_all_weights(blended)
                    elif self.elite_pool:
                        self._set_all_weights(self.elite_pool[0][1])
                    self.stagnation_counter = 0

            # 4. 変異 (足が完成するまでは大胆に振る)
            m_scale = 0.002 if is_record else 0.05
            if self.stagnation_counter > 10:
                m_scale *= 10

            self._apply_inplace_mutation(mutation_rate=0.2, mutation_scale=m_scale)

            # リセット
            self.cumulative_reward = 0.0
            self.step_proximity_sum = 0.0
            self.foot_reward_confirmed = 0.0
            self.jitter_penalty_sum = 0.0
            self.x_drift_penalty_sum = 0.0
            self.episode_hit_count = 0

    def _blend_weights(
        self, w1: Dict[str, Any], w2: Dict[str, Any], ratio: float
    ) -> Dict[str, Any]:
        """2つの重みセットをブレンドする"""
        blended = {}
        for k in w1:
            if isinstance(w1[k], dict):
                blended[k] = self._blend_weights(w1[k], w2[k], ratio)
            elif isinstance(w1[k], list):
                blended[k] = [
                    (
                        self._blend_weights(i1, i2, ratio)
                        if isinstance(i1, dict)
                        else i1 * ratio + i2 * (1 - ratio)
                    )
                    for i1, i2 in zip(w1[k], w2[k])
                ]
            else:
                blended[k] = w1[k] * ratio + w2[k] * (1.0 - ratio)
        return blended

    def _apply_inplace_mutation(self, mutation_rate: float, mutation_scale: float):
        """現在の重みを直接変異させる"""

        def mutate(w):
            mask = np.random.random(w.shape) < mutation_rate
            noise = np.random.randn(*w.shape) * mutation_scale
            return w + mask * noise

        self.input_encoder = mutate(self.input_encoder)
        self.input_bias = mutate(self.input_bias)
        self.pos_embedding = mutate(self.pos_embedding)

        # 主要な層を変異
        for block in self.transformer_blocks:
            for head in block["attention"]["heads"]:
                head["W_q"] = mutate(head["W_q"])
                head["W_k"] = mutate(head["W_k"])
                head["W_v"] = mutate(head["W_v"])
            block["attention"]["W_o"] = mutate(block["attention"]["W_o"])
            block["ffn_w1"] = mutate(block["ffn_w1"])
            block["ffn_w2"] = mutate(block["ffn_w2"])

        self.policy_w1 = mutate(self.policy_w1)
        self.policy_w_out = mutate(self.policy_w_out)

    def _get_all_weights(self) -> Dict[str, Any]:
        """現在の全重みのコピーを取得"""
        import copy

        return {
            "input_encoder": self.input_encoder.copy(),
            "input_bias": self.input_bias.copy(),
            "pos_embedding": self.pos_embedding.copy(),
            "transformer_blocks": copy.deepcopy(self.transformer_blocks),
            "policy_w1": self.policy_w1.copy(),
            "policy_w2": self.policy_w2.copy(),
            "policy_w_out": self.policy_w_out.copy(),
            "value_w1": self.value_w1.copy(),
            "value_w_out": self.value_w_out.copy(),
        }

    def _set_all_weights(self, weights: Dict[str, Any]):
        """重みを一括復元"""
        import copy

        self.input_encoder = weights["input_encoder"].copy()
        self.input_bias = weights["input_bias"].copy()
        self.pos_embedding = weights["pos_embedding"].copy()
        self.transformer_blocks = copy.deepcopy(weights["transformer_blocks"])
        self.policy_w1 = weights["policy_w1"].copy()
        self.policy_w2 = weights["policy_w2"].copy()
        self.policy_w_out = weights["policy_w_out"].copy()
        self.value_w1 = weights["value_w1"].copy()
        self.value_w_out = weights["value_w_out"].copy()

    def reset(self):
        """エピソード開始時にリセット"""
        self.last_pos = None
        self.last_move = np.array([0.0, 0.0])
        self.last_state_value = 0.0
        self.history = []  # 歴史もリセット
        self.episode_hit_count = 0  # ヒット数リセット
        self.step_proximity_sum = 0.0  # 重力報酬累積リセット
        self.foot_reward_confirmed = 0.0  # 確定足報酬リセット
        self.jitter_penalty_sum = 0.0
        self.x_drift_penalty_sum = 0.0
        self.min_ball_dist = float("inf")
        self.min_dist_y_err = 0.0

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1) -> "TransformerAgent":
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
            for j, head in enumerate(block["attention"]["heads"]):
                mut_block["attention"]["heads"][j]["W_q"] = mutate_weights(head["W_q"])
                mut_block["attention"]["heads"][j]["W_k"] = mutate_weights(head["W_k"])
                mut_block["attention"]["heads"][j]["W_v"] = mutate_weights(head["W_v"])
            mut_block["attention"]["W_o"] = mutate_weights(block["attention"]["W_o"])
            mut_block["ffn_w1"] = mutate_weights(block["ffn_w1"])
            mut_block["ffn_w2"] = mutate_weights(block["ffn_w2"])

        mutated.policy_w1 = mutate_weights(self.policy_w1)
        mutated.policy_w2 = mutate_weights(self.policy_w2)
        mutated.policy_w_out = mutate_weights(self.policy_w_out)
        mutated.value_w1 = mutate_weights(self.value_w1)
        mutated.value_w_out = mutate_weights(self.value_w_out)

        return mutated

    def _save_weights(self, directory: Path):
        """重みを保存"""
        weights = {
            "input_encoder": self.input_encoder,
            "input_bias": self.input_bias,
            "input_norm_gamma": self.input_norm_gamma,
            "input_norm_beta": self.input_norm_beta,
            "transformer_blocks": self.transformer_blocks,
            "policy_w1": self.policy_w1,
            "policy_b1": self.policy_b1,
            "policy_w2": self.policy_w2,
            "policy_b2": self.policy_b2,
            "policy_w_out": self.policy_w_out,
            "policy_b_out": self.policy_b_out,
            "value_w1": self.value_w1,
            "value_b1": self.value_b1,
            "value_w_out": self.value_w_out,
            "value_b_out": self.value_b_out,
        }
        np.savez(directory / "weights.npz", **weights)

    def _load_weights(self, directory: Path):
        """重みを読み込み"""
        weights = np.load(directory / "weights.npz", allow_pickle=True)
        self.input_encoder = weights["input_encoder"]
        self.input_bias = weights["input_bias"]
        self.input_norm_gamma = weights["input_norm_gamma"]
        self.input_norm_beta = weights["input_norm_beta"]
        self.transformer_blocks = weights["transformer_blocks"].tolist()
        self.policy_w1 = weights["policy_w1"]
        self.policy_b1 = weights["policy_b1"]
        self.policy_w2 = weights["policy_w2"]
        self.policy_b2 = weights["policy_b2"]
        self.policy_w_out = weights["policy_w_out"]
        self.policy_b_out = weights["policy_b_out"]
        self.value_w1 = weights["value_w1"]
        self.value_b1 = weights["value_b1"]
        self.value_w_out = weights["value_w_out"]
        self.value_b_out = weights["value_b_out"]
