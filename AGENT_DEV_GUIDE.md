# Agent Development Guide (LLM-Optimized)

2DTennisSimulator 用エージェント開発ガイド。これ1枚で実装可能。

---

## Court Layout

```
      0                    field_width (800)
    0 +------------------------------------------+
      |                    |                     |
      |   [A]              |              [B]    |
      |  id=0          +---+---+          id=1   |
      | (左側)         |in |in |        (右側)   |
      |  agent-a       |_A_|_B_|        agent-b  |
      |                    |                     |
  400 +------------------------------------------+
         ← A側の壁    中央(サーブ地点)    B側の壁 →
```

> **解説**: 横長コート。左にA、右にB。中央に2つのインエリア（in_A, in_B）がある。

---

## Game Flow (実況中継風)

**🎙️ 実況**: 「さあ、ポイント開始です！」

1. **サーブ** — ボールがコート中央から発射されます。左右どちらに飛ぶかはランダム。
   > **解説**: 今回は左に飛びました。A側のエージェント（id=0, agent-a）がレシーブです。

2. **打球** — ボールがプレイヤーの当たり判定（半径数ピクセルの円）に触れると、自動で打ち返します。打球角度は `action[1]` で指定した `hit_angle` の方向。
   > **解説**: このとき `is_in` が `False` にリセットされます。アウトになるかどうか、まだわかりません。

3. **インエリア通過** — ボールが相手側のインエリアを通過！
   > **解説**: 「`is_in=True` になりました！これでこのボールがB側の壁に到達すれば、Aの得点です」

4. **壁到達 → ポイント終了**
   - `is_in=True` で壁到達 → **打った側が得点！** 🎉
   - `is_in=False` で壁到達 → **打った側が失点（アウト）** 💀

**🎙️ 実況**: 「ボールがB側の壁に到達！is_in は True！A選手の得点です！」

> **解説まとめ**:
> - 1ポイント = 1エピソード
> - 打ったら `is_in=False`、相手のインエリア通過で `is_in=True`
> - 壁到達時の `is_in` で勝敗が決まる

---

## Observation (dict)

| Key | Type | Description |
|-----|------|-------------|
| `ball_x`, `ball_y` | float | ボール座標 |
| `ball_vx`, `ball_vy` | float | ボール速度 |
| `ball_is_in` | bool | インエリア通過済みか |
| `player_a_x`, `player_a_y` | float | プレイヤーA座標 |
| `player_b_x`, `player_b_y` | float | プレイヤーB座標 |
| `score_a`, `score_b` | int | スコア |
| `rally_count` | int | ラリー回数 |
| `field_width`, `field_height` | int | フィールドサイズ |

---

## Action (tuple[int, float])

```python
(movement, hit_angle)
```

- `movement`: 0-15 = 22.5°刻み16方向, 16 = 静止
- `hit_angle`: 0-360° (0=右, 90=下, 180=左, 270=上)

---

## Minimal Implementation

```python
# agents/my_agent.py
from agents.base import Agent, AgentConfig

class MyAgent(Agent):
    def __init__(self):
        super().__init__(AgentConfig(
            name="MyAgent",
            agent_type="my_agent",
            description="My custom agent"
        ))

    def act(self, obs: dict) -> tuple[int, float]:
        # Example: chase ball
        my_x = obs["player_a_x"] if self.player_id == 0 else obs["player_b_x"]
        my_y = obs["player_a_y"] if self.player_id == 0 else obs["player_b_y"]
        dx, dy = obs["ball_x"] - my_x, obs["ball_y"] - my_y

        import math
        angle = math.degrees(math.atan2(dy, dx)) % 360
        movement = int(angle / 22.5) % 16

        # hit toward opponent's side
        hit_angle = 180 if self.player_id == 0 else 0
        return (movement, hit_angle)

    def learn(self, reward: float, done: bool) -> None:
        pass  # Optional: implement learning
```

---

## Registration (3 steps)

### 1. `agents/__init__.py`
```python
from agents.my_agent import MyAgent
__all__ = [..., "MyAgent"]
```

### 2. `agents/base.py` の `get_agent_class()`
```python
classes = {
    ...,
    "my_agent": MyAgent,
}
```

### 3. Run
```bash
python main.py --agent-a my_agent --agent-b chase
```

---

## Tips

- `self.player_id`: 0=A(左側), 1=B(右側) - `set_player_id()` で自動設定
- 打撃判定はプレイヤーとボールの距離で自動判定
- `learn()` は毎ステップ呼ばれる（reward, done を受け取る）
- `reset()` はエピソード開始時に呼ばれる

---

## Test

```bash
python -m pytest tests/test_agents.py -v -k "MyAgent"
```
