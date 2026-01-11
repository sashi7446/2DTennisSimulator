# Agent Development Guide (LLM-Optimized)

2DTennisSimulator 用エージェント開発ガイド。これ1枚で実装可能。

---

## Game Rules

- 1ポイント = 1エピソード
- ボールが壁に到達 → ポイント終了
- `is_in=True` で壁到達 → 打った側が得点
- `is_in=False` で壁到達 → 打った側が失点（アウト）
- `is_in`: ボールがコート中央のインエリアを通過するとTrueになる

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
