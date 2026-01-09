# Architecture

読んで分からなければコードを読め。

---

## Core Engine

```
Game ─┬─ Field (壁・インエリア判定)
      ├─ Ball  (物理・in_flag管理)
      └─ Player×2 (移動・打撃判定)
```

- 1ポイント = 1エピソード
- `step()` で1フレーム進む、`StepResult` を返す

---

## Agent System

```
Agent (ABC)
  ├─ act(obs) → (movement, hit_angle)  # 必須
  ├─ learn(reward, done)               # 任意
  └─ reset()                           # 任意
```

**設計思想**: エージェントは `obs` を受け取り `action` を返すブラックボックス。内部実装は各実装者の自由。Configへの依存禁止。

---

## Debug Monitor (責務分離)

```
InputHandler → Renderer ← StatsTracker
   (状態)       (描画)      (統計)
```

| 層 | 責務 | 原則 |
|----|------|------|
| InputHandler | イベント処理、UIフラグ管理 | 状態を持つ |
| Renderer | 描画のみ | **受動的**。渡されたものを描くだけ |
| StatsTracker | 報酬・勝率の集計 | 描画から独立（ヘッドレス対応） |

**Overlay**: コンポジションで追加。`Overlay` Protocol 参照。

---

## Data Flow

```
           obs
Game ──────────→ Agent
  ↑                │
  │    action      │
  └────────────────┘

           obs, action, reward
Game ──────────────────────────→ DebugRenderer
                                      │
                         StatsTracker─┘
```

---

## Key Concepts

| 用語 | 意味 |
|------|------|
| `in_flag` | ボールがインエリアを通過済みか。Trueで打撃可能 |
| `last_hit_by` | 最後に打った人 (0=A, 1=B, None=サーブ) |
| `movement` | 0-15: 22.5°刻みの16方向、16: 静止 |
| `hit_angle` | 打球角度 (0°=右, 90°=下, 180°=左) |

---

## File Map

```
config.py        # 定数
game.py          # ゲームループ
ball.py, player.py, field.py  # 物理
env.py           # Gymnasium wrapper
agents/          # 各自勝手にやれ
renderer.py      # 描画（受動）
input_handler.py # 入力（状態）
stats_tracker.py # 統計（独立）
debug.py         # ログ（使ってない箇所多い）
```
