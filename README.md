# 2D Tennis Simulator

AIエージェント同士が対戦しながら学習していく過程を観察するシミュレーター。
「ホームポジションに戻る」などの基本戦術が強化学習によって自然発生するか、神視点で楽しむことを目的としています。

## 特徴

- AI vs AI のリアルタイム対戦観戦
- Policy Gradient（REINFORCE）による学習エージェント
- 学習過程をグラフで可視化（デバッグモード）
- エージェントの保存・読み込み機能
- 複数のエージェントタイプ（ルールベース/学習型）

## インストール

```bash
pip install pygame numpy
```

## クイックスタート

```bash
# AI同士の対戦を観戦
python main.py

# 学習エージェント vs ルールベースAI（デバッグ表示付き）
python main.py --agent-a neural --agent-b chase --debug

# 利用可能なエージェント一覧
python main.py --mode list
```

## エージェントタイプ

| タイプ | 説明 |
|--------|------|
| `chase` | ボールを追いかけるシンプルなAI（デフォルト） |
| `smart` | 位置取りを考慮した改良版チェイスAI |
| `random` | ランダム行動（ベースライン比較用） |
| `neural` | Policy Gradientで学習するニューラルネットワークAI |

## 使い方

### 対戦観戦モード

```bash
# 基本（chase vs chase）
python main.py

# エージェント指定
python main.py --agent-a neural --agent-b smart

# デバッグモード（報酬グラフ表示）
python main.py --agent-a neural --agent-b chase --debug

# ボール速度・勝利ポイント変更
python main.py --speed 7.0 --points 21
```

### キー操作

| キー | 動作 |
|------|------|
| `D` | デバッグ表示の切り替え |
| `S` | エージェントを保存（--save-dir指定時） |
| `R` | ゲームリセット |
| `ESC` | 終了 |

### ヘッドレス学習モード

高速学習用（描画なし）：

```bash
# 100エピソード学習
python main.py --mode headless --agent-a neural --agent-b neural --episodes 100

# エージェントを保存しながら学習
python main.py --mode headless --agent-a neural --agent-b chase --episodes 500 --save-dir saved_agents
```

### エージェントの保存と読み込み

```bash
# 学習しながら保存
python main.py --agent-a neural --agent-b chase --save-dir my_agents --debug

# 保存したエージェントを読み込んで対戦
python main.py --agent-a my_agents/agent_a_neural --agent-b chase

# 保存済みエージェント同士の対戦
python main.py --agent-a saved/champion_v1 --agent-b saved/champion_v2
```

## デバッグモード

`--debug` フラグで以下の情報を表示：

- ボールの位置・速度・インフラグ状態
- 各プレイヤーの位置・状態
- 報酬グラフ（4種類）：
  - Player A 累積報酬（エピソード単位）
  - Player A 5エピソード移動平均
  - Player B 累積報酬（エピソード単位）
  - Player B 5エピソード移動平均

## Gymnasium環境

強化学習ライブラリとの連携用：

```python
from env import TennisEnv, SinglePlayerTennisEnv
import numpy as np

# 2プレイヤー環境
env = TennisEnv(render_mode="human")
obs, info = env.reset()

# アクション: (移動方向 0-16, 打つ角度 0-360)
# ※打つ角度は自動的に正面±45度に滑らかにマッピングされます
action = (0, np.array([0.0]))
obs, reward, terminated, truncated, info = env.step(action)

# シングルプレイヤー環境（相手は自動AI）
env = SinglePlayerTennisEnv(opponent_policy="chase")
```

## ゲームルール

### フィールド
- 壁で囲まれた長方形のフィールド
- 中央に2つの「インエリア」（3:2比率、間に隙間）

### インフラグシステム
- ボールがインエリアを通過 → インフラグON
- プレイヤーが打ち返す → インフラグOFFにリセット

### ポイント判定
ボールが壁に到達した時：
- インフラグON → 打った側の**得点**
- インフラグOFF → 打った側の**失点**（アウト）

### 打ち返し条件
- ボールがリーチ距離内にある
- インフラグがONである

両方を満たす場合のみ打ち返し可能。

### 打球方向の制御
AI学習の安定化のため、エージェントが出力する打球角度（0-360度）は、プレイヤーの正面方向から **±45度** の有効範囲に滑らかに写像されます：

- `入力 0度` → `+45度`（上方向）
- `入力 180度` → `0度`（正面）
- `入力 360度` → `-45度`（下方向）

計算式：`offset = 45.0 * cos(radians(input_angle / 2))`
この写像により、境界値での勾配消失を防ぎ、学習効率を向上させています。

## 設定パラメータ

`config.py` で調整可能：

```python
Config(
    # フィールド
    field_width=800,
    field_height=400,

    # インエリア
    area_width=150,
    area_height=100,
    area_gap=100,

    # ボール
    ball_speed=5.0,
    serve_angle_range=15.0,

    # プレイヤー
    player_speed=3.0,
    reach_distance=30.0,

    # 報酬
    reward_point_win=1.0,
    reward_point_lose=-1.0,
    reward_rally=0.1,
)
```

## ファイル構成

```
├── main.py          # エントリーポイント（CLI）
├── config.py        # 設定パラメータ
├── field.py         # フィールドとインエリア
├── ball.py          # ボールの挙動とインフラグ
├── player.py        # プレイヤーの移動と打ち返し
├── game.py          # ゲームロジック
├── renderer.py      # Pygame描画（デバッグオーバーレイ含む）
├── env.py           # Gymnasium環境
├── debug.py         # デバッグログ・バリデーション
├── agents/          # エージェントシステム
│   ├── __init__.py
│   ├── base.py      # 基底クラス（save/load）
│   ├── chase.py     # ChaseAgent, SmartChaseAgent
│   ├── random_agent.py  # RandomAgent
│   └── neural.py    # NeuralAgent（Policy Gradient）
└── tests/           # ユニットテスト（96テスト）
```

## テスト

```bash
python -m unittest discover tests/ -v
```

## ライセンス

MIT
