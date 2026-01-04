# 2D Tennis Simulator

AIエージェント同士が対戦する2Dテニスシミュレーター。
「ホームポジションに戻る」などの基本戦術が学習によって自然発生するか観察することを目的としています。

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### ビジュアルモード（デモ）
```bash
python main.py --mode visual
```

### キーボード操作モード
```bash
python main.py --mode keyboard
```
- 矢印キー / WASD: プレイヤーAを移動
- ESC: 終了

### ヘッドレスモード
```bash
python main.py --mode headless
```

### オプション
- `--points N`: 勝利に必要なポイント数（デフォルト: 11）
- `--speed F`: ボール速度（デフォルト: 5.0）
- `--fps N`: フレームレート（デフォルト: 60）

## AI学習用環境

Gymnasium互換の環境を提供しています：

```python
from env import TennisEnv, SinglePlayerTennisEnv
from config import Config

# 2プレイヤー環境
env = TennisEnv(render_mode="human")
obs, info = env.reset()

# シングルプレイヤー環境（相手はシンプルなAI）
env = SinglePlayerTennisEnv(opponent_policy="chase")
obs, info = env.reset()

# ステップ実行
action = (0, np.array([45.0]))  # (移動方向, 打つ角度)
obs, reward, terminated, truncated, info = env.step(action)
```

## ゲームルール

### フィールド
- 壁で囲まれた長方形のフィールド
- 2つの「インエリア」（エリアA、エリアB）が配置されている

### インフラグ
- ボールが「インエリア」を通過するとインフラグがONになる
- プレイヤーが打ち返すとインフラグはOFFにリセットされる

### ポイント判定
- ボールが壁に到達した時：
  - インフラグON → 打った側の得点
  - インフラグOFF → 打った側の失点（アウト）

### 打ち返し条件
- ボールがリーチ距離内にある
- インフラグがONである

両方を満たす場合のみ打ち返しが可能。

## 設定パラメータ

`config.py` で設定可能：

```python
Config(
    # フィールド
    field_width=800,
    field_height=400,

    # インエリア（3:2比率）
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

- `config.py`: 設定パラメータ
- `field.py`: フィールドとインエリアの定義
- `ball.py`: ボールの挙動とインフラグ
- `player.py`: プレイヤーの移動と打ち返し
- `game.py`: ゲームロジック
- `renderer.py`: Pygame描画
- `env.py`: Gymnasium環境
- `main.py`: エントリーポイント
