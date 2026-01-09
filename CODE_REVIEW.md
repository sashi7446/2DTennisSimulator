# 2DTennisSimulator コードレビュー

率直に気になった点をまとめた。良いところも悪いところも遠慮なく書く。

---

## 全体的な印象

教育・実験目的としては面白いプロジェクト。ただ、「とりあえず動く」から「ちゃんとしたコード」への過渡期にある感じ。

---

## 気になる点

### 1. マジックナンバーの嵐 (transformer.py, neural.py)

```python
# transformer.py:234
dx = (ball_x - my_x) / 800.0
dy = (ball_y - my_y) / 400.0
```

```python
# transformer.py:249
target_x = 37.5 if self.player_id == 0 else 762.5
```

`config.py` でちゃんと定義してるのに、各所で `800`, `400`, `37.5`, `762.5` がハードコードされてる。設定変えたら全部壊れる。

### 2. TransformerAgent の「独り言」機能

```python
# transformer.py:347-351
if self.step_counter % 500 == 0:
    thoughts = ["ボールの軌道が見える...", "もっと速く...！", ...]
    print(f"  🤖 (独り言): {np.random.choice(thoughts)}")
```

正直これはコードに残すべきじゃない。デバッグで遊んでた名残だと思うけど、本番コードに絵文字と独り言があるとレビューで突っ込まれる。ログレベルで制御するか、削除した方がいい。

### 3. 日本語と英語のコメント混在

```python
# transformer.py
"""Transformer-based Neural Network Agent

より強力なアーキテクチャ:
- Multi-head Self-Attention: 複数の観測要素間の関係性を学習
"""
```

個人プロジェクトなら好みの問題だけど、統一した方が読みやすい。

### 4. 重複コード (ヘルパー関数)

`_get_my_pos()` と `_angle_to_direction()` が `chase.py` と `positional.py` で全く同じ実装。共通モジュールに出すべき。

```python
# chase.py:9-12
def _get_my_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    prefix = "player_a" if player_id == 0 else "player_b"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]

# positional.py:10-13  ← 完全に同じ
def _get_my_pos(obs: Dict[str, Any], player_id: int) -> Tuple[float, float]:
    prefix = "player_a" if player_id == 0 else "player_b"
    return obs[f"{prefix}_x"], obs[f"{prefix}_y"]
```

### 5. Config.to_dict() が手動列挙

```python
# config.py:43-62
def to_dict(self) -> Dict[str, Any]:
    return {
        "field_width": self.field_width,
        "field_height": self.field_height,
        # ... 全部手で書いてる
    }
```

`dataclasses.asdict(self)` で一発。フィールド追加するたびに `to_dict()` の更新を忘れるパターン。

### 6. player.py の can_hit() のロジック

```python
# player.py:41-42
def can_hit(self, ball: Ball) -> bool:
    return ball.in_flag and ball.distance_to(self.x, self.y) <= self.reach_distance
```

`ball.in_flag` が True のときだけ打てる仕様、直感に反する。ボールがインエリアを通過した後でないと打てないのは分かるけど、変数名が `in_flag` だと「インエリア内にいる」と誤解しやすい。`passed_through_in_area` とかの方が意図が伝わる。

### 7. game.py の step() が複雑

```python
# game.py:74-124
def step(self, action_a, action_b) -> StepResult:
```

50行のメソッドに条件分岐が多重でネスト。責務を分離して `_handle_ball_collision()` とか切り出した方がテストしやすい。

### 8. TransformerAgent.learn() が長すぎ (65行)

```python
# transformer.py:364-429
def learn(self, reward: float, done: bool) -> None:
```

日本語のprint文、エリートプール管理、変異処理、状態リセットが全部ここに詰め込まれてる。責務多すぎ。

### 9. グローバル変数 (debug.py)

```python
# debug.py:311-325
_global_logger: Optional[DebugLogger] = None

def get_logger() -> DebugLogger:
    global _global_logger
```

シングルトンパターンの悪い例。テストで困る。依存性注入にした方がいい。

### 10. env.py の observation_space 定義

```python
# env.py:30-42
self.observation_space = spaces.Dict({
    "ball_x": spaces.Box(0, 1, (1,), np.float32),
    # ...
    "score_a": spaces.Box(0, 100, (1,), np.int32),
})
```

スコアの上限が100に固定されてるけど、根拠は？ゲーム設計上、スコアに上限はないはず。

---

## 設計上の懸念

### 進化的アルゴリズムと勾配法の混在

TransformerAgentは進化的アルゴリズム（エリートプール、変異）を使ってるのに、NeuralAgentは純粋なPolicy Gradient。どっちかに統一するか、明確に分けた方がいい。今は中途半端に見える。

### `ball.in_flag` の命名

何度も書くけど、この名前は紛らわしい。「ボールがまだインエリアを通過していない」状態なのか「既に通過した」状態なのか、名前から分からない。

---

## 良いところ

フェアにいくと、良い点もある。

1. **テストが充実** - 96個のテストがあるのは立派。`run_tests.py` で簡単に実行できる。

2. **モジュール分離** - game, ball, player, field が分かれてるのは良い設計。

3. **型ヒント** - ほぼ全部に型ヒントがある。読みやすい。

4. **Agentの抽象化** - 基底クラスから派生させる設計は拡張しやすい。

5. **設定の外出し** - `Config` dataclass での管理は正しいアプローチ。（ただし、使い切れてない）

---

## 優先度高めの修正提案

| 優先度 | 項目 | 理由 |
|--------|------|------|
| 高 | マジックナンバー排除 | 設定変更時に壊れる |
| 高 | 独り言print削除 | 本番コードとして不適切 |
| 中 | ヘルパー関数の共通化 | DRY原則違反 |
| 中 | `in_flag` のリネーム | 可読性向上 |
| 低 | コメント言語統一 | 好みの問題 |

---

## 総評

「動くプロトタイプ」としては十分。ただ、長期的にメンテするなら上記の点は直した方がいい。特にマジックナンバーは早めに潰さないと後で地獄を見る。

TransformerAgentの「魔改造 V3.7 Interceptor-Prime v4」みたいなコメント、実験してる感じは伝わるけど、そのノリのままプロダクションに持っていくとコードレビューで詰められるので気をつけて。
