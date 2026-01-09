# 2DTennisSimulator コードレビュー

率直に気になった点をまとめた。良いところも悪いところも遠慮なく書く。

**注**: エージェント（agents/配下）は「obsを受け取ってactionを返すブラックボックス」という設計思想のため、各実装者の自由領域としてレビュー対象外とした。

---

## 全体的な印象

教育・実験目的としては面白いプロジェクト。ただ、「とりあえず動く」から「ちゃんとしたコード」への過渡期にある感じ。

---

## 気になる点

### 1. Config.to_dict() が手動列挙

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

### 2. player.py の can_hit() のロジック

```python
# player.py:41-42
def can_hit(self, ball: Ball) -> bool:
    return ball.in_flag and ball.distance_to(self.x, self.y) <= self.reach_distance
```

`ball.in_flag` が True のときだけ打てる仕様、直感に反する。ボールがインエリアを通過した後でないと打てないのは分かるけど、変数名が `in_flag` だと「インエリア内にいる」と誤解しやすい。`passed_through_in_area` とかの方が意図が伝わる。

### 3. game.py の step() が複雑

```python
# game.py:74-124
def step(self, action_a, action_b) -> StepResult:
```

50行のメソッドに条件分岐が多重でネスト。責務を分離して `_handle_ball_collision()` とか切り出した方がテストしやすい。

### 4. グローバル変数 (debug.py)

```python
# debug.py:311-325
_global_logger: Optional[DebugLogger] = None

def get_logger() -> DebugLogger:
    global _global_logger
```

シングルトンパターンの悪い例。テストで困る。依存性注入にした方がいい。

### 5. env.py の observation_space 定義

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

### `ball.in_flag` の命名

何度も書くけど、この名前は紛らわしい。「ボールがまだインエリアを通過していない」状態なのか「既に通過した」状態なのか、名前から分からない。

---

## 良いところ

フェアにいくと、良い点もある。

1. **テストが充実** - 96個のテストがあるのは立派。`run_tests.py` で簡単に実行できる。

2. **モジュール分離** - game, ball, player, field が分かれてるのは良い設計。

3. **型ヒント** - ほぼ全部に型ヒントがある。読みやすい。

4. **Agentの抽象化** - 基底クラスから派生させる設計は拡張しやすい。

5. **設定の外出し** - `Config` dataclass での管理は正しいアプローチ。

---

## 優先度高めの修正提案

| 優先度 | 項目 | 理由 |
|--------|------|------|
| 高 | `in_flag` のリネーム | 可読性向上、誤解防止 |
| 中 | `Config.to_dict()` を `asdict()` に | メンテ漏れ防止 |
| 中 | `game.step()` の分割 | テスタビリティ向上 |
| 低 | グローバルロガー廃止 | テスト容易性 |

---

## 総評

「動くプロトタイプ」としては十分。コアエンジン（game, ball, player, field）の設計は堅実。エージェントが独立したブラックボックスとして競い合う構造も面白い。

長期的にメンテするなら `in_flag` の命名だけは早めに直した方がいい。後から変えるとテストも全部書き直しになる。
