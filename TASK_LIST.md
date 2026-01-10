# 次にやることリスト (Task List)

2DTennisSimulator プロジェクトの改善提案リストです。
各タスクには難易度、期待効果、具体的な実装手順を記載しています。

---

## 目次

- [高優先度](#高優先度-high-priority)
- [中優先度](#中優先度-medium-priority)
- [低優先度](#低優先度-low-priority)
- [完了済み](#完了済み-completed)
- [推奨作業順序](#推奨作業順序)
- [クイックリファレンス](#クイックリファレンス)

---

## 高優先度 (High Priority)

### 1. 型チェック・リンターの導入

| 項目 | 内容 |
|------|------|
| 難易度 | ★★☆☆☆ (簡単) |
| 期待効果 | バグの早期発見、コード品質向上、レビュー効率化 |
| 関連ファイル | `.github/workflows/ci.yml`, `pyproject.toml` (新規) |

**実装手順:**

- [ ] `pyproject.toml` を作成し、ツール設定を統一管理
- [ ] **ruff** (高速リンター) の導入
  ```bash
  pip install ruff
  ruff check .
  ```
- [ ] **black** (フォーマッター) の導入
  ```bash
  pip install black
  black --check .
  ```
- [ ] **mypy** (型チェック) の導入
  ```bash
  pip install mypy
  mypy --strict .
  ```
- [ ] CI/CD に統合 (`.github/workflows/ci.yml` へ追加)
  ```yaml
  - name: Lint and type check
    run: |
      ruff check .
      black --check .
      mypy .
  ```

---

### 2. テストカバレッジの拡充

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★☆☆ (中程度) |
| 期待効果 | リグレッション防止、リファクタリング安全性向上 |
| 現状 | 98テスト (game/ball/player/field のみ) |
| 目標 | 150+ テスト (エージェント・環境含む) |

**実装手順:**

- [ ] **エージェントテスト** (`tests/test_agents.py` 新規)
  - [ ] 各エージェントの `.act()` が有効なアクションを返すか
  - [ ] `.learn()` でエラーが発生しないか
  - [ ] `.save()` / `.load()` の往復でデータが保持されるか
  - [ ] 対象: `ChaseAgent`, `SmartChaseAgent`, `RandomAgent`, `BaselinerAgent`, `PositionalAgent`

- [ ] **Gymnasium 環境テスト** (`tests/test_env.py` 新規)
  - [ ] `TennisEnv.reset()` が正しい observation を返すか
  - [ ] `TennisEnv.step()` の戻り値形式 (obs, reward, done, truncated, info)
  - [ ] action_space / observation_space の範囲検証
  - [ ] `SinglePlayerTennisEnv` の動作確認

- [ ] **CLI 統合テスト** (`tests/test_cli.py` 新規)
  - [ ] `--headless` モードでのエピソード実行
  - [ ] `--agent1`, `--agent2` オプションの動作確認
  - [ ] 無効な引数でのエラーハンドリング

- [ ] **pytest-cov** でカバレッジ計測を追加
  ```bash
  pip install pytest-cov
  pytest --cov=. --cov-report=html
  ```

---

### 3. ドキュメント整備

| 項目 | 内容 |
|------|------|
| 難易度 | ★★☆☆☆ (簡単) |
| 期待効果 | 新規コントリビューター参入障壁の低下、国際利用促進 |

**実装手順:**

#### 3.1 英語版 README

- [ ] `README.md` → `README_ja.md` にリネーム
- [ ] `README.md` を英語で新規作成
  - [ ] Features セクション
  - [ ] Quick Start (installation, basic usage)
  - [ ] Agent Types 一覧
  - [ ] Gymnasium Integration 例
  - [ ] Configuration 説明

#### 3.2 CONTRIBUTING.md 作成

- [ ] 開発環境セットアップ手順
  ```markdown
  ## Development Setup
  1. Clone the repository
  2. Create virtual environment
  3. Install dependencies: `pip install -r requirements-dev.txt`
  4. Run tests: `python run_tests.py`
  ```
- [ ] コーディング規約 (black, ruff 使用)
- [ ] PR プロセス説明
- [ ] Issue テンプレート

#### 3.3 docstring 追加

- [ ] `agents/chase.py` - クラス・メソッドの説明追加
- [ ] `agents/positional.py` - 戦略ロジックの説明追加
- [ ] `agents/baseliner.py` - 守備的戦略の説明追加

---

### 4. エラー処理・バリデーション強化

| 項目 | 内容 |
|------|------|
| 難易度 | ★★☆☆☆ (簡単) |
| 期待効果 | デバッグ容易性向上、予期せぬクラッシュ防止 |
| 関連ファイル | `game.py`, `agents/base.py`, `env.py` |

**実装手順:**

- [ ] **game.py:step()** — アクション範囲チェック追加
  ```python
  def step(self, action1, action2):
      action1 = np.clip(action1, -1.0, 1.0)
      action2 = np.clip(action2, -1.0, 1.0)
      # ...
  ```

- [ ] **agents/base.py** — act() の戻り値検証
  ```python
  def validate_action(self, action):
      movement, angle = action
      assert -1.0 <= movement <= 1.0
      assert -1.0 <= angle <= 1.0
      return action
  ```

- [ ] **env.py** — pygame 未インストール時の明確なエラー
  ```python
  try:
      import pygame
  except ImportError:
      raise ImportError(
          "pygame is required for visual mode. "
          "Install with: pip install pygame"
      )
  ```

---

## 中優先度 (Medium Priority)

### 5. renderer.py のリファクタリング

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★☆☆ (中程度) |
| 期待効果 | 保守性向上、機能追加の容易化 |
| 現状 | 708行の単一ファイル |
| 目標 | モジュール分割で各ファイル200行以下 |

**実装手順:**

- [ ] `renderer/` ディレクトリ作成
- [ ] 分割構成:
  ```
  renderer/
  ├── __init__.py      # 公開APIのエクスポート
  ├── base.py          # GameRenderer クラス (コア描画)
  ├── debug.py         # DebugRenderer クラス
  ├── overlays/
  │   ├── __init__.py
  │   ├── stats.py     # 統計オーバーレイ
  │   ├── trajectory.py # 軌道予測オーバーレイ
  │   └── graphs.py    # リワードグラフ
  └── utils.py         # 共通ユーティリティ
  ```
- [ ] 既存の import を更新 (`from renderer import GameRenderer`)
- [ ] テストで動作確認

---

### 6. Transformer エージェントの改善

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★☆☆ (中程度) |
| 期待効果 | コード可読性向上、メンテナンス容易化 |
| 関連ファイル | `agents/transformer.py` (578行) |

**実装手順:**

- [ ] **マジックナンバーの定数化**
  ```python
  # Before
  self.hidden_dim = 64

  # After
  DEFAULT_HIDDEN_DIM = 64
  DEFAULT_NUM_HEADS = 4
  DEFAULT_NUM_LAYERS = 2
  ```

- [ ] **コメント整理**
  - [ ] 「魔改造」などの不明瞭なコメントを具体的な説明に置換
  - [ ] 英語または明確な日本語に統一

- [ ] **変数名の改善**
  - [ ] 略語を避け、意図が明確な名前に変更

- [ ] **型ヒントの追加・修正**

---

### 7. ロギングフレームワーク導入

| 項目 | 内容 |
|------|------|
| 難易度 | ★★☆☆☆ (簡単) |
| 期待効果 | デバッグ効率向上、トレーニング監視の容易化 |
| 関連ファイル | 全体 (print文の置換) |

**実装手順:**

- [ ] `logging_config.py` 新規作成
  ```python
  import logging

  def setup_logging(level=logging.INFO, log_file=None):
      handlers = [logging.StreamHandler()]
      if log_file:
          handlers.append(logging.FileHandler(log_file))

      logging.basicConfig(
          level=level,
          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          handlers=handlers
      )
  ```

- [ ] 各モジュールで logger 使用
  ```python
  import logging
  logger = logging.getLogger(__name__)

  # Before
  print(f"Episode {ep} complete")

  # After
  logger.info(f"Episode {ep} complete")
  ```

- [ ] CLI に `--log-level` オプション追加

---

### 8. ベンチマーク・パフォーマンス計測

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★☆☆ (中程度) |
| 期待効果 | 最適化の基準値、エージェント比較の定量化 |

**実装手順:**

- [ ] `benchmarks/` ディレクトリ作成
- [ ] **エージェント対戦ベンチマーク** (`benchmarks/agent_comparison.py`)
  ```python
  # 各エージェントペアで1000エピソード実行
  # 勝率、平均ラリー長、平均報酬を計測
  ```
- [ ] **パフォーマンスベンチマーク** (`benchmarks/performance.py`)
  ```python
  # ヘッドレスモードでのエピソード/秒を計測
  # 各エージェントの act() 呼び出し時間を計測
  ```
- [ ] 結果を `benchmarks/results/` に JSON 保存
- [ ] README にベンチマーク結果を掲載

---

## 低優先度 (Low Priority)

### 9. マルチポイントゲームモード

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★★☆ (やや難) |
| 期待効果 | より現実的なテニスシミュレーション、長期戦略学習 |

**実装手順:**

- [ ] `Game` クラスにゲームモード追加
  ```python
  class GameMode(Enum):
      SINGLE_POINT = "single_point"
      TIEBREAK = "tiebreak"      # 7ポイント先取
      SET = "set"                # 6ゲーム先取
  ```
- [ ] ポイント間のリセット処理
- [ ] ゲーム/セット終了判定ロジック
- [ ] 対応する observation 拡張 (現在スコア情報)

---

### 10. エージェントトーナメントシステム

| 項目 | 内容 |
|------|------|
| 難易度 | ★★★☆☆ (中程度) |
| 期待効果 | エージェント間の客観的比較、開発モチベーション向上 |

**実装手順:**

- [ ] `tournament.py` 新規作成
- [ ] 総当たり戦 (Round Robin) 実装
- [ ] Elo レーティング計算
- [ ] 結果の可視化 (勝敗表、レーティング推移グラフ)
- [ ] CLI から実行可能に (`--tournament` オプション)

---

### 11. 設定ファイルサポート

| 項目 | 内容 |
|------|------|
| 難易度 | ★★☆☆☆ (簡単) |
| 期待効果 | 実験再現性向上、複雑な設定の管理容易化 |

**実装手順:**

- [ ] `tomllib` (Python 3.11+) または `toml` ライブラリ使用
- [ ] 設定ファイル例 (`config.toml`)
  ```toml
  [game]
  field_width = 800
  field_height = 400
  max_steps = 1000

  [agent1]
  type = "neural"
  learning_rate = 0.001

  [agent2]
  type = "chase"
  ```
- [ ] CLI に `--config` オプション追加
- [ ] CLI 引数が設定ファイルより優先

---

### 12. その他の改善

| タスク | 難易度 | 説明 |
|--------|--------|------|
| 進捗バー表示 | ★☆☆☆☆ | `tqdm` 導入でトレーニング進捗表示 |
| ヘルプテキスト充実 | ★☆☆☆☆ | `--help` の説明を詳細化 |
| CHANGELOG.md | ★☆☆☆☆ | バージョン履歴の記録開始 |
| デモ GIF 追加 | ★★☆☆☆ | README にゲームプレイ GIF を追加 |
| requirements.txt 固定 | ★☆☆☆☆ | `pip freeze` でバージョン固定 |
| 報酬シェーピング | ★★★☆☆ | カスタム報酬関数のサポート |
| ポリシー可視化 | ★★★★☆ | 学習済みエージェントの行動パターン表示 |

---

## 完了済み (Completed)

| タスク | 完了日 | 備考 |
|--------|--------|------|
| コードレビュー実施 | - | `CODE_REVIEW.md` 作成 |
| アーキテクチャドキュメント | - | `ARCHITECTURE.md` 作成 |
| CI/CD パイプライン構築 | - | GitHub Actions |
| 基本テスト作成 | - | 98テスト、100%パス |
| コードレビュー指摘修正 | - | `Config.to_dict()`, `is_in` 命名等 |

---

## 推奨作業順序

```
フェーズ 1: 基盤整備
├── 1. 型チェック・リンター導入 ──────┐
├── 3.1 英語版 README ─────────────────┼── 並行作業可能
└── 3.2 CONTRIBUTING.md ───────────────┘

フェーズ 2: 品質強化
├── 2. テストカバレッジ拡充 ───────────┐
├── 4. エラー処理強化 ─────────────────┼── 並行作業可能
└── 7. ロギング導入 ───────────────────┘

フェーズ 3: リファクタリング
├── 5. renderer.py 分割 ───────────────┐
└── 6. Transformer 改善 ───────────────┴── 順次作業推奨

フェーズ 4: 機能拡張
├── 8. ベンチマーク追加
├── 9. マルチポイントゲーム
├── 10. トーナメントシステム
└── 11. 設定ファイルサポート
```

---

## クイックリファレンス

### 難易度別タスク一覧

| 難易度 | タスク |
|--------|--------|
| ★☆☆☆☆ | 進捗バー、ヘルプテキスト、CHANGELOG、requirements固定 |
| ★★☆☆☆ | リンター導入、ドキュメント、エラー処理、ロギング、設定ファイル |
| ★★★☆☆ | テスト拡充、renderer分割、Transformer改善、ベンチマーク、トーナメント |
| ★★★★☆ | マルチポイントゲーム、ポリシー可視化 |

### すぐ始められるタスク (依存関係なし)

1. `pyproject.toml` 作成 + ruff/black 導入
2. 英語版 README 作成
3. CONTRIBUTING.md 作成
4. `tqdm` 導入で進捗バー追加
5. CHANGELOG.md 作成開始

### 依存関係のあるタスク

- テスト拡充 → pytest-cov 導入 (テスト基盤必要)
- renderer分割 → 既存テストで動作確認必要
- ベンチマーク → エージェントテスト完了後推奨

---

*最終更新: 2026-01-10*
