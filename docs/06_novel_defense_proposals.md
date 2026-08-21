# 新規研究提案: Curriculum DPO / 脅威語彙データセット / Attack Immune Memory

ユーザー要望「DPOを最適化するための学習の順番の研究」「LLMへの脅威度を3段階に分類した
単語ごとのデータセットの最適化」「世にない新たな防御手法」を受け、3つの実装可能な
新規研究テーマを提案・実装した。いずれも `docs/03_defense_survey.md` で指摘した
「学習アルゴリズムの研究は多いが、データ作成・提示順序・運用ループの研究が薄い」という
ギャップを埋める方向性である。

---

## 1. Curriculum DPO — 選好ペアの提示順序の最適化研究

### 1.1 動機

DPO自体(損失関数・正則化)の研究は多いが、**選好ペアをどの順序でモデルに提示するか**
というカリキュラム設計はほとんど検討されていない。関連する知見:

- 教師あり学習における easy-to-hard カリキュラム学習(Bengio et al., 2009)
- 自己ペース学習 self-paced learning(Kumar et al., 2010): 現在のモデルが「解ける」
  サンプルから提示する
- 継続学習(continual learning)における破局的忘却(catastrophic forgetting):
  後から学んだタスクが先に学んだタスクの性能を劣化させる

DPOによる安全アライメントは、実質的に「複数の安全カテゴリを順に(あるいは混ぜて)
学習する継続学習タスク」とみなせる。カテゴリを1つずつブロックで学習すると、
最後に学んだカテゴリの安全性は強いが、最初に学んだカテゴリの安全性が薄れているかもしれない
——これは実運用上重要な問題であり(「最近パッチされた脆弱性は塞がったが、昔のパッチが
リグレッションしていないか」という懸念と同型)、まだ体系的に検証されていない。

### 1.2 実装 (`src/defense/curriculum_dpo.py`)

7つの提示順序戦略を実装し、同一の初期重み・同一データで比較できるようにした:

| 戦略 | 説明 |
|---|---|
| `random` | ベースライン(シャッフルのみ) |
| `easy_to_hard` | 語彙脅威スコア(後述の脅威語彙データセット)が高い「わかりやすい」ペアから開始 |
| `hard_to_easy` | 逆順(anti-curriculum、比較対象) |
| `spaced_repetition` | 易→難で1周後、難しい後半1/3をエポック中に再注入(間隔反復) |
| `category_blocked` | 安全カテゴリごとにブロックして順に学習(継続学習の典型セットアップ) |
| `category_interleaved` | カテゴリをラウンドロビンで交互に学習 |
| `self_paced` | 各ステップ前に現在のモデルでの chosen/rejected 分離マージンを測り、易しい順に並べ替え |

**難易度スコアの定義**: 選好ペアの `rejected` 応答テキストを脅威語彙データセット
(下記2節)でスコアリングし、`weighted_score` が高い(強いtier3語を含む)ペアほど
「モデルにとって分離しやすい=易」とみなす。VJAのような難読化・曖昧な表現による
攻撃は、語彙シグナルが弱くなる=このスコアで「難」に分類されるため、
攻撃の巧妙さ(曖昧さ)とカリキュラム上の難易度が自然に対応する設計になっている。

**測定指標**:
- `steps_to_threshold`: 損失が閾値を下回るまでのステップ数(収束速度)
- `final_loss` / `mean_last_10pct_loss`: 最終的な学習の質
- `forgetting_score`(`category_blocked`戦略のみ): 最初に学習したカテゴリブロックについて
  「学習直後の損失」と「全カテゴリ学習後の損失」の差。正であれば破局的忘却が起きている。

### 1.3 実行方法

```bash
python -m src.defense.curriculum_dpo --data data/sample/dpo_preferences.jsonl \
    --strategies random,easy_to_hard,hard_to_easy,spaced_repetition,category_blocked,category_interleaved,self_paced \
    --epochs 3 --out outputs/curriculum_dpo_report.json
```

### 1.4 現状の制約と、実スケールでの検証手順

本リポジトリの mock モデル(`TinyCharTransformer`、数万パラメータ)は
60件程度のトイデータを数ステップで完全に記憶できてしまうため、
戦略間の差はほぼ観測されない(`steps_to_threshold` が全戦略で5-6ステップに収束する)。
これは**配線の正しさの検証**が目的であり、研究として意味のある比較には
以下のスケールアップが必要:

1. 実際のVLM(数億〜数十億パラメータ)+ LoRA で `train_safety_dpo.py --model-name <実モデル>` を使用
2. `IESBench` 実データ相当の数千件規模の選好ペア(`docs/04_ideal_dataset_design.md` の
   統一ガイドラインに沿って作成)
3. `forgetting_score` を全カテゴリについて測定し、`category_blocked` と
   `category_interleaved` の間でどの程度差が出るかを比較
4. 学習後のモデルを `src/eval/run_eval.py` でASR/HS/EV/HRR/FBR評価し、
   「収束が速い戦略」が「最終的な安全性・過剰拒否率でも優れているか」を検証
   (収束速度と汎化性能は必ずしも一致しないため、ここが本研究の核心的な問いになる)

### 1.5 仮説と期待される知見

- `category_interleaved` は `category_blocked` より forgetting_score が低い(継続学習の
  知見と整合するはず)
- `spaced_repetition` は `easy_to_hard` よりVJAのような「難」サンプル(曖昧な表現)への
  最終的な頑健性が高い(間隔反復の効果)
- `self_paced` は初期の収束は速いが、モデルが「解きやすい」サンプルに偏った学習をし、
  hard negativeへの汎化が `easy_to_hard` より劣る可能性がある(自己ペース学習の
  既知の弱点である「自己強化バイアス」)

これらは仮説であり、実スケールでの追試験が必要な**未検証の研究提案**である点を明記する。

---

## 2. 3段階脅威度の単語レベル語彙データセット + 最適化アルゴリズム

### 2.1 動機

`introspective_defense.py` のルールベース判定は元々フラットなキーワード一致
(該当すれば即拒否)だった。しかし現実には:

- 同じ単語("gun"等)でも文脈(博物館の展示 vs 武器の製造依頼)で危険度が全く異なる
- 単語単体では弱いシグナルでも、複数の単語が組み合わさると強いシグナルになる
- 「強い/確実な語」と「弱い/文脈依存の語」を区別しないと、過剰拒否(FBR上昇)を招く

そこで、**3段階(tier)の脅威度**を持つ語彙データセットを構築し、
「良性文脈での出現頻度」から自動的に曖昧度を較正するアルゴリズムを実装した。

### 2.2 実装 (`src/dataset/lexicon_optimizer.py`)

- **Seed lexicon**: 15安全ポリシー × 2言語(en/ja) × 3tier の種となる語彙(カテゴリを
  指し示す一般名詞レベルの語のみ。手口・製法などの具体的instructionsは含まない)
- **曖昧度較正**: 良性コンテキストコーパスでの出現頻度から `ambiguity_score` を算出し、
  一定閾値を超えたら自動的にtierを1段階下げる(例: "gun"はtier3→tier2、
  "drug"はtier2→tier1に降格。実測: 8/154語が降格)
- **被覆最適化**: 15カテゴリ×2言語×3tierの全セルが埋まっているか検証し、
  空セルがあれば `needs_human_review=True` のスタブとして出力する
  (**自動生成した語を無審査でブロック判定に使わせない安全設計**)
- **スコアリング**: `ThreatLexicon.score_text(text)` が tier別重み(1/3/9)で
  加重スコアと最大tier、カテゴリ別ヒットを返す

### 2.3 実行方法

```bash
python -m src.dataset.lexicon_optimizer --out data/sample/threat_lexicon.json --ambiguity-threshold 0.3
```

### 2.4 既存コンポーネントへの統合

- `introspective_defense.make_lexicon_reasoning_fn(lexicon)`: tier3→即時デナイ、
  tier2→デナイ(要人手レビュー注記付き)、tier1のみ→許可(過剰拒否回避)という
  3段階の判定関数を `IntrospectiveDefense(reasoning_fn=...)` に差し込める
- `curriculum_dpo.py`: 選好ペアの難易度スコアの入力信号として使用(1.2節)
- `immune_memory_defense.py`: 将来的に記憶照合の特徴量に追加可能(現状はguard classifierの
  埋め込みのみ使用、拡張ポイントとして doc に明記)

---

## 3. Attack Immune Memory Defense — 免疫記憶型防御(新規提案)

### 3.1 アイデア

生体の獲得免疫系は、一度遭遇した病原体を「記憶細胞」として保持し、再侵入時により
速く反応する。記憶は時間とともに減衰し、繰り返し曝露されるとブースター効果で
増強される。この比喩をJailbreak防御に応用した、**世にまだ無いと思われる新規の
防御アーキテクチャ**を提案・実装した。

### 3.2 メカニズム (`src/defense/immune_memory_defense.py`)

1. `adaptive_attack_optimizer.py`(closed-loop red teaming, 既存実装)が発見した
   「guard classifierの検知をすり抜けた」攻撃構成を、`GuardClassifier` の
   エンコーダ部分(分類ヘッド手前)を流用した埋め込みベクトルとして
   `AttackMemoryBank` に登録する。
2. 新規リクエストのたびに埋め込みを記憶細胞群とコサイン類似度で照合する
   (`guard_classifier`本体の推論より軽量な高速事前フィルタ = Layer 0)。
3. 類似度は指数関数的に時間減衰する(半減期パラメータ)。これにより:
   - 古い記憶が恒久的な誤検知源になることを防ぐ(概念ドリフトへの適応)
   - 一方で直近の攻撃キャンペーンには強く反応する
4. 同一カテゴリの記憶への「再曝露」(ヒット)が時間窓内で閾値を超えたら
   `booster_retrain_needed` フラグを立て、guard classifierの再学習を促す。
   これにより「単発の誤検知」と「継続的な攻撃キャンペーン」を区別できる。

### 3.3 closed-loopとしての位置づけ

```
adaptive_attack_optimizer.py (攻撃側の適応的最適化)
        │ 検知回避に成功した構成
        ▼
AttackMemoryBank.add_from_optimizer_history()  (記憶細胞として登録)
        │
        ▼
UnifiedDefensePipeline (Layer 0: 高速記憶照合)
        │ ヒット率が閾値超 → booster_retrain_needed
        ▼
train_guard_classifier.py の再学習トリガー(運用上は人間 or 自動ジョブが実行)
        │
        └──→ 再学習後、adaptive_attack_optimizer.py で再度探索 … (ループ)
```

これは `docs/02_attack_enhancement_proposals.md` で述べた「レッドチーム演習としての
活用方法」(月次ループ)を、**イベント駆動・自動トリガー型**に発展させたものである。

### 3.4 実行方法

```bash
python -m src.defense.immune_memory_defense --guard-ckpt outputs/guard_classifier.pt --n-steps 30
```

`unified_defense_pipeline.py` に `memory_bank=AttackMemoryBank(...)` を渡すことで
Layer 0として組み込まれる。

### 3.5 今後の検証課題

- 減衰半減期・類似度閾値・ブースター閾値のハイパーパラメータ探索
  (実データでのROC分析が必要、現状はデフォルト値のみ)
- 埋め込みをguard classifierのタスク特化特徴量ではなく、CLIP等の汎用画像埋め込みに
  差し替えた場合の汎化性能比較
- 複数の防御拠点(マルチテナント)で記憶バンクを共有した場合の、攻撃伝播の
  早期検知効果(「ある顧客で発見された攻撃パターンが他の顧客への攻撃も防ぐ」効果の検証)

---

## 4. まとめ

| 提案 | 新規性の位置づけ | 実装状態 |
|---|---|---|
| Curriculum DPO | 既存分野(カリキュラム学習・継続学習)の知見をDPO安全アライメントに**応用する研究提案**。ガイドライン不在という既知ギャップへの直接的な回答 | 7戦略の比較実験コード実装済み、トイスケールで動作確認済み。実スケール検証は今後の課題 |
| 3段階脅威語彙データセット | 既存の「フラットなキーワードリスト」を tier + 曖昧度較正で高度化する**具体的アルゴリズム提案** | 被覆最適化・曖昧度較正アルゴリズム実装済み、既存防御モジュールへ統合済み |
| Attack Immune Memory | 免疫学の比喩に基づく**新規の防御アーキテクチャ提案**(既存文献に同型のものは確認できていない) | 記憶・減衰・ブースタートリガーの全機構を実装し、closed-loopでの動作確認済み |
