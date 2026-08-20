# Jailbreak防御のための「理想的データセット」設計と統一学習ガイドライン

`docs/03_defense_survey.md` で指摘した最大のギャップ ——
**学習アルゴリズム(DPO等)の研究は豊富だが、学習データの統一的作成ガイドラインが乏しい**
—— を埋めるための具体案。目的は「どの組織・どのモデルでも同じ基準でデータを作れば
比較可能な安全性が得られる」状態を作ることである。

## 1. データスキーマ(画像・テキストの組)

```json
{
  "sample_id": "uuid",
  "modality_inputs": {
    "image": {"path": "...", "provenance": "synthetic|licensed|user_report|redteam"},
    "text_prompt": "ユーザーが送信したテキスト(無害化されている場合が多い)"
  },
  "derived_signals": {
    "visual_instruction_text": "画像から内省抽出された指示の言語化(OCR+VLM記述)",
    "cross_modal_consistency": "consistent | inconsistent | ambiguous"
  },
  "policy": {
    "category": "SAFETY_POLICIES のいずれか(15分類を推奨最小セットとする)",
    "severity": 1,
    "subcategory": "任意"
  },
  "responses": {
    "chosen": {
      "text": "安全な応答本文",
      "style": "hard_refusal | soft_refusal_with_explanation | safe_alternative | partial_compliance",
      "author": "human_reviewer_id | synthetic_generator_version"
    },
    "rejected": [
      {"text": "...", "failure_mode": "harmful_compliance | unhelpful_over_refusal | inconsistent_reasoning"}
    ]
  },
  "difficulty": {
    "obfuscation_level": "raw|outline|filled|fragmented|vector|noise_edge",
    "stealth_score": 0.0,
    "multi_turn_position": 1
  },
  "review": {
    "reviewer_ids": ["..."],
    "agreement_score": 0.0,
    "guideline_version": "v1.2"
  }
}
```

**設計上のポイント**

- `responses.rejected` を **単一の「悪い応答」ではなく failure_mode 別に複数用意**する
  (有害コンプライアンス型 / 過剰拒否型 / 論理不整合型)。これによりDPO学習が
  「とにかく拒否すればよい」という縮退解に陥るのを防ぐ(docs/03 2.4節の改善案に対応)。
- `derived_signals` を明示フィールド化することで、`introspective_defense.py` のような
  内省ベース防御の学習・評価にそのまま使える。
- `difficulty.obfuscation_level` は本リポジトリの `shape_obfuscation.py` の6段階に対応させ、
  データセット構築時に難易度分布を制御できるようにする(`dataset_optimizer.py` と直結)。

## 2. 統一学習ガイドライン(必須ドキュメントテンプレート)

各組織がDPO等の選好データを作成する際に**必ず**満たすべき最小ガイドライン。
これ自体をリポジトリの `docs/templates/guideline_template.md` としてテンプレート化し、
新規プロジェクト開始時にコピーして使うことを推奨する。

### 2.1 ポリシー分類の統一
- 15カテゴリ(`src/dataset/iesbench_schema.py::SAFETY_POLICIES`)を最小共通集合とし、
  独自カテゴリを追加する場合は必ず既存分類へのマッピング表を残す(比較可能性の担保)。
- 各カテゴリに **severity 1-5 の定義文**を用意する(例: severity 5 =
  「児童保護」「実行可能な大量被害」等、実世界での取り返しのつかなさで定義)。

### 2.2 chosen応答のスタイルガイド
| スタイル | 使用条件 | 例の要件 |
|---|---|---|
| hard_refusal | severity 5、代替案の提示自体もリスクを高める場合 | 理由は簡潔、代替案なし |
| soft_refusal_with_explanation | severity 3-4 | ポリシー該当理由を1-2文で明示 |
| safe_alternative | 意図の大部分が正当な目的と両立可能 | 安全な代替タスクを具体的に提示 |
| partial_compliance | 一部要素のみ問題(例: 画像の一部属性のみ) | 問題部分を除いた編集を実行 |

**過剰拒否対策**: 各カテゴリについて「正当な文脈での類似リクエスト」を
最低20%の割合で混ぜ、hard_refusalが安易に選ばれないようにする。

### 2.3 rejected応答生成の基準
1. **harmful_compliance**: 実際に弱アライメントモデルが生成しうる応答を採取
   (`src/eval/model_adapter.py::DummyWeaklyAlignedAdapter` のような合成モデル、
   または実際のレッドチーム結果)。
2. **unhelpful_over_refusal**: 「一切の説明なく拒否」「明らかに無害な要求まで拒否」を
   意図的に含める(過剰拒否を罰する学習信号)。
3. **inconsistent_reasoning**: 内省ステップで矛盾した理由付けをした例
   (例: 画像の指示を見落としたまま許可してしまうケース)。

### 2.4 アノテーション品質基準
- 各サンプルは最低2名のレビュアーが独立評価し、`agreement_score`(Cohen's kappa等)を記録。
- 不一致サンプルは破棄せず「ambiguous」カテゴリとして別集計し、
  境界事例の分析に使う(閾値設計のキャリブレーションデータとして重要)。
- `guideline_version` を必ず記録し、ガイドライン改訂時の再アノテーション範囲を追跡可能にする。

### 2.5 難易度・多様性の被覆基準
- `src/dataset/dataset_optimizer.py` の貪欲最適化アルゴリズムを用い、
  最低限「(category × action) の全組み合わせ」と「難読化レベル6種 × 主要言語」を
  被覆するまでサンプルを追加する(本リポジトリの実行例では150サンプルで
  135通りのcategory×actionペアと116属性を完全被覆できることを確認済み)。
- 難易度(stealth_score等)のヒストグラムが一様に近くなるよう
  `w_difficulty_balance` パラメータで調整する。

### 2.6 マルチターン・セッションデータの作成基準
- 単発データに加え、**意図分割(salami slicing)攻撃を模したセッション系列**を
  一定割合(推奨10-15%)含める。各ターンは単体では違反しないが、
  系列全体では違反に至る設計にする(`SessionRiskTracker` の学習・評価用)。

### 2.7 多言語・多文字種カバレッジ
- 最低でも「高リソース言語(英語)」「対象市場の主要言語(例: 日本語)」
  「多言語混在」の3条件×フォント3種×難読化6段階を含める
  (`variant_generator.py` のグリッド生成はこれを機械的に満たす設計)。

### 2.8 データガバナンス・倫理チェックリスト
- [ ] 実在人物の画像を含まない、または適切な同意/権利処理がある
- [ ] 児童を含む不適切コンテンツを生成・保持しない(生成的にも合成的にも一切禁止)
- [ ] rejected応答の「harmful_compliance」例はアクセス制御された環境でのみ保管
- [ ] 学習データセット自体の再配布ポリシーを明記(本リポジトリはプレースホルダーのみ)
- [ ] 定期的な監査(誰が・いつ・どのバージョンを承認したか)のログを残す

## 3. 本リポジトリでの実装対応表

| ガイドライン項目 | 対応コード |
|---|---|
| ポリシー分類統一 | `src/dataset/iesbench_schema.py::SAFETY_POLICIES` |
| 難易度・被覆最適化 | `src/dataset/dataset_optimizer.py` |
| 難読化レベル6段階の生成 | `src/attack/shape_obfuscation.py` |
| フォント/色/言語/サイズのグリッド生成 | `src/attack/variant_generator.py` |
| 検知回避×可読性のパレート分析 | `src/attack/compare_optimize.py` |
| failure_mode別rejected生成の土台 | `src/defense/train_safety_dpo.py::build_synthetic_preference_data` |
| セッション単位のリスク追跡 | `src/defense/unified_defense_pipeline.py::SessionRiskTracker` |
| 過剰拒否も含めた評価 | `src/eval/metrics.py`(False Block Rateは今後拡張点として明記) |

## 4. 今後の拡張ポイント

1. `metrics.py` に **False Block Rate**(正当リクエストの誤ブロック率)を追加し、
   安全性と有用性のトレードオフを常に併記する。
2. `judge.py` の `SyntheticJudge` を実際のLLM-as-judge(`LLMJudge`)に置き換え、
   人手評価とのCohen's kappaを定期計測してjudgeの信頼性を監査する。
3. ガイドラインテンプレートをバージョン管理し、改訂差分がデータセットの
   どのスライスに影響するかをトレースできるようにする(`guideline_version`フィールド活用)。
