# VJA 概要と追実験の全体像

## 1. 論文サマリ

**タイトル:** When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models
**著者/場:** Hou et al., ICML 2026 (Oral, 元々は Spotlight)
**arXiv:** 2602.10179 / **公式実装:** GitHub `CSU-JPG/VJA` / **データセット:** `CSU-JPG/IESBench` (Hugging Face)

### 1.1 問題設定

近年の画像編集モデル(Nano Banana Pro, GPT-Image-1.5, Qwen-Image-Edit, Seedream 4.5 等)は、
テキスト指示だけでなく「画像そのものに描かれた指示」(矢印・マーク・画像内テキストなど)から
編集意図を読み取れるようになった。これは UX 上のブレークスルーである一方、
**安全性ガードレールの多くはテキストプロンプトのみを検査しており、画像内に埋め込まれた指示を検査していない**
という **モダリティ不一致 (modality mismatch)** を生む。VJA はこの隙を突く。

### 1.2 VJA (Vision-Centric Jailbreak Attack) の核心アイデア

- 従来のタイポグラフィ攻撃(例: FigStep, MM-SafetyBench 系)は「有害なテキスト指示を画像化してVLMに読ませる」
  = **テキスト→画像** の変換が中心。
- VJA は **"visual-to-visual"**、すなわち有害な意図を最初から視覚要素(マーク・矢印・画像内注記・
  改変対象の視覚的指示)として構成し、テキストプロンプト側は無害・曖昧な指示にとどめる。
- 結果として、テキストのみを検査する安全フィルタは正常系のプロンプトしか見えず、
  実際の有害な編集意図は画像側の視覚チャネルに隠蔽されたまま実行される。

### 1.3 IESBench(ベンチマーク)

| 項目 | 値 |
|---|---|
| 画像数 | 1,054 |
| 安全ポリシー数 | 15 |
| 属性 (attributes) | 116 |
| 編集アクション (actions) | 9 |

各エントリのスキーマ(概念):

```json
{
  "image_id": "string",
  "image_path": "string",
  "question": "意図する画像編集内容(自然文)",
  "attributes": ["編集対象となる属性のリスト"],
  "action": "編集操作(例: add / remove / replace / style-transfer ...)",
  "category": "違反する安全ポリシーのカテゴリ",
  "rewrite": "LLMによって書き換えられた(無害化された)テキストプロンプト"
}
```

本リポジトリの `src/dataset/iesbench_schema.py` はこのスキーマ互換の dataclass / JSON Schema を提供する。

### 1.4 評価指標

- **ASR (Attack Success Rate):** 安全ガードを突破した割合
- **HS (Harmfulness Score, 1–5):** 生成された編集結果の有害度(LLM-as-judge等で採点)
- **EV (Editing Validity):** 突破した上で「意図通りの意味のある編集」が行われたか
- **HRR (High Risk Ratio):** ASR かつ EV も満たす、実効的に危険な生成の割合
  (ASRだけでは「突破したが意味不明な出力」も含んでしまうため、HRRの方が実害に近い指標)

### 1.5 主要な結果(論文報告値の要約)

- 弱アライメントモデル(Qwen-Image-Edit, Seedream 4.5等)はテキスト攻撃・VJA双方に脆弱だが、VJAの方がASRが高い。
- 商用モデルでも Nano Banana Pro で ASR 80.9%, GPT-Image-1.5 で ASR 70.1%(論文/リポジトリ記載値)に達するケースが報告されている。
- 提案防御(内省的マルチモーダル推論による training-free トリガー)は、攻撃面を「視覚→言語」へ引き戻すことで、
  追加学習なしに安全性を大きく改善する。

## 2. 本リポジトリでの追実験方針

公式データセット・重みへの直接アクセスが本セッションの権限/ネットワークでは制限されるため、
以下の二段構えで進める。

1. **スキーマ互換パイプラインの構築**(本リポジトリの主目的)
   - IESBench 互換スキーマでの合成データセット生成・最適化アルゴリズム
   - VJA型(視覚指示中心)/ 旧来型(テキスト→画像タイポグラフィ)双方の攻撃生成器
   - ASR/HS/EV/HRR を計算する評価ハーネス
   - DPO・guard分類器・introspective defense の学習/推論コード
2. **実データ・実モデルでの追実験**(利用者側で許可された環境が必要)
   - `docs/05_reproduction_guide.md` の手順に従い、`CSU-JPG/IESBench` を取得し、
     本リポジトリのハーネスに読み込ませることでそのまま追実験可能な設計にしてある
     (`src/dataset/iesbench_schema.py` のローダーは公式JSON形式をそのまま受け付ける)

## 3. 関連研究(比較対象・ベースライン)

- **FigStep** (Gong et al.): 有害指示をタイポグラフィで画像化し「空欄を埋めよ」形式で誘導
- **MM-SafetyBench**: テキスト→画像変換 + 補助的テキストプロンプトの体系的ベンチマーク
- **Visual Adversarial Examples** (Qi et al.): 勾配ベースの敵対的摂動によるVLM Jailbreak
- **HADES / SneakyPrompt系**: 拡散モデルを用いた有害概念の画像的難読化
- **BlueSuffix**: 強化学習によるブルーチーミング(防御側)フレームワーク
- **MMJ-Bench / IESBench**: 攻撃・防御を横断的に比較するベンチマーク群

これらは `docs/02_attack_enhancement_proposals.md` および `docs/03_defense_survey.md` で
攻撃強化・防御双方の設計の参照点として扱う。

画像編集モデル以外(GUIエージェント、動画理解モデル、マルチ画像文脈、
Vision-Language-Actionロボティクス等)への"visual-to-visual"攻撃構造の一般化と、
それぞれの領域での関連研究(PopupAttack, TempJail, SIVA/MIDAS, VLA-Hijack等)は
`docs/08_visual_to_visual_threat_expansion.md` にまとめている。
