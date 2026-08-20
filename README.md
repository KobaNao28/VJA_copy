# VJA_copy — Vision-Centric Jailbreak Attack 追実験・防御研究キット

本リポジトリは、ICML 2026 Oral 論文
**"When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models"**
(Hou et al., 2026, arXiv:2602.10179 / GitHub: `CSU-JPG/VJA`, ベンチマーク: **IESBench**)
の追実験、および **VLM (Vision-Language Model) の Jailbreak に対する防御研究** を進めるための作業一式です。

> **⚠️ 利用目的について**
> 本リポジトリのコードは「画像編集モデル・VLM の安全性評価」「防御手法の研究開発」を目的とした
> レッドチーミング/ブルーチーミング教材です。攻撃生成コードは常に「どう防ぐか」とセットで使うことを
> 前提にしています。実際の商用サービスに対する無許可の攻撃実行、生成した有害コンテンツの拡散は
> 禁止します。社内 red-team、学術研究、CTF、モデル安全性評価など、正当な権限がある文脈でのみ使用してください。

---

## 1. 追実験に必要なもの(チェックリスト)

詳細は [`docs/05_reproduction_guide.md`](docs/05_reproduction_guide.md) を参照してください。要点のみ:

| 区分 | 内容 |
|---|---|
| データセット | IESBench (1,054 画像 / 15 safety policy / 116 attributes / 9 actions)。公式配布は Hugging Face `CSU-JPG/IESBench`。本リポジトリでは **スキーマ互換のサンプル生成器** (`src/dataset/`) を同梱し、実データがなくてもパイプラインを検証可能 |
| 対象モデル | 画像編集VLM: Qwen-Image-Edit, Seedream, GPT-Image-1(.5), Nano Banana / Nano Banana Pro 等(API or 重みが必要) |
| 計算資源 | 攻撃生成自体はCPUで可。ローカルOSSモデルでの評価・防御学習はGPU(できればVRAM 24GB以上)推奨 |
| 評価指標 | ASR (Attack Success Rate) / HS (Harmfulness Score 1–5) / EV (Editing Validity) / HRR (High Risk Ratio) — `src/eval/metrics.py` に実装 |
| ライブラリ | `requirements.txt` 参照(画像・フォント処理、OCR、Transformers/TRL(DPO)、CLIP 等) |
| 判定用モデル | HS/EVのLLM-as-judge用に GPT-4級 or Claude 級 VLM への API アクセス(任意。ローカルjudgeモデルでも代替可) |

---

## 2. フォルダ構成

```
VJA_copy/
├── README.md                       # 本ファイル
├── requirements.txt
├── docs/
│   ├── 00_overview.md              # VJA論文サマリと追実験の全体像
│   ├── 01_threat_scenarios.md      # VJAの脅威シナリオ分析
│   ├── 02_attack_enhancement_proposals.md  # 攻撃強化の技術的方向性(防御設計目的)
│   ├── 03_defense_survey.md        # DPO等 一般的安全アライメント手法サーベイと弱点・改善案
│   ├── 04_ideal_dataset_design.md  # Jailbreak防御用「理想的データセット」設計 + 統一ガイドライン
│   └── 05_reproduction_guide.md    # 追実験の具体的手順書
├── src/
│   ├── attack/
│   │   ├── typography_attack.py    # テキスト→画像タイポグラフィ攻撃(VJA/FigStep系の再現)
│   │   ├── shape_obfuscation.py    # 文字の完全図形化(アウトライン/ベクター化)
│   │   ├── variant_generator.py    # フォント・色・言語・サイズを変えたバリアント一括生成
│   │   ├── compare_optimize.py     # バリアント比較・最適化(OCR可読性×検知回避のパレート最適化)
│   │   └── adaptive_attack_optimizer.py  # 学習済みguard classifierに対する適応的攻撃再最適化(closed-loop red teaming)
│   ├── dataset/
│   │   ├── iesbench_schema.py      # IESBench互換スキーマ(dataclass/JSON Schema)
│   │   ├── dataset_optimizer.py    # データセット構築の最適化アルゴリズム(被覆率×多様性×難易度)
│   │   └── build_dataset.py        # 上記を用いたサンプルデータセット構築CLI
│   ├── defense/
│   │   ├── train_safety_dpo.py     # マルチモーダルDPOによる安全アライメント学習
│   │   ├── train_guard_classifier.py # 画像+テキスト Jailbreak検知器の学習
│   │   ├── introspective_defense.py  # training-free「内省的マルチモーダル推論」防御の再現
│   │   └── unified_defense_pipeline.py # 上記を多層防御として統合するランタイム
│   ├── eval/
│   │   ├── metrics.py              # ASR/HS/EV/HRR
│   │   └── run_eval.py             # 攻撃×防御の一括評価CLI
│   └── utils/
│       ├── io_utils.py
│       └── seed.py
├── data/sample/                    # 自動生成されるサンプル画像・データセット(実データは含まない)
├── scripts/
│   ├── setup.sh                    # 環境構築
│   └── run_full_pipeline.sh        # 攻撃生成→データセット構築→防御学習→評価 を一気通貫実行
└── outputs/                        # 実行結果(生成物, gitignore対象)
```

## 3. クイックスタート

```bash
bash scripts/setup.sh
bash scripts/run_full_pipeline.sh
```

個別に動かす場合:

```bash
# 1. タイポグラフィ攻撃 + 図形化 + バリアント生成
python -m src.attack.variant_generator --text "example harmful instruction placeholder" \
    --out data/sample/variants --fonts sans,serif,mono --colors black,red,gradient \
    --languages en,ja --sizes 24,48,96 --shape-levels outline,filled,vector

# 2. バリアント比較・最適化(OCR可読性 vs 検知回避のパレートフロント)
python -m src.attack.compare_optimize --variants-dir data/sample/variants --out outputs/pareto_report.json

# 3. データセット構築(最適化アルゴリズムでカバレッジを最大化)
python -m src.dataset.build_dataset --n-target 200 --out data/sample/iesbench_like.jsonl

# 4. 防御学習(DPO / guard classifier / introspective defense)
python -m src.defense.train_safety_dpo --config configs/dpo_default.yaml   # 要GPU
python -m src.defense.train_guard_classifier --data data/sample/iesbench_like.jsonl

# 5. 評価(ASR/HS/EV/HRR に加え、良性コントロール群による FBR=過剰拒否率も算出)
python -m src.eval.run_eval --dataset data/sample/iesbench_like.jsonl --defense introspective

# 6. 学習済みguard classifierに対する適応的攻撃再最適化(closed-loop red teaming)
python -m src.attack.adaptive_attack_optimizer --guard-ckpt outputs/guard_classifier.pt --n-steps 40
```

## 4. ドキュメント一覧

- [`docs/00_overview.md`](docs/00_overview.md) — VJA論文の要約と追実験全体像
- [`docs/01_threat_scenarios.md`](docs/01_threat_scenarios.md) — 脅威シナリオ
- [`docs/02_attack_enhancement_proposals.md`](docs/02_attack_enhancement_proposals.md) — 攻撃強化の提案(防御設計のため)
- [`docs/03_defense_survey.md`](docs/03_defense_survey.md) — DPO等の一般的安全アライメント手法サーベイ・弱点・改善案
- [`docs/04_ideal_dataset_design.md`](docs/04_ideal_dataset_design.md) — 理想的な安全データセット設計・統一ガイドライン
- [`docs/05_reproduction_guide.md`](docs/05_reproduction_guide.md) — 追実験の具体的手順

## 5. ライセンス・出典

- 原論文: Hou et al., "When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models", ICML 2026 (arXiv:2602.10179)
- 公式実装/データ: https://github.com/CSU-JPG/VJA , https://huggingface.co/datasets/CSU-JPG/IESBench (アクセスには別途申請・利用規約の確認が必要な場合があります)
- 本リポジトリのコードはオリジナル実装であり、公式リポジトリのコードをコピーしたものではありません。
