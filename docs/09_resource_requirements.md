# 実行環境要件: VRAM・ディスク容量・データセット

本リポジトリの全モジュールについて、実際にこの開発環境(4 vCPU, RAM 15GB, GPU無し,
`torch==2.13.0+cu130`)で実行して計測した値を基に、必要リソースをまとめる。
**結論から言うと、`train_safety_dpo.py --model-name <実モデル>` を使う場合を除き、
本リポジトリの全パイプラインはGPU無し・CPUのみで動作する**(既定の`--mock`や
他の学習スクリプトはいずれもCPU専用で、`.cuda()`は一切呼ばれない)。

## 1. VRAM(GPU)要件

| 実行対象 | GPU要否 | 目安VRAM |
|---|---|---|
| `variant_generator.py` / `mark_variant_generator.py` / `ui_injection_variant_generator.py` / `trajectory_variant_generator.py`(攻撃バリアント生成) | 不要 | — |
| `compare_optimize.py` / `adaptive_attack_optimizer.py`(OCR・パレート最適化・closed-loop探索) | 不要 | — |
| `dataset_optimizer.py` / `build_dataset.py` / `lexicon_optimizer.py`(データセット構築) | 不要 | — |
| `train_guard_classifier.py` / `mark_detector.py` / `ui_injection_detector.py` / `trajectory_detector.py`(軽量CNN/MLP学習) | 不要 | — (数万〜数十万パラメータ規模、CPUで数分以内) |
| `train_safety_dpo.py --mock`(TinyCharTransformer) | 不要 | — |
| `curriculum_dpo.py`(TinyCharTransformer、全戦略比較) | 不要 | — |
| `immune_memory_defense.py` / `unified_defense_pipeline.py`(推論のみ) | 不要 | — |
| `run_eval.py` / `vja_gap_eval.py`(評価ハーネス) | 不要 | — |
| **`train_safety_dpo.py --model-name <実モデル>`**(実VLM/LLMをLoRAで安全アライメント学習する場合のみ) | **要(推奨)** | モデル規模に応じ下表参照 |

### 1.1 `--model-name` 実行時のVRAM目安(LoRA + fp16、`peft`使用)

| モデル規模 | LoRA微調整(推奨) | フル微調整(非推奨、参考) |
|---|---|---|
| 1B〜3B(例: 小型VLM) | 6〜10GB | 20〜30GB |
| 7B〜8B(例: Qwen2-VL-7B, LLaVA-7B相当) | 16〜20GB | 60GB超 |
| 13B前後 | 24〜32GB | 100GB超 |

- `requirements.txt` に含まれる `bitsandbytes` で4bit/8bit量子化すれば、上記目安の
  概ね半分〜1/4程度まで削減可能(精度とのトレードオフに注意)。
- **重要な既知の制約**: 現在の `train_safety_dpo.py::build_hf_policy_and_ref()` は
  `transformers.AutoModelForCausalLM` を使用しており、テキスト生成インターフェースを
  持つモデル(VLMのチャット/推論バックボーンとして使われるもの等)には対応するが、
  Qwen-Image-Edit等の**拡散モデルベースの画像編集モデル本体を直接LoRA学習する
  用途には未対応**(`docs/06_novel_defense_proposals.md` 1.4節でも言及)。
  実際の画像編集モデルへの適用には `AutoModelForVision2Seq` 等への差し替えと、
  画像入力(pixel_values)を条件付けに使うようforward呼び出しを拡張する必要がある。
- GPU無し環境では `--mock` を使うことで配線検証は可能(本リポジトリの開発・検証も
  全て `--mock` で行った)。

## 2. ディスク容量(実測値)

### 2.1 ソフトウェア依存関係

| 項目 | サイズ | 備考 |
|---|---|---|
| `torch`(CUDA同梱版, `cu130`) | 約1.2GB | **CPUのみで良い場合は不要**。`pip install torch --index-url https://download.pytorch.org/whl/cpu` 相当のCPU専用版なら数百MB程度まで縮小可能 |
| `nvidia-*` CUDAライブラリ群(torchのCUDA依存) | 約2.7GB | 同上、CPU専用torchなら不要 |
| `opencv-python-headless` | 約72MB | |
| `pillow` / `numpy` | 約7MB / 約45MB | |
| Tesseract OCR本体+言語データ(eng/jpn) | 約17MB | `apt-get install tesseract-ocr tesseract-ocr-jpn` |
| フォント一式(既存OSのプリインストール分) | 約94MB | Noto/DejaVu/Liberation/WenQuanYi等。最小構成ならsans/serif/mono各1書体で十分 |
| **合計目安(CUDA版torch込み)** | **約4.9GB** | |
| **合計目安(CPU専用torchに置き換えた場合)** | **約1.5〜2GB程度** | 本リポジトリの用途ではこちらで十分 |

### 2.2 リポジトリ本体(コード)

| 項目 | サイズ |
|---|---|
| `src/`(全モジュール) | 716KB |
| `docs/`(全ドキュメント) | 116KB |
| `tests/` | 68KB |
| `scripts/` | 12KB |
| **合計** | **約930KB** |

### 2.3 生成データ(実測、本開発環境でのフル実行後)

いずれも合成データであり、生成のたびに再現可能(gitignore対象、リポジトリには含まれない)。

| 生成物 | 件数 | 実測サイズ |
|---|---|---|
| `data/sample/variants/`(タイポグラフィ攻撃バリアント) | 324件 | 4.8MB |
| `data/sample/visual_marks/`(VJA型視覚指示マーク) | 1,200件 | 11MB |
| `data/sample/ui_injection/`(GUI注入スクリーンショット) | 72件 | 740KB |
| `data/sample/trajectories/`(時系列軌跡, 16フレーム×270系列) | 4,320フレーム | 19MB |
| `data/sample/images/`(IESBench互換データセットのプレースホルダー画像) | 200件 | 1.6MB |
| **合計(このセッションでのフル実行分)** | | **約37MB** |

### 2.4 モデルチェックポイント(実測)

全て軽量CNN/MLP(数万〜数十万パラメータ)であり、実VLMの重みは一切含まない。

| チェックポイント | サイズ |
|---|---|
| `outputs/guard_classifier.pt` | 50KB |
| `outputs/mark_detector.pt` | 107KB |
| `outputs/ui_injection_detector.pt` | 105KB |
| `outputs/trajectory_detector.pt` | 22KB |
| (`train_safety_dpo.py --mock`のTinyCharTransformerは既定で保存しない。保存時も同程度の数百KB〜数MB) |

**目安として、`outputs/`ディレクトリ一式(チェックポイント+デモ画像+レポートJSON)で
2MB未満**(実測1.8MB)。

### 2.5 合計ディスク容量の目安

| シナリオ | 目安 |
|---|---|
| リポジトリのクローンのみ | 1MB未満 |
| + Python依存関係(CPU専用torch) | 約1.5〜2GB |
| + Python依存関係(CUDA版torch、GPU環境向け) | 約4.9GB |
| + 本ドキュメントに記載の全パイプラインをデフォルト設定でフル実行 | 上記 + 約40MB |
| + 実IESBenchデータセット(下記3節) | 上記 + 数百MB〜数GB(公式配布のサイズに依存、未確認) |

## 3. データセット要件

### 3.1 本リポジトリの合成データ(追加ダウンロード不要)

以下は全て**このリポジトリ内のコードだけでゼロから生成可能**であり、外部データセットの
ダウンロードは一切不要:

- タイポグラフィ攻撃バリアント(`variant_generator.py`)
- VJA型視覚指示マーク(`mark_variant_generator.py`)
- GUI注入スクリーンショット(`ui_injection_variant_generator.py`)
- 時系列軌跡フレーム列(`trajectory_variant_generator.py`)
- IESBench互換の合成データセット(`build_dataset.py`、被覆最適化アルゴリズムで生成)
- 3段階脅威語彙データセット(`lexicon_optimizer.py`、内蔵のseed語彙から構築)
- DPO選好データ(`train_safety_dpo.py --build-from-iesbench`、上記の合成データセットから自動合成)

### 3.2 外部データセット(任意、実追実験の場合のみ必要)

| データセット | 用途 | 入手方法 | 本リポジトリでの必須度 |
|---|---|---|---|
| **IESBench**(公式, `CSU-JPG/IESBench`) | VJA論文の数値を厳密に再現する場合 | Hugging Face(要申請/規約確認、本セッションからはアクセス不可) | 任意。`src/dataset/iesbench_schema.py::load_entries()` が公式JSON形式をそのまま読み込める設計のため、入手できれば即座に差し替え可能 |
| 実写ベースの背景画像(mark_detector/ui_injection_detector/trajectory_detectorの本番学習用) | 本番相当の汎化性能を検証する場合 | 任意の画像データセット(COCO等) | 任意。現状は完全合成シーンで代替しており、`docs/07`/`docs/08` に記載の通りトイスケールでの機構検証に留まる |
| 対象VLM/画像編集モデルの重み | `train_safety_dpo.py --model-name` を実モデルで使う場合 | Hugging Face等(モデルごとのライセンス確認要) | `--mock` を使えば不要 |

### 3.3 データセット容量の参考値

- 生成済みサンプルデータセットは前掲2.3節の通り数MB〜十数MB程度。
- 公式IESBench(1,054画像)は、画像編集ベンチマークとしては中規模であり、
  一般的な画像解像度(数百KB/枚と仮定)から**数百MB〜1GB程度**と推定される
  (未確認。公式配布ページで要確認)。

## 4. 実行時間の目安(実測、4 vCPU環境)

| コマンド | 実測時間 |
|---|---|
| `variant_generator.py`(324バリアント) | 数秒 |
| `dataset_optimizer.py` / `build_dataset.py`(200件、ベクトル化後) | 約3秒 |
| `mark_detector.py` 学習(1,200サンプル, 12epoch) | 約96〜115秒 |
| `ui_injection_detector.py` 学習(72サンプル, 10epoch) | 約29秒 |
| `trajectory_detector.py` 学習(270系列, 300epoch + 単一フレーム基準比較) | 約3分22秒 |
| `curriculum_dpo.py`(7戦略比較, 60ペア, 2epoch) | 約28秒 |
| `vja_gap_eval.py`(324+600サンプル、OCR含む) | 数分程度(OCR呼び出しが律速) |
| `scripts/run_full_pipeline.sh --dry-run`(7ステップ一括) | 約3分22秒〜4分 |

## 5. まとめ: 最小構成での動作確認に必要なもの

1. Python 3.10+ 環境
2. `pip install -r requirements.txt`(GPU不要なら**CPU専用のtorch**に差し替え推奨、
   ディスクを約3GB節約できる)
3. `apt-get install tesseract-ocr tesseract-ocr-jpn`(OCR系機能を使う場合)
4. ディスク空き容量: 目安**2GB程度**(依存関係込み、CPU専用torch想定)
5. 外部データセットのダウンロードは一切不要(全パイプラインが自己完結)
6. GPU/VRAM: 不要(実VLMでのDPO学習を試す場合のみ、1節の目安を参照)
