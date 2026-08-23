# 実行環境要件: VRAM・ディスク容量・データセット

本リポジトリの全モジュールについて、実際にこの開発環境(4 vCPU, RAM 15GB, GPU無し,
`torch==2.13.0+cu130`)で実行して計測した値を基に、必要リソースをまとめる。
**結論から言うと、`train_safety_dpo.py --model-name <実モデル>` を使う場合を除き、
本リポジトリの全パイプラインはGPU無し・CPUのみで動作する**(既定の`--mock`や
他の学習スクリプトはいずれもCPU専用で、`.cuda()`は一切呼ばれない)。

> **本セッションでの外部データセット/重みの準備について**: IESBench本体・
> Qwen-Image-Edit重み・代替の実写データセット(COCO等)はいずれも
> `huggingface.co`/`modelscope.cn`/`zenodo.org`/`cocodataset.org`等でのみ配布されており、
> これらは本セッションの組織egressポリシーにより明示的にブロックされている
> (`curl`で直接疎通確認済み、403応答。ポリシー上リトライ・回避は行っていない)。
> **そのためこのセッション内では外部データセットを準備できなかった**。
> 3.2節に、ユーザー自身の環境で実行できる正確な入手手順・引用情報をまとめた。

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
| **`train_qwen_image_edit_dpo.py`**(実際のQwen-Image-Edit本体を4bit/8bit量子化+LoRAでDiffusion-DPO学習) | **要** | 32GB級を想定(1.1節参照) |
| **`run_eval.py --qwen-image-edit`**(実際のQwen-Image-Editで画像編集を実行し評価、学習ではなく推論のみ) | **要** | 学習時より軽い(下記注記) |

### 1.1 `--model-name` 実行時のVRAM目安(LoRA + fp16、`peft`使用)

| モデル規模 | LoRA微調整(推奨) | フル微調整(非推奨、参考) |
|---|---|---|
| 1B〜3B(例: 小型VLM) | 6〜10GB | 20〜30GB |
| 7B〜8B(例: Qwen2-VL-7B, LLaVA-7B相当) | 16〜20GB | 60GB超 |
| 13B前後 | 24〜32GB | 100GB超 |

- **この表は一般的なテキスト生成LLM/VLM(数B〜十数Bパラメータ)を想定した目安であり、
  VJA論文が実際に対象とする `Qwen/Qwen-Image-Edit`(約20Bパラメータ級の拡散モデル)には
  適用できない**。実モデル・データセットの詳細な調査結果(公式リポジトリの
  `requirements.txt`確認、正確なBibTeX等)は3.2節にまとめた。
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

### 1.2 `run_eval.py --qwen-image-edit`(実モデルでの評価、推論のみ)

`src/eval/qwen_image_edit_adapter.py` は `train_qwen_image_edit_dpo.py::load_real_pipeline()`
で確認済みのAPIを使い、実際のQwen-Image-Editで画像編集を実行して`run_eval.py`の
ASR/HS/EV/HRR評価に接続する(学習ではなく推論のみ)。

- **学習(32GB級)より軽いはず**: 推論時は勾配・オプティマイザ状態が不要なため、
  同じ4bit量子化+`enable_model_cpu_offload()`であれば学習より少ないVRAMで動く可能性が高い。
  ただし本セッションでは実際の重みをダウンロードできないため、**具体的なGB数は未実測**。
  まずは`--qwen-quantization 4bit`で試し、OOMになる場合は解像度を下げる/
  `--qwen-steps`(推論ステップ数)を減らす等で調整すること。
- **誠実な注記**: `pipe(image=..., prompt=..., num_inference_steps=..., true_cfg_scale=...)`
  という呼び出しは diffusers の画像編集系パイプラインの一般的な慣例に基づく実装であり、
  実際にQwen-Image-Editの重みに対して呼び出し確認はできていない。エラーが出た場合は
  `qwen_image_edit_adapter.py::_call_pipe()` の引数名を実際のAPIに合わせて調整すること。
- 「モデルが指示を拒否したか(`complied`)」の判定は本アダプタでは行わず、常に`True`を返す
  (拡散モデル本体は明示的な拒否機構が無い限り何らかの画像を生成するため)。実際の安全性判定は
  `judge.py::LLMJudge`(実LLMによる画像レビュー)で行うことを推奨する。

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
| + 実IESBenchデータセット(下記3.2節、本セッションでは未取得) | 上記 + 数百MB〜1GB程度(未確認) |
| + Qwen-Image-Edit本体の重み(実モデルでの追実験、本セッションでは未取得) | 上記 + **数十GB規模**(20Bパラメータ級、fp16で概算約40GB、未確認) |

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

### 3.2 外部データセット・重みの取得を試みた結果(本セッションでの実地検証)

**このセッションから実際に取得を試みたが、いずれも組織のegressポリシーにより
ブロックされており、本セッション内では準備できなかった**(`curl`で直接疎通確認、
`403 Forbidden`/`CONNECT tunnel failed`を確認済み。プロキシの仕様上
"Do not retry or route around it" と明記されているため、これ以上の回避は行っていない)。

| ホスト | 用途 | 疎通確認結果 |
|---|---|---|
| `huggingface.co` | IESBench本体、Qwen-Image-Edit等の重み | ブロック(403) |
| `modelscope.cn` | 代替ホスティングの可能性を確認 | ブロック(403) |
| `zenodo.org` / `archive.org` | 学術データの代替ホスティング | ブロック(403) |
| `cocodataset.org` / `paperswithcode.com` / `kaggle.com` | 実写背景画像の代替入手先 | ブロック(403) |

一方、`github.com`(公式実装リポジトリ `CSU-JPG/VJA` のclone)は疎通可能だったため、
**コード・README・ライセンス情報等は取得済み**(データセット本体は含まれていないことを確認)。
以下は公式リポジトリから直接確認できた正確な情報。

#### IESBench(データセット本体)

- **入手先**: https://huggingface.co/datasets/CSU-JPG/IESBench (これのみが唯一の配布元。
  GitHubリポジトリにはデータ本体は同梱されていない)
- **スキーマ**(公式README記載、本リポジトリの`iesbench_schema.py`と完全一致するよう設計済み):
  `question` / `image-path` / `attributes` / `action` / `category` / `rewrite` / `image_id`
- **入手手順(ユーザー側の環境で実行)**:
  ```bash
  pip install huggingface_hub
  huggingface-cli download CSU-JPG/IESBench --repo-type dataset --local-dir ./iesbench_official
  # または
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='CSU-JPG/IESBench', repo_type='dataset', local_dir='./iesbench_official')"
  ```
  取得後、`python -m src.dataset.iesbench_schema --load ./iesbench_official --validate` で
  本リポジトリのローダーにそのまま読み込めることを確認できる。
- **論文**: Hou, Jiacheng / Sun, Yining / Jin, Ruochong / Han, Haochen / Liu, Fangming /
  Chan, Wai Kin Victor / Wang, Alex Jinpeng. "When the Prompt Becomes Visual: Vision-Centric
  Jailbreak Attacks for Large Image Editing Models." ICML 2026 (Oral). arXiv:2602.10179.
  ```bibtex
  @misc{hou2026vja,
        title={When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models},
        author={Jiacheng Hou and Yining Sun and Ruochong Jin and Haochen Han and Fangming Liu and Wai Kin Victor Chan and Alex Jinpeng Wang},
        year={2026}, eprint={2602.10179}, archivePrefix={arXiv}, primaryClass={cs.CV},
        url={https://arxiv.org/pdf/2602.10179}}
  ```

#### 対象モデル(画像編集モデル本体)

公式リポジトリの`src/requirements.txt`を確認したところ、防御実装は
`torch==2.8.0` / `diffusers==0.36.0.dev0` / `transformers==4.57.1` に依存し、
ベースモデルは既定で **`Qwen/Qwen-Image-Edit`**(Hugging Face, 拡散モデルベースの
画像編集モデル)。これは本リポジトリの`train_safety_dpo.py`が前提とする
`transformers.AutoModelForCausalLM`(テキスト生成インターフェース)とは
**アーキテクチャが異なる**(`diffusers`のパイプラインクラスであり、pixel_values由来の
画像条件付けを直接扱う)。`docs/06_novel_defense_proposals.md` 1.4節で述べた
「現実装は拡散モデル型の画像編集モデル本体には未対応」という指摘は、この確認により
裏付けられた。実際にQwen-Image-Edit自体をDPOで安全アライメントする場合は
`diffusers`のLoRA/DreamBooth系ユーティリティに合わせた学習ループへの置き換えが必要。

- 入手先: https://huggingface.co/Qwen/Qwen-Image-Edit(同じくブロック対象ホスト)
- モデル規模: Qwen-Imageファミリーは約20Bパラメータ級の拡散トランスフォーマーであり、
  推論だけでも一般的なコンシューマ向けGPU(VRAM 16GB以下)では収まらない可能性が高い
  (公式`src/run.py`に`--cpu_offload`オプションが用意されているのはこのため)。
  1節に記載したVRAM目安(1-13Bクラスの一般的なLLM/VLM微調整を想定した値)は
  **この規模の拡散モデル本体には適用できない**点に注意。

#### リーダーボード数値(公式README, 参考値として記録)

公式リポジトリのREADMEに掲載されている全モデルの数値をそのまま転記する
(`docs/00_overview.md`の要約値の裏付けとして、また追実験時の比較対象として)。

| モデル | ASR (AVG) | HS (AVG) | EV (AVG) | HRR (AVG) |
|---|---:|---:|---:|---:|
| [O] Qwen-Image-Edit-Safe (公式提案手法) | 66.9 | 3.4 | 62.8 | 61.7 |
| [C] GPT Image 1.5 | 70.3 | 3.2 | 63.0 | 52.0 |
| [C] Nano Banana Pro | 80.9 | 3.8 | 79.1 | 70.6 |
| [C] Seedream 4.5 | 94.1 | 4.4 | 86.3 | 83.8 |
| [C] Qwen-Image-Edit (オンライン版, 無防御) | 97.5 | 4.1 | 87.7 | 73.8 |
| [O] BAGEL | 100.0 | 4.1 | 82.0 | 70.6 |
| [O] Flux2.0 [dev] | 100.0 | 4.6 | 87.1 | 84.6 |
| [O] Qwen-Image-Edit* (ローカル版, 無防御) | 100.0 | 4.6 | 92.9 | 90.3 |

([C]=商用モデル, [O]=オープンソースモデル。ASR/HS/EV/HRRの定義は`src/eval/metrics.py`と同一。
per-category(I1〜I15)の内訳値は公式READMEに全て記載されているが、本表では紙面の都合上AVGのみ抜粋)

### 3.3 データセット容量の参考値

- 生成済みサンプルデータセットは前掲2.3節の通り数MB〜十数MB程度。
- 公式IESBench(1,054画像)の正確なファイルサイズは、配布元(Hugging Face)への
  アクセスがブロックされているため本セッションからは直接確認できなかった
  (画像編集ベンチマークの一般的な傾向から数百MB〜1GB程度と推定されるが未確認)。
- Qwen-Image-Editの重み一式は、20Bパラメータ級の拡散モデルとして
  **数十GB規模**になる可能性が高い(fp16で約40GB程度が一般的な目安、要確認)。

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
7. 論文の数値を厳密に追実験する場合のみ、ユーザー自身の環境(このセッションの
   egressポリシー外)で `pip install huggingface_hub` の上、3.2節の手順で
   IESBenchと`Qwen/Qwen-Image-Edit`を取得してください(合計ディスク容量の
   目安は2.5節、モデル規模はQwen-Imageファミリー約20Bパラメータ級)
