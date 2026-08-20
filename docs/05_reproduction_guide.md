# VJA 追実験ガイド

## 1. 環境構築

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Tesseract本体(OCR評価用)
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
```

GPUで防御学習(DPO/guard classifier)を回す場合は CUDA 対応 PyTorch を別途インストールしてください。

## 2. 必要な外部リソース(入手要否チェックリスト)

| リソース | 必須/任意 | 入手方法 | 本リポジトリでの代替 |
|---|---|---|---|
| IESBench データセット | 追実験の再現性を厳密に取るなら必須 | Hugging Face `CSU-JPG/IESBench`(要申請/規約確認) | `src/dataset/build_dataset.py` でスキーマ互換の合成データを自動生成 |
| 画像編集VLM(評価対象) | 必須 | 商用API(GPT-Image, Gemini/Nano Banana等)またはOSS重み(Qwen-Image-Edit等) | `src/eval/run_eval.py` はモデル呼び出しを `ModelAdapter` として抽象化(ダミーアダプタ同梱) |
| Judgeモデル(HS/EV採点) | 必須(自動評価する場合) | GPT-4級/Claude級APIキー、またはローカルVLM | ルールベースjudgeのフォールバックを実装 |
| フォント(多言語比較用) | 推奨 | Noto Sans/Serif等をシステムにインストール | フォント未検出時はPillowデフォルトフォントにフォールバック |
| GPU(DPO学習) | 防御学習を実行するなら推奨 | ローカル or クラウドGPU | `--dry-run` で1ステップだけCPU実行し配線を検証可能 |

## 3. 手順

### Step 1: パイプラインの疎通確認(データなしで実行可能)

```bash
bash scripts/run_full_pipeline.sh --dry-run
```

これは合成データのみで攻撃生成→データセット構築→(擬似)防御学習→評価まで一気通貫させ、
コードパスにバグがないかを確認するためのモードです。

### Step 2: 実データセットでの追実験(オプション)

1. `CSU-JPG/IESBench` を取得し `data/iesbench/` に配置(利用規約に従うこと)
2. `python -m src.dataset.iesbench_schema --load data/iesbench --validate` でスキーマ検証
3. `python -m src.eval.run_eval --dataset data/iesbench/annotations.json --model-adapter <対象モデル>` で評価

### Step 3: 対象モデルの接続

`src/eval/model_adapter.py`(`run_eval.py` 内で定義)の `ModelAdapter` 抽象クラスを継承し、
`edit(image, prompt) -> image` を実装すれば任意のモデル(商用API/ローカルOSS)に差し替え可能です。

### Step 4: 防御の適用と比較

```bash
# 素のモデル
python -m src.eval.run_eval --dataset <path> --defense none
# introspective defense (training-free)
python -m src.eval.run_eval --dataset <path> --defense introspective
# guard classifier で前段フィルタ
python -m src.eval.run_eval --dataset <path> --defense guard_classifier --guard-ckpt outputs/guard.pt
# DPOで安全アライメント済みのモデル自体を評価
python -m src.eval.run_eval --dataset <path> --model-adapter <DPO後のモデル>
```

ASR/HS/EV/HRR を `none` / `introspective` / `guard_classifier` / `dpo_aligned` の4条件で
比較することで、論文の Table 相当(防御手法比較)を再現できます。

## 4. 再現性のための注意点

- 論文の絶対値(ASR 80.9%など)は評価対象モデルのバージョン・時期に強く依存するため、
  「絶対値の完全一致」ではなく「相対的な傾向(VJA > テキストのみ攻撃、防御ありで大幅減少)」の
  再現を目標にすることを推奨します。
- Judgeモデルの違いでHS/EVの値は変動するため、比較実験では同一judgeを使い続けてください。
- 有害生成物を保存・共有する際は、社内ポリシー・法令を遵守し、アクセス制御された環境に限定してください。
