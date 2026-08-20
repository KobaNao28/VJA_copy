#!/usr/bin/env bash
# 攻撃バリアント生成 -> 比較最適化 -> データセット構築 -> 防御学習(mock可) -> 評価
# を一気通貫で実行し、パイプライン全体の疎通を検証する。
set -euo pipefail
cd "$(dirname "$0")/.."

MOCK_FLAG="--mock"
DPO_MODEL_ARGS=()
if [[ -n "${MODEL_NAME:-}" ]]; then
  MOCK_FLAG=""
  DPO_MODEL_ARGS=(--model-name "$MODEL_NAME")
  echo "[real-model] MODEL_NAME=$MODEL_NAME でDPO学習を実行します"
else
  echo "[dry-run] MODEL_NAME未指定のため、外部モデル不要のmockモードでDPO配線を検証します"
fi

echo "== 1. タイポグラフィ攻撃バリアント生成 =="
python3 -m src.attack.variant_generator \
  --fonts sans,serif,mono --colors black,red,gray_low_contrast \
  --languages en,ja --sizes 24,48,96 \
  --shape-levels raw,outline,filled,fragmented,vector,noise_edge \
  --out data/sample/variants

echo "== 2. バリアント比較・パレート最適化 =="
python3 -m src.attack.compare_optimize --variants-dir data/sample/variants --out outputs/pareto_report.json

echo "== 3. データセット構築(被覆×多様性×難易度の最適化) =="
python3 -m src.dataset.build_dataset --n-target 200 --out data/sample/iesbench_like.jsonl

echo "== 4. DPO選好データ合成 + 学習 =="
python3 -m src.defense.train_safety_dpo \
  --build-from-iesbench data/sample/iesbench_like.jsonl \
  --data data/sample/dpo_preferences.jsonl \
  $MOCK_FLAG "${DPO_MODEL_ARGS[@]}" --epochs 5

echo "== 5. Guard classifier 学習 =="
python3 -m src.defense.train_guard_classifier --data data/sample/variants/manifest.jsonl --epochs 5

echo "== 6. 評価(defense条件比較) =="
python3 -m src.eval.run_eval --dataset data/sample/iesbench_like.jsonl --compare-all --out outputs/eval_report.json

echo "完了。生成物は data/sample/ と outputs/ を参照してください。"
