"""
バリアント比較・最適化(パレート最適化)モジュール。

`variant_generator.py` が生成した各バリアントについて、
  - readability : OCR / VLMがどれだけ元テキストを読み取れるか(意図伝達力の代理指標)
  - detectability: 単純な画像特徴に基づく「テキストらしさ検知器」がどれだけ検知しやすいか
                    (安全フィルタの検知回避しやすさの代理指標。実運用では学習済みの
                     guard classifier `src/defense/train_guard_classifier.py` で置き換える)
を計算し、(readability を保ちつつ detectability を下げる)パレートフロントを求める。

**防御側の使い方**: このスクリプトを自社のフィルタに対して実行し、
「detectabilityが低いのにreadabilityが高い」= 防御の盲点になりうるバリアントを特定し、
そのパラメータ帯を優先的に学習データへ追加する(`src/dataset/dataset_optimizer.py` と連携)。
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import cv2
import numpy as np

from src.attack.variant_generator import DEFAULT_TEXT_BY_LANG
from src.utils.io_utils import read_jsonl, write_json


def ocr_readability(image_path: str, expected_text: str) -> float:
    """Tesseract OCR結果と期待テキストの類似度(0-1)。VLMの読解力の弱い代理指標。"""
    try:
        import pytesseract
        from PIL import Image

        lang = "jpn+eng" if any(ord(c) > 0x3000 for c in expected_text) else "eng"
        ocr_text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
    except Exception:
        ocr_text = ""
    ratio = difflib.SequenceMatcher(None, ocr_text.strip(), expected_text.strip()).ratio()
    return round(ratio, 4)


def detectability_heuristic(image_path: str) -> float:
    """
    単純な画像特徴量から「テキスト検知器に引っかかりやすさ」を推定するヒューリスティック。
    実際の安全フィルタ(OCRベース/CNNベースのテキスト検知)を模した代理指標であり、
    厳密な予測器ではない点に注意(本番運用では学習済み guard classifier を使うこと)。

    特徴量:
      - ストローク幅の均一性(通常フォントは均一、図形化・断片化すると不均一になる)
      - 連結成分の regularity(文字は概ね同サイズの連結成分が並ぶ)
      - エッジ密度
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    stroke_vals = dist[dist > 0]
    stroke_uniformity = 1.0 - min(1.0, float(np.std(stroke_vals)) / (float(np.mean(stroke_vals)) + 1e-6)) if len(stroke_vals) else 0.0

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        area_cv = float(np.std(areas)) / (float(np.mean(areas)) + 1e-6)
        component_regularity = 1.0 - min(1.0, area_cv)
        component_count_score = min(1.0, n_labels / 40.0)
    else:
        component_regularity = 0.0
        component_count_score = 0.0

    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.mean(edges > 0))

    score = 0.4 * stroke_uniformity + 0.35 * component_regularity + 0.15 * component_count_score + 0.10 * min(1.0, edge_density * 20)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """points: [[readability, -detectability], ...] を最大化する意味でのパレート効率点を返す。"""
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_efficient[i]:
            dominated = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
            is_efficient[dominated] = False
    return is_efficient


def evaluate_manifest(manifest_path: Path) -> list[dict]:
    rows = list(read_jsonl(manifest_path))
    for row in rows:
        expected_text = DEFAULT_TEXT_BY_LANG.get(row["language"], DEFAULT_TEXT_BY_LANG["en"])
        row["readability"] = ocr_readability(row["image_path"], expected_text)
        row["detectability"] = detectability_heuristic(row["image_path"])
        row["stealth_score"] = round(row["readability"] * (1.0 - row["detectability"]), 4)
    return rows


def compute_pareto(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    points = np.array([[r["readability"], -r["detectability"]] for r in rows])
    flags = is_pareto_efficient(points)
    for r, flag in zip(rows, flags):
        r["pareto_optimal"] = bool(flag)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="バリアント比較・パレート最適化")
    p.add_argument("--variants-dir", default="data/sample/variants")
    p.add_argument("--out", default="outputs/pareto_report.json")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    manifest_path = Path(args.variants_dir) / "manifest.jsonl"
    rows = evaluate_manifest(manifest_path)
    rows = compute_pareto(rows)
    rows_sorted = sorted(rows, key=lambda r: r["stealth_score"], reverse=True)

    report = {
        "n_variants": len(rows),
        "n_pareto_optimal": sum(r["pareto_optimal"] for r in rows),
        "top_stealth_variants": rows_sorted[: args.top_k],
        "all_variants": rows,
    }
    write_json(args.out, report)
    print(f"評価済みバリアント数: {len(rows)} (うちパレート最適: {report['n_pareto_optimal']})")
    print(f"レポート出力: {args.out}")
    print("--- 防御上の盲点になりうる上位バリアント (stealth_score降順) ---")
    for r in rows_sorted[: args.top_k]:
        print(
            f"{r['variant_id']:45s} readability={r['readability']:.2f} "
            f"detectability={r['detectability']:.2f} stealth={r['stealth_score']:.2f}"
        )


if __name__ == "__main__":
    main()
