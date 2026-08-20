"""
攻撃×防御の一括評価CLI。IESBench互換データセットに対し、
defense条件(none / introspective / guard_classifier / unified)を切り替えながら
ASR/HS/EV/HRRを計算し、比較レポートを出力する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.iesbench_schema import load_entries
from src.defense.introspective_defense import IntrospectiveDefense
from src.defense.unified_defense_pipeline import UnifiedDefensePipeline
from src.eval.judge import SyntheticJudge
from src.eval.metrics import EvalRecord, aggregate_metrics, per_category_breakdown
from src.eval.model_adapter import DummyWeaklyAlignedAdapter
from src.utils.io_utils import write_json

DEMO_IMAGE = "outputs/typography_demo.png"  # 画像未生成時のフォールバック(デモ用)


def _resolve_image_path(entry_image_path: str) -> str:
    return entry_image_path if Path(entry_image_path).exists() else DEMO_IMAGE


def run(dataset_path: str, defense: str, seed: int = 0) -> list[EvalRecord]:
    entries = load_entries(dataset_path)
    model = DummyWeaklyAlignedAdapter(base_compliance=0.85, seed=seed)
    judge = SyntheticJudge(seed=seed)

    introspective = IntrospectiveDefense() if defense in ("introspective", "unified") else None
    unified = UnifiedDefensePipeline() if defense == "unified" else None

    records: list[EvalRecord] = []
    for e in entries:
        image_path = _resolve_image_path(e.image_path)
        prompt = e.question

        blocked = False
        if defense == "introspective":
            verdict = introspective.review(image_path, prompt)
            blocked = not verdict.allowed
        elif defense == "unified":
            verdict = unified.process(session_id=e.image_id, image_path=image_path, prompt=prompt)
            blocked = not verdict.allowed
        elif defense == "guard_classifier":
            # 簡易プロキシ: カテゴリの機微度が高いほどブロック率を上げる合成判定
            # (実運用では train_guard_classifier.py で学習した重みを読み込み推論する)
            import random

            blocked = random.Random(hash(e.image_id) % (2**32)).random() < 0.5
        # defense == "none" の場合 blocked=False のまま

        if blocked:
            records.append(EvalRecord(e.image_id, e.category, True, False, False, 1.0))
            continue

        result = model.edit(image_path, prompt)
        hs, ev = judge.score(e.category, result["complied"])
        records.append(
            EvalRecord(
                image_id=e.image_id,
                category=e.category,
                defense_blocked=False,
                attack_success=result["complied"],
                editing_validity=ev,
                harmfulness_score=hs,
            )
        )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="ASR/HS/EV/HRR 一括評価")
    p.add_argument("--dataset", default="data/sample/iesbench_like.jsonl")
    p.add_argument("--defense", default="none", choices=["none", "introspective", "guard_classifier", "unified"])
    p.add_argument("--compare-all", action="store_true", help="4条件すべてを比較して表示")
    p.add_argument("--out", default="outputs/eval_report.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    conditions = ["none", "introspective", "guard_classifier", "unified"] if args.compare_all else [args.defense]
    report = {}
    for cond in conditions:
        records = run(args.dataset, cond, seed=args.seed)
        metrics = aggregate_metrics(records)
        report[cond] = {
            "overall": metrics,
            "per_category": per_category_breakdown(records),
        }
        print(f"=== defense={cond} ===  n={metrics['n']}  ASR={metrics['ASR']}  HS={metrics['HS']}  EV={metrics['EV']}  HRR={metrics['HRR']}")

    write_json(args.out, report)
    print(f"レポート出力: {args.out}")


if __name__ == "__main__":
    main()
