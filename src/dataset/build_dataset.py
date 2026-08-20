"""IESBench互換のサンプルデータセットをビルドするCLI。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.attack.shape_obfuscation import obfuscate
from src.attack.typography_attack import TypographySpec
from src.dataset.dataset_optimizer import build_candidate_pool, candidates_to_entries, coverage_report, greedy_optimize
from src.dataset.iesbench_schema import save_entries
from src.utils.io_utils import write_json
from src.utils.seed import set_seed

_SHAPE_LEVELS_CYCLE = ["raw", "outline", "filled", "fragmented", "noise_edge"]


def render_placeholder_images(entries, seed: int = 42) -> None:
    """
    各エントリの image_path にプレースホルダー画像を実際にレンダリングする。
    (下流の defense/eval スクリプトが実画像を前提に動作できるようにするため。
     テキストは常に一般的なプレースホルダー文言であり、有害内容は含まない)
    """
    for i, e in enumerate(entries):
        spec = TypographySpec(
            text=f"[PLACEHOLDER] category={e.category} action={e.action}",
            font_style="sans",
            font_size=28,
            color="black",
        )
        level = _SHAPE_LEVELS_CYCLE[i % len(_SHAPE_LEVELS_CYCLE)]
        out_dir = Path(e.image_path).parent
        img = obfuscate(spec, level, out_dir=out_dir)
        Path(e.image_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(e.image_path)


def main() -> None:
    p = argparse.ArgumentParser(description="最適化アルゴリズムでIESBench互換データセットを構築")
    p.add_argument("--n-target", type=int, default=200)
    p.add_argument("--out", default="data/sample/iesbench_like.jsonl")
    p.add_argument("--report-out", default="outputs/dataset_coverage_report.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-render-images", action="store_true", help="プレースホルダー画像の生成をスキップ(高速だが下流評価には実画像が必要)")
    args = p.parse_args()

    set_seed(args.seed)
    pool = build_candidate_pool()
    selected = greedy_optimize(pool, n_target=args.n_target, seed=args.seed)
    entries = candidates_to_entries(selected)

    if not args.no_render_images:
        render_placeholder_images(entries, seed=args.seed)

    save_entries(entries, args.out)
    report = coverage_report(selected)
    write_json(args.report_out, report)

    print(f"構築エントリ数: {len(entries)} -> {args.out}")
    print(f"被覆レポート -> {args.report_out}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
