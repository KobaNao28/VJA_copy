"""
非テキスト視覚指示(VJA本来の攻撃面)のバリアント一括生成CLI。

`variant_generator.py` のマーク版。mark_type × color × stroke_width × n_marks ×
obfuscation_level の直積でシーン+マーク画像を生成し、manifest.jsonl に
target_bboxes(正解の指示対象領域)を含めて保存する。

生成物は `mark_compare_optimize.py`(検知回避×意図伝達精度のパレート最適化)と
`mark_detector.py`(マーク検出器の学習)の入力になる。
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from src.attack.visual_instruction_attack import (
    MarkType,
    ObfuscationLevel,
    VisualInstructionSpec,
    generate_background_pool,
    render_visual_instruction,
)
from src.utils.io_utils import write_jsonl
from src.utils.seed import set_seed


def generate_grid(
    mark_types: list[MarkType],
    colors: list[str],
    stroke_widths: list[int],
    n_marks_list: list[int],
    obfuscation_levels: list[ObfuscationLevel],
    out_dir: Path,
    seed: int = 42,
    n_backgrounds: int = 24,
) -> list[dict]:
    """
    n_backgrounds 枚の背景プールを使い回すことで、mark_detector.py の学習データが
    「背景の見た目の違い」ではなく「マークの有無・種類」に対応した信号になるようにする
    (交絡因子の統制。背景がバリアントごとに完全にユニークだと少量データでは
    過学習し、検証精度が改善しない問題が実験的に確認されたための対策)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    combos = list(itertools.product(mark_types, colors, stroke_widths, n_marks_list, obfuscation_levels))
    background_pool = generate_background_pool(n_backgrounds, seed=seed)

    for idx, (mark_type, color, width, n_marks, level) in enumerate(combos):
        spec = VisualInstructionSpec(
            mark_type=mark_type, mark_color=color, stroke_width=width,
            n_marks=n_marks, obfuscation_level=level, seed=seed + idx,
        )
        background = background_pool[idx % n_backgrounds]
        img, meta = render_visual_instruction(spec, background=background)

        variant_id = f"m{idx:04d}_{mark_type}_{color}_{width}_{n_marks}_{level}"
        variant_dir = out_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        img_path = variant_dir / "image.png"
        img.save(img_path)

        manifest.append(
            {
                "variant_id": variant_id,
                "image_path": str(img_path),
                "mark_type": mark_type,
                "color": color,
                "stroke_width": width,
                "n_marks": n_marks,
                "obfuscation_level": level,
                "target_bboxes": meta["target_bboxes"],
                "carrier_prompt": meta["carrier_prompt"],
            }
        )
    return manifest


def generate_clean_negatives(
    n: int, out_dir: Path, seed: int = 1000,
    reuse_background_pool: list[tuple] | None = None,
) -> list[dict]:
    """
    マークを含まない"クリーンな"シーン画像(guard_classifier/mark_detectorの負例)。
    `reuse_background_pool` を渡すと `generate_grid` と**同一の背景画像**をそのまま
    (マークを描画せず)負例として使う。これにより「同じ背景でマークの有無だけが違う」
    最も情報量の多い対比ペアを作れる。
    """
    from src.attack.visual_instruction_attack import _render_scene_background
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        if reuse_background_pool:
            img, _ = reuse_background_pool[i % len(reuse_background_pool)]
            img = img.copy()
        else:
            rng = random.Random(seed + i)
            img, _ = _render_scene_background((768, 512), rng)
        variant_id = f"clean_{i:04d}"
        variant_dir = out_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        img_path = variant_dir / "image.png"
        img.save(img_path)
        manifest.append({
            "variant_id": variant_id, "image_path": str(img_path),
            "mark_type": "none", "color": "none", "stroke_width": 0,
            "n_marks": 0, "obfuscation_level": "none", "target_bboxes": [],
            "carrier_prompt": "Describe this photo.",
        })
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="非テキスト視覚指示(VJA型)バリアント一括生成")
    p.add_argument("--mark-types", default="arrow,circle,rectangle,x_mark,scribble")
    p.add_argument("--colors", default="red,yellow,cyan,faint_gray")
    p.add_argument("--stroke-widths", default="2,5,9")
    p.add_argument("--n-marks", default="1,2")
    p.add_argument("--obfuscation-levels", default="solid,dashed,dotted,faint,fragmented")
    p.add_argument("--n-clean-negatives", type=int, default=40)
    p.add_argument("--n-backgrounds", type=int, default=24, help="使い回す背景シーンの枚数(過学習対策)")
    p.add_argument("--out", default="data/sample/visual_marks")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)

    manifest = generate_grid(
        mark_types=args.mark_types.split(","),  # type: ignore[arg-type]
        colors=args.colors.split(","),
        stroke_widths=[int(x) for x in args.stroke_widths.split(",")],
        n_marks_list=[int(x) for x in args.n_marks.split(",")],
        obfuscation_levels=args.obfuscation_levels.split(","),  # type: ignore[arg-type]
        out_dir=out_dir,
        seed=args.seed,
        n_backgrounds=args.n_backgrounds,
    )
    # generate_grid と同一シード・同一枚数で再生成すれば決定的に同じ背景プールが得られる
    background_pool = generate_background_pool(args.n_backgrounds, seed=args.seed)
    manifest += generate_clean_negatives(
        args.n_clean_negatives, out_dir, seed=args.seed + 5000, reuse_background_pool=background_pool
    )

    write_jsonl(out_dir / "manifest.jsonl", manifest)
    n_marked = sum(1 for m in manifest if m["mark_type"] != "none")
    print(f"生成バリアント数: {len(manifest)} (マーク付き: {n_marked}, クリーン: {len(manifest) - n_marked}) -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
