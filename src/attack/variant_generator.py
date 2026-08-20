"""
バリアント一括生成CLI。

フォント・色・言語・サイズ・図形化レベルの直積(グリッド)でタイポグラフィ画像を生成し、
メタデータ(manifest.jsonl)と共に保存する。生成物は `compare_optimize.py` の入力になる。

言語別サンプルテキストはすべてプレースホルダー(意味を持たない/害のない定型文)であり、
実運用では利用者が `--text-map` でカスタムのテキストマップ(JSON)を差し替える想定。
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from src.attack.shape_obfuscation import ShapeLevel, obfuscate
from src.attack.typography_attack import TypographySpec
from src.utils.io_utils import write_jsonl
from src.utils.seed import set_seed

DEFAULT_TEXT_BY_LANG = {
    "en": "SAMPLE INSTRUCTION PLACEHOLDER TEXT",
    "ja": "サンプル・プレースホルダー・テキスト",
    "zh": "示例占位文本",
    "mixed": "SAMPLE テキスト 示例 PLACEHOLDER",
}

LANG_TO_FONT_STYLE_OVERRIDE = {
    "ja": "cjk",
    "zh": "cjk",
    "mixed": "cjk",
}


def generate_grid(
    fonts: list[str],
    colors: list[str],
    languages: list[str],
    sizes: list[int],
    shape_levels: list[ShapeLevel],
    out_dir: Path,
    text_map: dict[str, str] | None = None,
) -> list[dict]:
    text_map = text_map or DEFAULT_TEXT_BY_LANG
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    combos = list(itertools.product(fonts, colors, languages, sizes, shape_levels))
    for idx, (font, color, lang, size, level) in enumerate(combos):
        font_style = LANG_TO_FONT_STYLE_OVERRIDE.get(lang, font)
        text = text_map.get(lang, DEFAULT_TEXT_BY_LANG["en"])
        spec = TypographySpec(
            text=text,
            font_style=font_style,
            font_size=size,
            color=color,
            language=lang,
        )
        variant_id = f"v{idx:04d}_{font}_{color}_{lang}_{size}_{level}"
        variant_dir = out_dir / variant_id
        img = obfuscate(spec, level, out_dir=variant_dir)
        img_path = variant_dir / "image.png"
        variant_dir.mkdir(parents=True, exist_ok=True)
        img.save(img_path)

        manifest.append(
            {
                "variant_id": variant_id,
                "image_path": str(img_path),
                "font_requested": font,
                "font_style_used": font_style,
                "color": color,
                "language": lang,
                "font_size": size,
                "shape_level": level,
                "text_length": len(text),
            }
        )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="タイポグラフィ攻撃バリアント一括生成")
    p.add_argument("--fonts", default="sans,serif,mono")
    p.add_argument("--colors", default="black,red,gray_low_contrast")
    p.add_argument("--languages", default="en,ja")
    p.add_argument("--sizes", default="24,48,96")
    p.add_argument("--shape-levels", default="raw,outline,filled,fragmented,vector,noise_edge")
    p.add_argument("--out", default="data/sample/variants")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)

    manifest = generate_grid(
        fonts=args.fonts.split(","),
        colors=args.colors.split(","),
        languages=args.languages.split(","),
        sizes=[int(s) for s in args.sizes.split(",")],
        shape_levels=args.shape_levels.split(","),  # type: ignore[arg-type]
        out_dir=out_dir,
    )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    print(f"生成バリアント数: {len(manifest)} -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
