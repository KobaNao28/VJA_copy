"""GUI注入攻撃バリアントの一括生成CLI。"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from src.attack.ui_injection_attack import DialogStyle, UIInjectionSpec, _render_base_screenshot, render_ui_injection
from src.utils.io_utils import write_jsonl
from src.utils.seed import set_seed


def generate_background_pool(n: int, size: tuple[int, int], seed: int) -> list:
    return [_render_base_screenshot(size, random.Random(seed + i)) for i in range(n)]


def generate_grid(
    dialog_styles: list[DialogStyle],
    opacities: list[float],
    n_clean: int,
    out_dir: Path,
    seed: int = 42,
    n_backgrounds: int = 20,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    canvas_size = (1024, 640)
    background_pool = generate_background_pool(n_backgrounds, canvas_size, seed)

    idx = 0
    for style, opacity in itertools.product(dialog_styles, opacities):
        for rep in range(3):  # 各組み合わせにつき3枚(ボタン文言・位置のランダム性を反映)
            spec = UIInjectionSpec(canvas_size=canvas_size, dialog_style=style, injected=True,
                                    obfuscation_opacity=opacity, seed=seed + idx)
            background = background_pool[idx % n_backgrounds]
            img, meta = render_ui_injection(spec, background=background)
            variant_id = f"ui{idx:04d}_{style}_op{opacity}"
            variant_dir = out_dir / variant_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            img_path = variant_dir / "image.png"
            img.save(img_path)
            manifest.append({
                "variant_id": variant_id, "image_path": str(img_path),
                "injected": True, "dialog_style": style, "obfuscation_opacity": opacity,
                "dialog_bbox": meta["dialog_bbox"],
            })
            idx += 1

    for i in range(n_clean):
        background = background_pool[i % n_backgrounds]
        variant_id = f"clean_{i:04d}"
        variant_dir = out_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        img_path = variant_dir / "image.png"
        background.save(img_path)
        manifest.append({
            "variant_id": variant_id, "image_path": str(img_path),
            "injected": False, "dialog_style": "none", "obfuscation_opacity": 1.0,
            "dialog_bbox": None,
        })

    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="GUI/エージェント・ハイジャック(偽UI注入)バリアント一括生成")
    p.add_argument("--dialog-styles", default="banner_top,modal_center,toast_bottom_right,inline_button")
    p.add_argument("--opacities", default="1.0,0.7,0.5")
    p.add_argument("--n-clean", type=int, default=None, help="省略時はマーク付きと同数にして被覆を均衡させる")
    p.add_argument("--out", default="data/sample/ui_injection")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    dialog_styles = args.dialog_styles.split(",")
    opacities = [float(x) for x in args.opacities.split(",")]

    n_marked_estimate = len(dialog_styles) * len(opacities) * 3
    n_clean = args.n_clean if args.n_clean is not None else n_marked_estimate

    manifest = generate_grid(dialog_styles, opacities, n_clean, out_dir, seed=args.seed)
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    n_injected = sum(1 for m in manifest if m["injected"])
    print(f"生成バリアント数: {len(manifest)} (注入あり: {n_injected}, クリーン: {len(manifest) - n_injected}) -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
