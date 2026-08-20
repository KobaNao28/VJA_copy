"""
文字の図形化(shape obfuscation)モジュール。

「文字」というシンボルとしての情報を保ちながら、OCRエンジンや文字ベースの
安全フィルタが「テキスト」として認識しにくい図形表現に変換する度合いを
段階的に制御する。VJAの脅威分析(docs/01, docs/02)に基づき、
以下の難読化レベルを提供する:

  raw        : 通常のフォントレンダリング(typography_attack.render_typography と同じ)
  outline    : グリフのアウトラインのみ(塗りつぶしなし)
  filled     : グリフを塗りつぶしシルエット化(アンチエイリアス除去済み二値マスク)
  fragmented : 塗りつぶしシルエットを小領域に分割し、僅かにずらして再配置(文字の連結性を破壊)
  vector     : 輪郭をSVGパスとして抽出したベクター図形(ラスタ⇄ベクター比較用)
  noise_edge : シルエットの輪郭のみを粒子状の点群で表現(輪郭抽出+間引き)

これにより「文字」を最終的に完全な図形(輪郭・塗りつぶし多角形・点群)として
表現したバリアントを生成し、`compare_optimize.py` でOCR可読性/検知回避性を比較できる。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

from src.attack.typography_attack import TypographySpec, render_typography

ShapeLevel = Literal["raw", "outline", "filled", "fragmented", "vector", "noise_edge"]


def _to_binary_mask(img: Image.Image, threshold: int = 200) -> np.ndarray:
    gray = np.array(img.convert("L"))
    return (gray < threshold).astype(np.uint8) * 255


def _mask_to_image(mask: np.ndarray, fg=(20, 20, 20), bg=(255, 255, 255)) -> Image.Image:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = bg
    out[mask > 0] = fg
    return Image.fromarray(out)


def apply_outline(img: Image.Image) -> Image.Image:
    import cv2

    mask = _to_binary_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (20, 20, 20), thickness=1)
    return Image.fromarray(canvas)


def apply_filled(img: Image.Image) -> Image.Image:
    mask = _to_binary_mask(img)
    return _mask_to_image(mask)


def apply_fragmented(img: Image.Image, n_splits: int = 4, jitter: int = 6, seed: int = 0) -> Image.Image:
    import cv2

    rng = np.random.default_rng(seed)
    mask = _to_binary_mask(img)
    h, w = mask.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw == 0 or ch == 0:
            continue
        band_h = max(1, ch // n_splits)
        for i in range(n_splits):
            y0, y1 = y + i * band_h, min(y + (i + 1) * band_h, y + ch)
            if y1 <= y0:
                continue
            band_mask = np.zeros_like(mask)
            band_mask[y0:y1, x:x + cw] = mask[y0:y1, x:x + cw]
            dx, dy = rng.integers(-jitter, jitter + 1, size=2)
            shifted = np.roll(np.roll(band_mask, dy, axis=0), dx, axis=1)
            canvas[shifted > 0] = (20, 20, 20)
    return Image.fromarray(canvas)


def apply_vector_paths(img: Image.Image, out_svg_path: str | Path) -> str:
    """輪郭をSVGパスとして書き出し、SVGファイルパスを返す(ベクター化)。"""
    import cv2
    import svgwrite

    mask = _to_binary_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape

    out_svg_path = Path(out_svg_path)
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    dwg = svgwrite.Drawing(str(out_svg_path), size=(w, h))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))
    for cnt in contours:
        if len(cnt) < 3:
            continue
        points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        dwg.add(dwg.polygon(points=points, fill="#141414"))
    dwg.save()
    return str(out_svg_path)


def apply_noise_edge(img: Image.Image, keep_ratio: float = 0.35, seed: int = 0) -> Image.Image:
    import cv2

    rng = np.random.default_rng(seed)
    mask = _to_binary_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    canvas = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) == 0:
            continue
        keep = rng.random(len(pts)) < keep_ratio
        for (x, y) in pts[keep]:
            canvas[max(0, y - 1):y + 1, max(0, x - 1):x + 1] = (20, 20, 20)
    return Image.fromarray(canvas)


def obfuscate(spec: TypographySpec, level: ShapeLevel, out_dir: str | Path | None = None) -> Image.Image:
    base = render_typography(spec)
    if level == "raw":
        result = base
    elif level == "outline":
        result = apply_outline(base)
    elif level == "filled":
        result = apply_filled(base)
    elif level == "fragmented":
        result = apply_fragmented(base)
    elif level == "noise_edge":
        result = apply_noise_edge(base)
    elif level == "vector":
        if out_dir is None:
            raise ValueError("vector レベルには out_dir (SVG出力先) が必要です")
        svg_path = Path(out_dir) / "vector.svg"
        apply_vector_paths(base, svg_path)
        result = base.filter(ImageFilter.CONTOUR)  # プレビュー用ラスタ表現
    else:
        raise ValueError(f"未知の shape level: {level}")
    return result


@dataclass
class ObfuscationResult:
    level: ShapeLevel
    image: Image.Image


if __name__ == "__main__":
    spec = TypographySpec(text="SAMPLE PLACEHOLDER TEXT", font_size=64)
    out_dir = Path("outputs/shape_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    for lvl in ("raw", "outline", "filled", "fragmented", "vector", "noise_edge"):
        im = obfuscate(spec, lvl, out_dir=out_dir)
        im.save(out_dir / f"{lvl}.png")
        print(f"saved {out_dir / f'{lvl}.png'}")
