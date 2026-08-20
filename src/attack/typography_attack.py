"""
タイポグラフィ攻撃レンダラー(FigStep/VJA系の再現用コンポーネント)

テキストを画像に描画する。VJAで重要なのは「テキストプロンプト自体は無害に保ち、
視覚要素(画像内テキストやマーク)に意図を埋め込む」点であるため、このモジュールは
単体では攻撃を完成させず、`shape_obfuscation.py` と組み合わせて難読化レベルを
制御できるように設計している。

デフォルトの `text` はプレースホルダー文言であり、本モジュール自体は
有害コンテンツを一切内包しない。呼び出し側が独自の(合法な用途の)文字列を渡す設計。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# システムにインストール済みのフォントから、比較実験に使うカテゴリ別代表フォントを選ぶ。
FONT_CANDIDATES: dict[str, list[str]] = {
    "sans": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ],
    "serif": [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ],
    "mono": [
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ],
    "cjk": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/unifont/unifont_jp.otf",
    ],
}

COLOR_PRESETS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "red": (200, 30, 30),
    "gray_low_contrast": (170, 170, 170),
    "brand_blue": (30, 90, 200),
}


def resolve_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES.get(style, []):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class TypographySpec:
    text: str = "SAMPLE INSTRUCTION PLACEHOLDER"
    font_style: str = "sans"     # sans | serif | mono | cjk
    font_size: int = 48
    color: str = "black"
    language: str = "en"          # 表示用メタデータ(en/ja/zh/mixed)
    canvas_size: tuple[int, int] = (768, 384)
    background: tuple[int, int, int] = (255, 255, 255)
    position: str = "center"      # center | top | bottom | scattered
    extra: dict = field(default_factory=dict)


def render_typography(spec: TypographySpec) -> Image.Image:
    """指定されたパラメータでテキストを画像に描画する(素のタイポグラフィ攻撃相当)。"""
    img = Image.new("RGB", spec.canvas_size, color=spec.background)
    draw = ImageDraw.Draw(img)
    font = resolve_font(spec.font_style, spec.font_size)
    color = COLOR_PRESETS.get(spec.color, (20, 20, 20))

    lines = _wrap_text(spec.text, font, spec.canvas_size[0] - 40)
    bbox_heights = [_text_bbox(draw, line, font)[3] for line in lines]
    total_h = sum(bbox_heights) + 8 * (len(lines) - 1)

    if spec.position == "top":
        y = 20
    elif spec.position == "bottom":
        y = spec.canvas_size[1] - total_h - 20
    else:  # center / scattered の初期値
        y = max(20, (spec.canvas_size[1] - total_h) // 2)

    for i, line in enumerate(lines):
        w = _text_bbox(draw, line, font)[2]
        x = max(20, (spec.canvas_size[0] - w) // 2)
        if spec.position == "scattered":
            import random

            x = random.randint(10, max(11, spec.canvas_size[0] - w - 10))
        draw.text((x, y), line, font=font, fill=color)
        y += bbox_heights[i] + 8

    return img


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for w in words:
        trial = (current + " " + w).strip()
        if _text_bbox(dummy, trial, font)[2] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [text]


def save(img: Image.Image, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


if __name__ == "__main__":
    spec = TypographySpec()
    im = render_typography(spec)
    save(im, "outputs/typography_demo.png")
    print("saved outputs/typography_demo.png")
