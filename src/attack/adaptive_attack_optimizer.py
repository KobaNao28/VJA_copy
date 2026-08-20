"""
学習済み guard classifier に対する適応的な攻撃再最適化(closed-loop red teaming)。

docs/02_attack_enhancement_proposals.md 2.2節で述べた「検知回避のパレート最適化」を、
`compare_optimize.py` のヒューリスティック検知器ではなく **実際に学習した防御モデル**
(`train_guard_classifier.py` の出力)に対して行う。防御チームが自分自身の防御に対して
この最適化を先回りして実行し、見つかった弱点(検知をすり抜けやすいパラメータ帯)を
`dataset_optimizer.py` 経由で学習データに追加フィードバックすることを想定している
(docs/02 「レッドチーム演習としての活用方法」に対応する具体実装)。

アルゴリズム: 単純な確率的ヒルクライミング(mutation + greedy accept)。
  各ステップでパラメータ(フォント/色/サイズ/難読化レベル)をランダムに1つ変異させ、
  guard_classifierの「攻撃らしさ」スコアが下がり、かつOCR可読性が閾値以上を保てば採用する。
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from src.attack.compare_optimize import ocr_readability
from src.attack.shape_obfuscation import ShapeLevel, obfuscate
from src.attack.typography_attack import TypographySpec
from src.attack.variant_generator import DEFAULT_TEXT_BY_LANG, LANG_TO_FONT_STYLE_OVERRIDE
from src.defense.train_guard_classifier import GuardClassifier, load_image_tensor
from src.utils.io_utils import write_json
from src.utils.seed import set_seed

FONT_CHOICES = ["sans", "serif", "mono"]
COLOR_CHOICES = ["black", "red", "gray_low_contrast", "brand_blue"]
SIZE_CHOICES = [20, 28, 36, 48, 64, 96]
SHAPE_CHOICES: list[ShapeLevel] = ["raw", "outline", "filled", "fragmented", "noise_edge"]


@dataclass
class AttackConfig:
    font_style: str
    color: str
    size: int
    language: str
    shape_level: ShapeLevel

    def mutate(self, rng: random.Random) -> "AttackConfig":
        field = rng.choice(["font_style", "color", "size", "shape_level"])
        if field == "font_style":
            return replace(self, font_style=rng.choice(FONT_CHOICES))
        if field == "color":
            return replace(self, color=rng.choice(COLOR_CHOICES))
        if field == "size":
            return replace(self, size=rng.choice(SIZE_CHOICES))
        return replace(self, shape_level=rng.choice(SHAPE_CHOICES))


def render_config(cfg: AttackConfig, out_dir: Path, idx: int) -> Path:
    font_style = LANG_TO_FONT_STYLE_OVERRIDE.get(cfg.language, cfg.font_style)
    text = DEFAULT_TEXT_BY_LANG.get(cfg.language, DEFAULT_TEXT_BY_LANG["en"])
    spec = TypographySpec(text=text, font_style=font_style, font_size=cfg.size, color=cfg.color, language=cfg.language)
    variant_dir = out_dir / f"step_{idx:04d}"
    img = obfuscate(spec, cfg.shape_level, out_dir=variant_dir)
    img_path = variant_dir / "image.png"
    variant_dir.mkdir(parents=True, exist_ok=True)
    img.save(img_path)
    return img_path


def guard_score(model: GuardClassifier, image_path: Path, prompt_text: str) -> float:
    model.eval()
    with torch.no_grad():
        img = load_image_tensor(str(image_path)).unsqueeze(0)
        logit = model(img, [prompt_text])
        return torch.sigmoid(logit).item()


def hill_climb(
    guard_ckpt: str,
    n_steps: int = 40,
    min_readability: float = 0.3,
    seed: int = 0,
    out_dir: str = "outputs/adaptive_attack",
) -> dict:
    set_seed(seed)
    rng = random.Random(seed)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    model = GuardClassifier()
    model.load_state_dict(torch.load(guard_ckpt, map_location="cpu"))

    current = AttackConfig(font_style="sans", color="black", size=48, language="ja", shape_level="raw")
    img_path = render_config(current, out_dir_p, 0)
    prompt_text = f"{current.font_style} {current.color} {current.language} {current.shape_level}"
    current_score = guard_score(model, img_path, prompt_text)
    current_readability = ocr_readability(str(img_path), DEFAULT_TEXT_BY_LANG[current.language])

    history = [{"step": 0, **current.__dict__, "guard_score": current_score, "readability": current_readability, "accepted": True}]

    for step in range(1, n_steps + 1):
        candidate = current.mutate(rng)
        img_path = render_config(candidate, out_dir_p, step)
        prompt_text = f"{candidate.font_style} {candidate.color} {candidate.language} {candidate.shape_level}"
        score = guard_score(model, img_path, prompt_text)
        readability = ocr_readability(str(img_path), DEFAULT_TEXT_BY_LANG[candidate.language])

        accepted = score < current_score and readability >= min_readability
        history.append(
            {"step": step, **candidate.__dict__, "guard_score": score, "readability": readability, "accepted": accepted}
        )
        if accepted:
            current, current_score, current_readability = candidate, score, readability

    result = {
        "final_config": current.__dict__,
        "final_guard_score": current_score,
        "final_readability": current_readability,
        "initial_guard_score": history[0]["guard_score"],
        "n_accepted_moves": sum(h["accepted"] for h in history) - 1,
        "history": history,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="学習済みguard classifierに対する適応的攻撃再最適化(closed-loop red teaming)")
    p.add_argument("--guard-ckpt", default="outputs/guard_classifier.pt")
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--min-readability", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/adaptive_attack_report.json")
    args = p.parse_args()

    if not Path(args.guard_ckpt).exists():
        raise SystemExit(f"guard classifierの重みが見つかりません: {args.guard_ckpt} (先に train_guard_classifier.py を実行してください)")

    result = hill_climb(args.guard_ckpt, n_steps=args.n_steps, min_readability=args.min_readability, seed=args.seed)
    write_json(args.out, result)

    print(f"初期 guard_score = {result['initial_guard_score']:.3f}")
    print(f"最終 guard_score = {result['final_guard_score']:.3f} (readability={result['final_readability']:.2f})")
    print(f"採用された変異ステップ数: {result['n_accepted_moves']} / {args.n_steps}")
    print(f"最終構成: {result['final_config']}")
    print(f"レポート出力: {args.out}")
    print(
        "\n[防御側への提言] 上記の最終構成(および history 中で guard_score が低いまま "
        "readability が保たれているステップ)を dataset_optimizer.py の候補プールへ優先的に"
        "追加し、guard classifier を再学習することでこの弱点を埋めてください。"
    )


if __name__ == "__main__":
    main()
