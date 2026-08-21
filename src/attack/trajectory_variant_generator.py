"""時系列軌跡攻撃バリアントの一括生成CLI。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.attack.temporal_trajectory_attack import (
    BENIGN_GESTURES,
    SEMANTIC_GESTURES,
    GestureType,
    TrajectorySpec,
    render_frame_sequence,
    save_frame_sequence,
)
from src.attack.visual_instruction_attack import generate_background_pool
from src.utils.io_utils import write_jsonl
from src.utils.seed import set_seed


def generate_grid(
    gestures: list[GestureType],
    colors: list[str],
    n_frames: int,
    n_reps: int,
    out_dir: Path,
    seed: int = 42,
    n_backgrounds: int = 16,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_size = (640, 480)
    background_pool = generate_background_pool(n_backgrounds, canvas_size, seed)

    manifest = []
    idx = 0
    for gesture in gestures:
        for color in colors:
            for rep in range(n_reps):
                spec = TrajectorySpec(
                    canvas_size=canvas_size, gesture_type=gesture, marker_color=color,
                    n_frames=n_frames, seed=seed + idx,
                )
                background = background_pool[idx % n_backgrounds]
                frames, meta = render_frame_sequence(spec, background=background)
                variant_id = f"traj{idx:04d}_{gesture}_{color}"
                frame_paths = save_frame_sequence(frames, out_dir / variant_id)
                manifest.append({
                    "variant_id": variant_id,
                    "frame_paths": frame_paths,
                    "gesture_type": gesture,
                    "marker_color": color,
                    "n_frames": n_frames,
                    "is_semantic": meta["is_semantic"],
                })
                idx += 1
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="時系列軌跡エンコーディング攻撃バリアント一括生成")
    p.add_argument("--gestures", default=",".join(SEMANTIC_GESTURES + BENIGN_GESTURES))
    p.add_argument("--colors", default="red,yellow,cyan")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--n-reps", type=int, default=15, help="各(gesture,color)の組み合わせにつき生成する系列数")
    p.add_argument("--out", default="data/sample/trajectories")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    manifest = generate_grid(
        gestures=args.gestures.split(","),  # type: ignore[arg-type]
        colors=args.colors.split(","),
        n_frames=args.n_frames,
        n_reps=args.n_reps,
        out_dir=out_dir,
        seed=args.seed,
    )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    n_semantic = sum(1 for m in manifest if m["is_semantic"])
    print(f"生成系列数: {len(manifest)} (意味のある軌跡: {n_semantic}, 良性動作: {len(manifest) - n_semantic}) -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
