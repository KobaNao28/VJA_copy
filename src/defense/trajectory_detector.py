"""
時系列軌跡エンコーディング攻撃(`temporal_trajectory_attack.py`)の検出器。

**核心的な主張**: 単一フレームだけを見る検査(既存のmark_detector/guard_classifier含む
あらゆる静止画ベースの安全フィルタ)は、意味のある軌跡(attack)とランダムな動き/静止
(benign)を**原理的に区別できない**——どちらも個々のフレームは「小さな色付きの点が
1つあるだけ」の画像であり、静止画特徴だけでは同一に見えるためである。
本モジュールはこれを実証する2つのコンポーネントを提供する:

  1. `naive_single_frame_flag()`: 各フレームを独立に検査する最も寛容な基準
     (色付きマーカーの存在有無のみを見る)。意味のある軌跡か無害な動きかを
     一切区別できないことを示すベースライン。
  2. `TrajectoryClassifierMLP`: フレーム間のマーカー座標系列から特徴量
     (正規化座標・総移動距離・正味変位・累積回頭角=曲率)を抽出し、
     系列全体を見て初めて意味のある軌跡(attack)かどうかを判別する検出器。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.attack.visual_instruction_attack import MARK_COLOR_PRESETS
from src.utils.io_utils import read_jsonl, write_json
from src.utils.seed import set_seed

GESTURE_CLASSES = ["static", "random_noise", "arrow_sweep", "circle_then_tap", "zigzag_alert", "underline_sweep"]


def extract_centroid(frame_path: str, tolerance: int = 40) -> tuple[float, float] | None:
    """既知のマーカー色プリセットに近い画素の重心を求める(簡易カラーブロブ検出)。"""
    img = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.int16)
    mask = np.zeros(img.shape[:2], dtype=bool)
    for color in MARK_COLOR_PRESETS.values():
        c = np.array(color, dtype=np.int16)
        dist = np.abs(img - c).sum(axis=-1)
        mask |= dist < tolerance
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def naive_single_frame_flag(frame_path: str) -> bool:
    """最も寛容な単一フレーム基準: 何らかのマーカー色が存在するか否かのみを見る。"""
    return extract_centroid(frame_path) is not None


def sequence_to_feature_vector(frame_paths: list[str], n_frames: int, canvas_size: tuple[int, int]) -> np.ndarray:
    w, h = canvas_size
    centroids = []
    last = (w / 2, h / 2)
    for p in frame_paths:
        c = extract_centroid(p)
        if c is None:
            c = last
        centroids.append(c)
        last = c

    arr = np.array(centroids, dtype=np.float32)
    norm = arr / np.array([w, h], dtype=np.float32)  # 0-1に正規化

    deltas = np.diff(arr, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=1).sum())
    net_displacement = float(np.linalg.norm(arr[-1] - arr[0]))

    curvature = 0.0
    for i in range(1, len(deltas)):
        v1, v2 = deltas[i - 1], deltas[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-3 and n2 > 1e-3:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curvature += float(np.arccos(cos_angle))

    summary = np.array([
        path_length / (w + h),
        net_displacement / (w + h),
        curvature / max(1, len(deltas)),
    ], dtype=np.float32)

    return np.concatenate([norm.flatten(), summary])


class TrajectoryClassifierMLP(nn.Module):
    def __init__(self, n_frames: int, n_classes: int = len(GESTURE_CLASSES)):
        super().__init__()
        in_dim = n_frames * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.semantic_head = nn.Linear(32, 1)     # is_semantic logit
        self.gesture_head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        return self.semantic_head(feat).squeeze(-1), self.gesture_head(feat)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rows = list(read_jsonl(args.data))
    if not rows:
        raise SystemExit(f"データが空です: {args.data}")
    n_frames = rows[0]["n_frames"]
    canvas_size = (640, 480)

    features, semantic_labels, gesture_labels = [], [], []
    for r in rows:
        feat = sequence_to_feature_vector(r["frame_paths"], n_frames, canvas_size)
        features.append(feat)
        semantic_labels.append(int(r["is_semantic"]))
        gesture_labels.append(GESTURE_CLASSES.index(r["gesture_type"]))

    idx = list(range(len(rows)))
    random.Random(args.seed).shuffle(idx)
    split = max(1, int(len(idx) * 0.8))
    train_idx, val_idx = idx[:split], idx[split:] or idx[:1]

    X = torch.tensor(np.stack(features), dtype=torch.float32)
    Y_sem = torch.tensor(semantic_labels, dtype=torch.float32)
    Y_ges = torch.tensor(gesture_labels, dtype=torch.long)

    model = TrajectoryClassifierMLP(n_frames=n_frames)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    def run_epoch(indices: list[int], train_mode: bool) -> dict:
        model.train(train_mode)
        xb, yb_sem, yb_ges = X[indices], Y_sem[indices], Y_ges[indices]
        sem_logits, ges_logits = model(xb)
        loss = bce(sem_logits, yb_sem) + ce(ges_logits, yb_ges)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sem_acc = ((torch.sigmoid(sem_logits) > 0.5).float() == yb_sem).float().mean().item()
        ges_acc = (ges_logits.argmax(dim=-1) == yb_ges).float().mean().item()
        return {"loss": loss.item(), "semantic_acc": sem_acc, "gesture_acc": ges_acc}

    history = []
    best_val_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        tr = run_epoch(train_idx, True)
        with torch.no_grad():
            va = run_epoch(val_idx, False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"[epoch {epoch}] train_loss={tr['loss']:.4f} semantic_acc={tr['semantic_acc']:.3f} gesture_acc={tr['gesture_acc']:.3f} "
            f"| val semantic_acc={va['semantic_acc']:.3f} gesture_acc={va['gesture_acc']:.3f}"
        )
        if va["semantic_acc"] > best_val_acc:
            best_val_acc, best_state = va["semantic_acc"], {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"最良検証semantic_acc: {best_val_acc:.3f} の重みを採用")

    # 比較: 素朴な単一フレーム基準は semantic/benign を区別できるか
    naive_semantic_flag_rate = np.mean([any(naive_single_frame_flag(p) for p in r["frame_paths"]) for r in rows if r["is_semantic"]])
    naive_benign_flag_rate = np.mean([any(naive_single_frame_flag(p) for p in r["frame_paths"]) for r in rows if not r["is_semantic"]])
    print(f"\n[比較] 素朴な単一フレーム基準(マーカー存在の有無のみ):")
    print(f"  意味のある軌跡(attack)での検知率: {naive_semantic_flag_rate:.3f}")
    print(f"  良性な動き(benign)での誤検知率  : {naive_benign_flag_rate:.3f}")
    print(f"  -> 差 = {naive_semantic_flag_rate - naive_benign_flag_rate:.3f} (0に近いほど単一フレーム基準に判別力が無いことを意味する)")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "n_frames": n_frames}, args.save)
        print(f"保存: {args.save}")
    if args.history_out:
        write_json(args.history_out, {
            "history": history,
            "naive_baseline": {
                "semantic_flag_rate": float(naive_semantic_flag_rate),
                "benign_flag_rate": float(naive_benign_flag_rate),
            },
        })


def main() -> None:
    p = argparse.ArgumentParser(description="時系列軌跡エンコーディング攻撃の検出器学習")
    p.add_argument("--data", default="data/sample/trajectories/manifest.jsonl")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="outputs/trajectory_detector.pt")
    p.add_argument("--history-out", default="outputs/trajectory_train_history.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
