"""
GUI注入攻撃(偽ダイアログ/ボタン)の検出器。

エージェントが行動判断に使うスクリーンショットに、正規のUIフローに存在しない
矩形ダイアログ様の要素が重畳されていないかを検出する。`mark_detector.py` で得た
知見(RGB入力の必要性、AdaptiveMaxPool+BatchNormでのスパース信号検出)を踏襲する
軽量CNN。ただしダイアログはマークより占有面積が大きいスパースでない信号のため、
AvgPoolでも学習しうるが、一貫性のため同じ安定した構成を採用する。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from src.defense.mark_detector import load_image_tensor_rgb
from src.utils.io_utils import read_jsonl, write_json
from src.utils.seed import set_seed


class UIInjectionDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images).flatten(1)
        return self.head(feat).squeeze(-1)


def detect_ui_injection(model: UIInjectionDetectorCNN, image_path: str) -> dict:
    model.eval()
    with torch.no_grad():
        img = load_image_tensor_rgb(image_path).unsqueeze(0)
        logit = model(img)
        prob = torch.sigmoid(logit).item()
    return {"injected": prob > 0.5, "confidence": round(prob, 4)}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rows = list(read_jsonl(args.data))
    if not rows:
        raise SystemExit(f"データが空です: {args.data}")
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split], rows[split:] or rows[:1]

    model = UIInjectionDetectorCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    def run_epoch(data_rows: list[dict], train_mode: bool) -> dict:
        model.train(train_mode)
        total_loss, correct = 0.0, 0
        for i in range(0, len(data_rows), args.batch_size):
            batch = data_rows[i:i + args.batch_size]
            images = torch.stack([load_image_tensor_rgb(r["image_path"]) for r in batch])
            labels = torch.tensor([float(r["injected"]) for r in batch])
            logits = model(images)
            loss = bce(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch)
            correct += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
        n = len(data_rows)
        return {"loss": total_loss / n, "acc": correct / n}

    history = []
    best_val_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        tr = run_epoch(train_rows, True)
        with torch.no_grad():
            va = run_epoch(val_rows, False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"[epoch {epoch}] train_loss={tr['loss']:.4f} acc={tr['acc']:.3f} | val_loss={va['loss']:.4f} acc={va['acc']:.3f}")
        if va["acc"] > best_val_acc:
            best_val_acc, best_state = va["acc"], {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"最良検証acc: {best_val_acc:.3f} の重みを採用")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"保存: {args.save}")
    if args.history_out:
        write_json(args.history_out, history)


def main() -> None:
    p = argparse.ArgumentParser(description="GUI注入(偽ダイアログ)検出器の学習")
    p.add_argument("--data", default="data/sample/ui_injection/manifest.jsonl")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="outputs/ui_injection_detector.pt")
    p.add_argument("--history-out", default="outputs/ui_injection_train_history.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
