"""
画像+テキスト Jailbreak (VJA型攻撃) 検知器の学習。

`variant_generator.py` / `compare_optimize.py` が生成する多様な難読化バリアントを
「攻撃的(label=1)」、通常のUI要素的な画像内テキスト(例: 商品ラベル、字幕等を模した
低難読化のもの)を「良性(label=0)」として、軽量なマルチモーダル分類器を学習する。

アーキテクチャ:
  - 画像エンコーダ: 小型CNN(64x64グレースケール入力、事前学習重み不要でCPUでも高速)
  - テキストエンコーダ: OCR抽出テキストのbyte-levelバッグオブワード埋め込み
  - 融合: 特徴量を結合してMLPで2値分類(sigmoid)

本番運用では画像エンコーダをCLIP画像エンコーダ等の事前学習済みモデルに差し替えることを推奨する
(このリポジトリでは依存を増やさずアルゴリズム/学習ループの正しさを検証することを優先している)。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.attack.compare_optimize import ocr_readability  # noqa: F401 (将来の特徴量拡張用)
from src.utils.io_utils import read_jsonl, write_json
from src.utils.seed import set_seed


class TinyImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class TextBoWEncoder(nn.Module):
    def __init__(self, vocab_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.fc = nn.Linear(vocab_dim, out_dim)

    def _hash_bow(self, text: str) -> torch.Tensor:
        import hashlib

        vec = torch.zeros(self.vocab_dim)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.vocab_dim] += 1.0
        return vec

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = torch.stack([self._hash_bow(t) for t in texts])
        return self.fc(batch)


class GuardClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_enc = TinyImageEncoder()
        self.text_enc = TextBoWEncoder()
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        img_feat = self.image_enc(images)
        txt_feat = self.text_enc(texts)
        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return self.head(fused).squeeze(-1)


def load_image_tensor(path: str, size: int = 64) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def build_labeled_dataset(manifest_path: str, ocr_cache: bool = True) -> list[dict]:
    """
    variant manifest から、shape_level と color(低コントラスト)を根拠に
    暫定ラベルを付与する(合成データでの弱教師付けラベリング)。
    実運用では人手レビュー済みラベルに置き換えること。
    """
    rows = list(read_jsonl(manifest_path))
    labeled = []
    for r in rows:
        is_attack_like = r["shape_level"] in {"fragmented", "noise_edge", "vector"} or r["color"] == "gray_low_contrast"
        labeled.append({**r, "label": int(is_attack_like)})
    return labeled


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rows = build_labeled_dataset(args.data)
    if not rows:
        raise SystemExit(f"データが空です: {args.data}")
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split], rows[split:] or rows[:1]

    model = GuardClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(data_rows: list[dict], train_mode: bool) -> tuple[float, float]:
        model.train(train_mode)
        total_loss, correct = 0.0, 0
        for i in range(0, len(data_rows), args.batch_size):
            batch = data_rows[i:i + args.batch_size]
            images = torch.stack([load_image_tensor(r["image_path"]) for r in batch])
            texts = [f"{r['font_requested']} {r['color']} {r['language']} {r['shape_level']}" for r in batch]
            labels = torch.tensor([float(r["label"]) for r in batch])

            logits = model(images, texts)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
        return total_loss / len(data_rows), correct / len(data_rows)

    history = []
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(train_rows, True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_rows, False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"保存: {args.save}")
    if args.history_out:
        write_json(args.history_out, history)


def main() -> None:
    p = argparse.ArgumentParser(description="Jailbreak(VJA型攻撃) guard classifier の学習")
    p.add_argument("--data", default="data/sample/variants/manifest.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="outputs/guard_classifier.pt")
    p.add_argument("--history-out", default="outputs/guard_train_history.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
