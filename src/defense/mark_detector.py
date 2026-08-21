"""
非テキスト視覚指示(矢印・丸囲み・矩形マーキング等)の検出器。

`introspective_defense.py::extract_visual_marks()` はOCRでは拾えない視覚指示を
検出するための最初のヒューリスティック(輪郭の頂点数で矢印らしさを判定)だったが、
非常に粗く、`fragmented`/`dotted` のような難読化バリアントには対応できない。

本モジュールは `mark_variant_generator.py` が生成したマーク付き/クリーン画像で
軽量CNN分類器を学習し、
  (a) 画像に何らかの視覚指示マークが含まれるか(2値)
  (b) どの種類のマークか(5クラス: arrow/circle/rectangle/x_mark/scribble)
を推定できるようにする。`train_guard_classifier.py` のテキストエンコーダに依存しない
**画像単体の特徴量のみ**で判定する点が重要(VJAはテキストプロンプト側に一切シグナルを
残さないため、テキスト特徴に頼る既存のguard_classifierだけでは原理的に検知できない)。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.attack.visual_instruction_attack import MarkType
from src.utils.io_utils import read_jsonl, write_json
from src.utils.seed import set_seed

MARK_CLASSES: list[str] = ["none", "arrow", "circle", "rectangle", "x_mark", "scribble"]


def load_image_tensor_rgb(path: str, size: int = 64) -> torch.Tensor:
    """
    マークは色(赤・黄・シアン等)が主要な識別信号のため、train_guard_classifier.py の
    グレースケール読み込みではなく **RGB 3チャンネル** で読み込む。
    (実験的に確認: グレースケール化するとpresence_accが多数派クラス精度から
     全く改善しない=色情報の消失で検知不能になることを確認した)
    """
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)


class MarkDetectorCNN(nn.Module):
    """RGB画像のみを入力とする軽量CNN。has_mark(2値)とmark_type(多クラス)を同時に予測する。"""

    def __init__(self, n_classes: int = len(MARK_CLASSES)):
        super().__init__()
        # マークは画像全体に対してごく少数の色付き細線ピクセルとしてしか現れないスパースな信号のため、
        # AdaptiveAvgPool2d(平均)ではなく AdaptiveMaxPool2d(最大値)を使う。
        # 実験的に確認: AvgPool+BatchNorm無しの構成では presence_acc が完全にチャンスレベル(0.50)から
        # 一切改善しなかった(スパースなマーク信号が空間平均で消えてしまうため)。
        # BatchNorm2dを各層に追加し学習を安定化させた上で MaxPool に切り替えたところ、
        # 小規模データでの過学習テストで直ちに100%まで収束することを確認済み。
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.presence_head = nn.Linear(64, 1)          # has_mark logit
        self.type_head = nn.Linear(64, n_classes)       # mark_type logits(has_mark=Falseの場合は"none"が正解)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(images).flatten(1)
        return self.presence_head(feat).squeeze(-1), self.type_head(feat)


def build_labeled_dataset(manifest_path: str) -> list[dict]:
    rows = list(read_jsonl(manifest_path))
    for r in rows:
        r["has_mark"] = int(r["mark_type"] != "none")
        r["type_label"] = MARK_CLASSES.index(r["mark_type"]) if r["mark_type"] in MARK_CLASSES else 0
    return rows


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rows = build_labeled_dataset(args.data)
    if not rows:
        raise SystemExit(f"データが空です: {args.data}")
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split], rows[split:] or rows[:1]

    model = MarkDetectorCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    def run_epoch(data_rows: list[dict], train_mode: bool) -> dict:
        model.train(train_mode)
        total_loss, correct_presence, correct_type = 0.0, 0, 0
        for i in range(0, len(data_rows), args.batch_size):
            batch = data_rows[i:i + args.batch_size]
            images = torch.stack([load_image_tensor_rgb(r["image_path"]) for r in batch])
            presence_labels = torch.tensor([float(r["has_mark"]) for r in batch])
            type_labels = torch.tensor([r["type_label"] for r in batch], dtype=torch.long)

            presence_logits, type_logits = model(images)
            loss = bce(presence_logits, presence_labels) + ce(type_logits, type_labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch)
            correct_presence += ((torch.sigmoid(presence_logits) > 0.5).float() == presence_labels).sum().item()
            correct_type += (type_logits.argmax(dim=-1) == type_labels).sum().item()

        n = len(data_rows)
        return {"loss": total_loss / n, "presence_acc": correct_presence / n, "type_acc": correct_type / n}

    history = []
    best_val_presence_acc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        tr = run_epoch(train_rows, True)
        with torch.no_grad():
            va = run_epoch(val_rows, False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"[epoch {epoch}] train_loss={tr['loss']:.4f} presence_acc={tr['presence_acc']:.3f} type_acc={tr['type_acc']:.3f} "
            f"| val_loss={va['loss']:.4f} presence_acc={va['presence_acc']:.3f} type_acc={va['type_acc']:.3f}"
        )
        # BatchNorm+小規模データのため検証損失が一部エポックで不安定にスパイクすることがある。
        # 最終エポックではなく検証presence_accが最良のエポックの重みを保存する。
        if va["presence_acc"] > best_val_presence_acc:
            best_val_presence_acc = va["presence_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"最良検証presence_acc: {best_val_presence_acc:.3f} の重みを採用")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"保存: {args.save}")
    if args.history_out:
        write_json(args.history_out, history)


def detect_marks(model: MarkDetectorCNN, image_path: str) -> dict:
    """introspective_defense.extract_visual_marks() を置き換える推論API。"""
    model.eval()
    with torch.no_grad():
        img = load_image_tensor_rgb(image_path).unsqueeze(0)
        presence_logit, type_logits = model(img)
        has_mark = torch.sigmoid(presence_logit).item() > 0.5
        mark_type = MARK_CLASSES[int(type_logits.argmax(dim=-1).item())]
        confidence = float(torch.softmax(type_logits, dim=-1).max().item())
    return {"has_mark": has_mark, "mark_type": mark_type if has_mark else "none", "confidence": round(confidence, 4)}


def main() -> None:
    p = argparse.ArgumentParser(description="非テキスト視覚指示マークの検出器学習")
    p.add_argument("--data", default="data/sample/visual_marks/manifest.jsonl")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="outputs/mark_detector.pt")
    p.add_argument("--history-out", default="outputs/mark_detector_train_history.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
