"""
Attack Immune Memory Defense(免疫記憶型防御)。

## アイデア

生体の獲得免疫系は、一度遭遇した病原体の特徴を「記憶細胞」として保持し、
再侵入時により速く強く反応する。また記憶は時間とともに減衰し、
同じ病原体に繰り返し曝露されると増強(ブースター効果)される。

本モジュールはこの比喩をJailbreak防御に応用する:

  1. `adaptive_attack_optimizer.py`(closed-loop red teaming)が発見した
     「guard classifierの検知をすり抜けた」攻撃構成を「記憶細胞」として
     `AttackMemoryBank` に登録する。
  2. 新規リクエストが来るたびに、その特徴ベクトルを記憶細胞群と比較する
     (Layer 0: 学習済みモデルの推論より軽量な高速事前フィルタ)。
  3. 類似度は時間減衰する(古い記憶は薄れる = 概念ドリフトへの適応、
     恒久的なブロックリスト化による誤検知の蓄積を防ぐ)。
  4. 同一カテゴリの記憶細胞への「再曝露」(ヒット)が閾値を超えたら、
     "booster_retrain_needed" フラグを立て、guard_classifierの再学習を促す
     (継続的なキャンペーン攻撃の検知 = 一過性の誤検知と区別する)。

`unified_defense_pipeline.py` に Layer 0 として追加することで、
「学習済み防御に対する適応攻撃 → 記憶 → 高速検知 → 再学習トリガー」という
closed-loopを完成させる(docs/02_attack_enhancement_proposals.md のレッドチーム
運用フローの自動化版)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.defense.train_guard_classifier import GuardClassifier, load_image_tensor


def embed(model: GuardClassifier, image_path: str, text: str) -> np.ndarray:
    """GuardClassifierのエンコーダ部分を流用し、分類ヘッド手前の特徴量を埋め込みとして使う。"""
    model.eval()
    with torch.no_grad():
        img = load_image_tensor(image_path).unsqueeze(0)
        img_feat = model.image_enc(img)
        txt_feat = model.text_enc([text])
        fused = torch.cat([img_feat, txt_feat], dim=-1).squeeze(0).numpy()
    norm = np.linalg.norm(fused)
    return fused / norm if norm > 0 else fused


@dataclass
class MemoryEntry:
    embedding: np.ndarray
    category: str
    created_at: float
    hit_count: int = 0
    last_hit_at: float = 0.0
    source: str = "adaptive_attack_optimizer"


@dataclass
class MemoryMatch:
    matched: bool
    best_similarity: float = 0.0
    category: str | None = None
    booster_retrain_needed: bool = False


@dataclass
class AttackMemoryBank:
    decay_half_life_seconds: float = 7 * 24 * 3600.0  # 既定: 1週間で類似度の重みが半減
    similarity_threshold: float = 0.85
    booster_hit_threshold: int = 5
    booster_window_seconds: float = 24 * 3600.0        # 直近24時間内のヒットを「キャンペーン」とみなす
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, embedding: np.ndarray, category: str, now: float | None = None,
            source: str = "adaptive_attack_optimizer") -> None:
        now = now if now is not None else time.time()
        self.entries.append(MemoryEntry(embedding=embedding, category=category, created_at=now, source=source))

    def add_from_optimizer_history(self, history: list[dict], embed_fn, image_dir, now: float | None = None) -> int:
        """adaptive_attack_optimizer.hill_climb() の history から「採用された(accepted)」変異を記憶に登録する。"""
        n_added = 0
        for h in history:
            if not h.get("accepted"):
                continue
            img_path = image_dir / f"step_{h['step']:04d}" / "image.png"
            if not img_path.exists():
                continue
            prompt_text = f"{h['font_style']} {h['color']} {h['language']} {h['shape_level']}"
            emb = embed_fn(str(img_path), prompt_text)
            self.add(emb, category="adaptive_evasion", now=now)
            n_added += 1
        return n_added

    def _decay_weight(self, entry: MemoryEntry, now: float) -> float:
        age = max(0.0, now - entry.created_at)
        return 0.5 ** (age / self.decay_half_life_seconds)

    def query(self, embedding: np.ndarray, now: float | None = None) -> MemoryMatch:
        now = now if now is not None else time.time()
        if not self.entries:
            return MemoryMatch(matched=False)

        best_sim, best_entry = -1.0, None
        for entry in self.entries:
            raw_sim = float(np.dot(embedding, entry.embedding))
            decayed_sim = raw_sim * self._decay_weight(entry, now)
            if decayed_sim > best_sim:
                best_sim, best_entry = decayed_sim, entry

        matched = best_sim >= self.similarity_threshold
        booster_needed = False
        if matched and best_entry is not None:
            best_entry.hit_count += 1
            best_entry.last_hit_at = now
            recent_hits = sum(
                1 for e in self.entries
                if e.category == best_entry.category and now - e.last_hit_at <= self.booster_window_seconds and e.hit_count > 0
            )
            booster_needed = recent_hits >= self.booster_hit_threshold

        return MemoryMatch(
            matched=matched,
            best_similarity=round(best_sim, 4),
            category=best_entry.category if matched and best_entry else None,
            booster_retrain_needed=booster_needed,
        )

    def prune(self, now: float | None = None, min_weight: float = 0.01) -> int:
        """減衰しきった記憶を削除する(免疫記憶の自然な忘却)。削除件数を返す。"""
        now = now if now is not None else time.time()
        before = len(self.entries)
        self.entries = [e for e in self.entries if self._decay_weight(e, now) >= min_weight]
        return before - len(self.entries)

    def stats(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        by_category: dict[str, int] = {}
        for e in self.entries:
            by_category[e.category] = by_category.get(e.category, 0) + 1
        return {
            "n_entries": len(self.entries),
            "by_category": by_category,
            "mean_decay_weight": round(float(np.mean([self._decay_weight(e, now) for e in self.entries])), 4) if self.entries else 0.0,
        }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.attack.adaptive_attack_optimizer import hill_climb

    p = argparse.ArgumentParser(description="Attack Immune Memory Defense のデモ実行")
    p.add_argument("--guard-ckpt", default="outputs/guard_classifier.pt")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--attack-dir", default="outputs/adaptive_attack")
    args = p.parse_args()

    if not Path(args.guard_ckpt).exists():
        raise SystemExit(f"guard classifierの重みが見つかりません: {args.guard_ckpt}")

    print("Step 1: closed-loop red teamingで検知回避パターンを探索...")
    result = hill_climb(args.guard_ckpt, n_steps=args.n_steps, out_dir=args.attack_dir)

    guard_model = GuardClassifier()
    guard_model.load_state_dict(torch.load(args.guard_ckpt, map_location="cpu"))

    bank = AttackMemoryBank()
    n_added = bank.add_from_optimizer_history(
        result["history"], embed_fn=lambda img, txt: embed(guard_model, img, txt), image_dir=Path(args.attack_dir)
    )
    print(f"Step 2: {n_added}件の検知回避パターンを免疫記憶に登録")
    print(f"  記憶バンク統計: {bank.stats()}")

    # 最終構成に近い新規リクエストを模擬し、記憶が反応するか確認
    final_cfg = result["final_config"]
    final_img = Path(args.attack_dir) / f"step_{args.n_steps:04d}" / "image.png"
    if final_img.exists():
        prompt_text = f"{final_cfg['font_style']} {final_cfg['color']} {final_cfg['language']} {final_cfg['shape_level']}"
        query_emb = embed(guard_model, str(final_img), prompt_text)
        match = bank.query(query_emb)
        print(f"Step 3: 類似の新規リクエストに対する照合結果: {match}")
