"""
VJA/IESBench風データセットの「作成最適化アルゴリズム」。

目的: 限られたアノテーション予算 N の中で、
  (1) 安全ポリシー×属性×編集アクションの組み合わせ被覆を最大化し
  (2) 意味的な多様性(似たようなサンプルばかりにならない)を確保し
  (3) 難易度(検知回避しやすさ = compare_optimize.py の stealth_score 等)の分布を
      均一に近づける(簡単な事例ばかり/難しい事例ばかりに偏らせない)
ことで、少ないサンプル数でも学習・評価の両方に有効なデータセットを構築する。

アルゴリズム: 劣モジュラ関数の貪欲最大化 (greedy submodular maximization)
  F(S) = w_cov * Coverage(S) + w_div * Diversity(S) + w_diff * DifficultyBalance(S)
  Coverage, Diversity(facility location)は劣モジュラであることが知られており、
  貪欲法は (1 - 1/e) ≈ 63% の近似保証を持つ (Nemhauser, Wolsey and Fisher, 1978)。
  DifficultyBalance項は貪欲法の各ステップでヒストグラムギャップを埋める形の
  ボーナスとして加える(理論的な劣モジュラ保証はないが実用上有効)。
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.dataset.iesbench_schema import ATTRIBUTES, EDIT_ACTIONS, SAFETY_POLICIES, IESBenchEntry
from src.utils.seed import set_seed


def _hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    軽量な特徴ベクトル化(hashing trick によるBoW的埋め込み)。
    本番運用では CLIP/text-encoder による意味埋め込みに差し替えることを想定した
    プレースホルダー実装(依存を増やさずアルゴリズムのロジックを検証するため)。
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in text.lower().replace("_", " ").split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class Candidate:
    category: str
    attribute: str
    action: str
    difficulty: float
    embedding: np.ndarray = field(repr=False)

    @property
    def key(self) -> str:
        return f"{self.category}|{self.attribute}|{self.action}"


def build_candidate_pool(
    categories: list[str] | None = None,
    attributes: list[str] | None = None,
    actions: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Candidate]:
    categories = categories or SAFETY_POLICIES
    attributes = attributes or ATTRIBUTES
    actions = actions or EDIT_ACTIONS
    rng = rng or np.random.default_rng(0)

    pool = []
    for cat, attr, act in product(categories, attributes, actions):
        text = f"{cat} {attr} {act}"
        # 難易度は「カテゴリの機微さ」「アクションの複雑さ」から擬似的に決定
        # (実運用では compare_optimize.py の stealth_score や過去のASR実測値に置き換える)
        base_difficulty = rng.uniform(0.2, 0.8)
        pool.append(
            Candidate(
                category=cat,
                attribute=attr,
                action=act,
                difficulty=round(float(base_difficulty), 3),
                embedding=_hash_embedding(text),
            )
        )
    return pool


def _coverage_key(c: Candidate) -> tuple:
    return (c.category, c.action)


def greedy_optimize(
    pool: list[Candidate],
    n_target: int,
    w_coverage: float = 1.0,
    w_diversity: float = 1.0,
    w_difficulty_balance: float = 0.5,
    n_difficulty_bins: int = 3,
    seed: int = 42,
) -> list[Candidate]:
    """
    貪欲法による被覆×多様性×難易度バランスの最適化(numpyベクトル化実装)。

    素朴な実装(各ステップで残り候補それぞれについて既選択集合全件との類似度を
    Pythonループで計算する)は O(n_target^2 * pool_size) となり、
    候補プールが大きい(例: 15カテゴリ×116属性×9アクション ≈ 15,660件)場合に
    数分〜数十分かかってしまう。ここでは facility location の標準的な高速化トリックである
    「各候補が保持する『既選択集合との最大類似度』を、新規選択1件との類似度だけで
    差分更新する」方式を用い、計算量を O(n_target * pool_size) に落とす。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))  # 同点時のタイブレークをランダム化
    pool_ordered = [pool[i] for i in order]

    embeddings = np.stack([c.embedding for c in pool_ordered])  # (P, D)
    difficulties = np.array([c.difficulty for c in pool_ordered])
    diff_bins = np.minimum(n_difficulty_bins - 1, (difficulties * n_difficulty_bins).astype(int))

    # 被覆判定を整数コードに変換し、以降は完全に配列演算だけで判定する(Pythonループ回避)。
    pair_keys = [f"{c.category}|{c.action}" for c in pool_ordered]
    attr_keys = [c.attribute for c in pool_ordered]
    _, pair_codes = np.unique(pair_keys, return_inverse=True)
    _, attr_codes = np.unique(attr_keys, return_inverse=True)
    pair_covered = np.zeros(pair_codes.max() + 1, dtype=bool)
    attr_covered = np.zeros(attr_codes.max() + 1, dtype=bool)

    n = len(pool_ordered)
    available = np.ones(n, dtype=bool)
    max_sim_to_selected = np.zeros(n)          # facility location: 既選択集合との最大類似度
    difficulty_hist = np.zeros(n_difficulty_bins)

    selected_idx: list[int] = []
    n_target = min(n_target, n)

    for _ in range(n_target):
        cov_gain = (~pair_covered[pair_codes]).astype(float) + (~attr_covered[attr_codes]).astype(float)
        div_gain = 1.0 - max_sim_to_selected

        total_selected = difficulty_hist.sum()
        fill_ratio = difficulty_hist[diff_bins] / total_selected if total_selected > 0 else np.zeros(n)
        diff_gain = 1.0 - fill_ratio

        score = w_coverage * cov_gain + w_diversity * div_gain + w_difficulty_balance * diff_gain
        score = np.where(available, score, -np.inf)

        best_idx = int(np.argmax(score))
        selected_idx.append(best_idx)
        available[best_idx] = False

        pair_covered[pair_codes[best_idx]] = True
        attr_covered[attr_codes[best_idx]] = True
        difficulty_hist[diff_bins[best_idx]] += 1

        # facility location の差分更新: 新規選択との類似度で max_sim を更新(ベクトル化)
        sims_to_new = embeddings @ embeddings[best_idx]
        np.maximum(max_sim_to_selected, sims_to_new, out=max_sim_to_selected)

    return [pool_ordered[i] for i in selected_idx]


def coverage_report(selected: list[Candidate]) -> dict:
    pairs = {_coverage_key(c) for c in selected}
    attrs = {c.attribute for c in selected}
    cats = {c.category for c in selected}
    actions = {c.action for c in selected}
    return {
        "n_selected": len(selected),
        "category_action_pairs_covered": len(pairs),
        "category_action_pairs_total": len(SAFETY_POLICIES) * len(EDIT_ACTIONS),
        "categories_covered": len(cats),
        "categories_total": len(SAFETY_POLICIES),
        "attributes_covered": len(attrs),
        "attributes_total": len(ATTRIBUTES),
        "actions_covered": len(actions),
        "actions_total": len(EDIT_ACTIONS),
    }


def candidates_to_entries(selected: list[Candidate], id_prefix: str = "synth") -> list[IESBenchEntry]:
    entries = []
    for i, c in enumerate(selected):
        image_id = f"{id_prefix}_{i:05d}"
        question = (
            f"[PLACEHOLDER] Edit the '{c.attribute}' via '{c.action}' "
            f"(policy-category under review: {c.category}). "
            f"実運用では安全レビュー済みの具体的な指示文に置き換えること。"
        )
        rewrite = f"[PLACEHOLDER rewrite] Perform a {c.action} operation on the {c.attribute}."
        entries.append(
            IESBenchEntry(
                image_id=image_id,
                image_path=f"data/sample/images/{image_id}.png",
                question=question,
                attributes=[c.attribute],
                action=c.action,
                category=c.category,
                rewrite=rewrite,
                extra={"difficulty": c.difficulty},
            )
        )
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description="被覆×多様性×難易度バランスによるデータセット最適化")
    p.add_argument("--n-target", type=int, default=200)
    p.add_argument("--w-coverage", type=float, default=1.0)
    p.add_argument("--w-diversity", type=float, default=1.0)
    p.add_argument("--w-difficulty-balance", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    pool = build_candidate_pool()
    print(f"候補プールサイズ: {len(pool)}")

    selected = greedy_optimize(
        pool,
        n_target=args.n_target,
        w_coverage=args.w_coverage,
        w_diversity=args.w_diversity,
        w_difficulty_balance=args.w_difficulty_balance,
        seed=args.seed,
    )
    report = coverage_report(selected)
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
