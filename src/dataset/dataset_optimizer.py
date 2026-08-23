"""
VJA/IESBench風データセットの「1から作成する」最適化アルゴリズム(合成生成)。

このモジュールは SAFETY_POLICIES × ATTRIBUTES × EDIT_ACTIONS の直積からプレースホルダー
候補プールを合成生成し、被覆×多様性×難易度バランスを最大化するN件を貪欲選択することで
「架空だが体系的に偏りのない」データセットを1から組み立てる(実在の外部データセットには
一切依存しない)。

公式IESBench等、既に存在するデータセットを"加工"して同様の最適化選抜を行いたい場合は、
本モジュールではなく src/dataset/dataset_adapter.py を使う(実データのentryを候補として
扱う点のみが異なり、選抜アルゴリズム自体は src/dataset/coverage_optimizer.py を共有する)。

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
  貪欲選択の本体(ベクトル化実装)は src/dataset/coverage_optimizer.py::greedy_select() を使う。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.dataset.coverage_optimizer import greedy_select, hash_embedding
from src.dataset.iesbench_schema import ATTRIBUTES, EDIT_ACTIONS, SAFETY_POLICIES, IESBenchEntry
from src.utils.seed import set_seed

_hash_embedding = hash_embedding  # 後方互換のためのエイリアス(このモジュール内で従来使用)


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
    貪欲法による被覆×多様性×難易度バランスの最適化。
    本体は src/dataset/coverage_optimizer.py::greedy_select() (numpyベクトル化実装、
    src/dataset/dataset_adapter.py と共有)に委譲し、ここでは Candidate ->
    (embedding, difficulty, coverage_key, diversity_key) への変換のみを行う。
    """
    embeddings = np.stack([c.embedding for c in pool])  # (P, D)
    difficulties = np.array([c.difficulty for c in pool])
    coverage_keys = [f"{c.category}|{c.action}" for c in pool]
    diversity_keys = [c.attribute for c in pool]

    idx = greedy_select(
        embeddings, difficulties, coverage_keys, diversity_keys, n_target,
        w_coverage=w_coverage, w_diversity=w_diversity, w_difficulty_balance=w_difficulty_balance,
        n_difficulty_bins=n_difficulty_bins, seed=seed,
    )
    return [pool[i] for i in idx]


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
