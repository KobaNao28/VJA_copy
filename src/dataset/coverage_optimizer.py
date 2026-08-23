"""
被覆(coverage)×多様性(diversity, facility location)×難易度バランスの
貪欲法(greedy submodular maximization)を行う汎用エンジン。

src/dataset/dataset_optimizer.py(「1から作成」: SAFETY_POLICIES/EDIT_ACTIONSの直積から
プレースホルダー候補を合成生成する)と src/dataset/dataset_adapter.py(「既存データセットから
加工」: IESBench等の実データをそのまま候補として扱う)の両方が本モジュールの
greedy_select() を共有する。アルゴリズムの本体は両者で共通であり、「候補プールの作り方
(合成生成 or 実データ読み込み)」だけが異なるため、コアロジックをここに集約して重複を避ける。

アルゴリズム: F(S) = w_cov * Coverage(S) + w_div * Diversity(S) + w_diff * DifficultyBalance(S)
Coverage/Diversity(facility location)は劣モジュラであることが知られており、貪欲法は
(1 - 1/e) ≈ 63% の近似保証を持つ (Nemhauser, Wolsey and Fisher, 1978)。
"""
from __future__ import annotations

import hashlib

import numpy as np


def hash_embedding(text: str, dim: int = 64) -> np.ndarray:
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


def greedy_select(
    embeddings: np.ndarray,
    difficulties: np.ndarray,
    coverage_keys: list[str],
    diversity_keys: list[str],
    n_target: int,
    w_coverage: float = 1.0,
    w_diversity: float = 1.0,
    w_difficulty_balance: float = 0.5,
    n_difficulty_bins: int = 3,
    seed: int = 42,
) -> list[int]:
    """
    候補プール(embeddings[i]が候補iの意味ベクトル、coverage_keys[i]が主被覆軸(例:
    'category|action')、diversity_keys[i]が副被覆軸(例: attribute))から、被覆×多様性×
    難易度バランスを最大化する n_target 件を貪欲選択し、元の(引数として渡した順序での)
    インデックス列を返す。

    素朴な実装(各ステップで残り候補それぞれについて既選択集合全件との類似度をPythonループで
    計算する)はO(n_target^2 * pool_size)となり候補プールが大きい場合に非常に遅い。ここでは
    facility locationの標準的な高速化トリックである「各候補が保持する『既選択集合との
    最大類似度』を、新規選択1件との類似度だけで差分更新する」方式を用い、
    計算量をO(n_target * pool_size)に落とす。
    """
    n_total = len(coverage_keys)
    if n_total == 0:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)  # 同点時のタイブレークをランダム化
    emb_o = embeddings[order]
    diff_o = difficulties[order]
    cov_keys_o = [coverage_keys[i] for i in order]
    div_keys_o = [diversity_keys[i] for i in order]

    diff_bins = np.minimum(n_difficulty_bins - 1, (diff_o * n_difficulty_bins).astype(int))
    diff_bins = np.maximum(0, diff_bins)

    # 被覆判定を整数コードに変換し、以降は完全に配列演算だけで判定する(Pythonループ回避)。
    _, cov_codes = np.unique(cov_keys_o, return_inverse=True)
    _, div_codes = np.unique(div_keys_o, return_inverse=True)
    cov_covered = np.zeros(cov_codes.max() + 1, dtype=bool)
    div_covered = np.zeros(div_codes.max() + 1, dtype=bool)

    n = n_total
    available = np.ones(n, dtype=bool)
    max_sim_to_selected = np.zeros(n)  # facility location: 既選択集合との最大類似度
    difficulty_hist = np.zeros(n_difficulty_bins)

    selected_local: list[int] = []
    n_target = min(n_target, n)

    for _ in range(n_target):
        cov_gain = (~cov_covered[cov_codes]).astype(float) + (~div_covered[div_codes]).astype(float)
        div_gain = 1.0 - max_sim_to_selected

        total_selected = difficulty_hist.sum()
        fill_ratio = difficulty_hist[diff_bins] / total_selected if total_selected > 0 else np.zeros(n)
        diff_gain = 1.0 - fill_ratio

        score = w_coverage * cov_gain + w_diversity * div_gain + w_difficulty_balance * diff_gain
        score = np.where(available, score, -np.inf)

        best_idx = int(np.argmax(score))
        selected_local.append(best_idx)
        available[best_idx] = False

        cov_covered[cov_codes[best_idx]] = True
        div_covered[div_codes[best_idx]] = True
        difficulty_hist[diff_bins[best_idx]] += 1

        # facility location の差分更新: 新規選択との類似度で max_sim を更新(ベクトル化)
        sims_to_new = emb_o @ emb_o[best_idx]
        np.maximum(max_sim_to_selected, sims_to_new, out=max_sim_to_selected)

    return [int(order[i]) for i in selected_local]
