"""
既存データセット(公式IESBench配布物、または同スキーマ互換の他データセット)を
「加工」して最適化サブセットを作るモジュール。

src/dataset/dataset_optimizer.py が「1から作成」(SAFETY_POLICIES/EDIT_ACTIONSの直積から
プレースホルダー候補を合成生成する)のに対し、本モジュールは既に存在するデータセットの
エントリ(実画像・実プロンプト・実カテゴリラベル)そのものを候補として扱い、同じ被覆×多様性×
難易度バランスの貪欲法(src/dataset/coverage_optimizer.py::greedy_select() を共有)で
「小さく、しかし偏りのないサブセット」を選抜する。プレースホルダーの合成テキストは一切生成せず、
選抜結果は入力データセットの実エントリ(画像パス・question・rewrite等)をそのまま保持する。

想定用途:
  - 32GB程度のVRAM/限られた計算時間で実験したい場合に、公式IESBench全1054件から
    ポリシー(I1〜I15)×アクション種別の被覆をできるだけ保ったままN件に間引く
    (docs/09_resource_requirements.md の計算資源制約への対応)
  - train_safety_dpo.py --build-from-iesbench に渡す選好データのソースを、
    特定カテゴリに偏らない代表的なサブセットに絞り込む
  - IESBench以外のデータセットでも、iesbench_schema.load_entries() が読める形式
    (image_id/image_path/question/attributes/action/category/rewrite のJSON/JSONL)に
    変換さえすれば同じ加工パイプラインに載せられる
    (docs/10_official_dataset_workflow.md 参照)。

difficulty(難易度)は人手評価が無い場合のプロキシとして lexicon_optimizer.py の脅威語彙
スコアを使う(語彙的に明白なほど「易」とみなすプレースホルダーの仮定であり、実運用では
実測ASR等の難易度指標に置き換えること)。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from src.dataset.coverage_optimizer import greedy_select, hash_embedding
from src.dataset.iesbench_schema import IESBenchEntry, action_label, category_label, load_entries, save_entries
from src.dataset.lexicon_optimizer import ThreatLexicon, build_lexicon
from src.utils.io_utils import write_json
from src.utils.seed import set_seed


@dataclass
class AdaptedCandidate:
    entry: IESBenchEntry
    category: str  # category_label()で正規化済み(複数ポリシーは'+'結合)
    action: str     # action_label()で正規化済み
    attribute: str
    difficulty: float
    embedding: np.ndarray = field(repr=False)


def _default_lexicon() -> ThreatLexicon:
    entries, _ = build_lexicon()
    return ThreatLexicon(entries)


def build_candidate_pool_from_dataset(
    source: str, lexicon: ThreatLexicon | None = None,
) -> list[AdaptedCandidate]:
    """
    iesbench_schema.load_entries() が読める形式(ディレクトリ/.json/.jsonl)なら、
    公式IESBenchでも、同スキーマに変換した他データセットでも受け付ける。
    """
    lexicon = lexicon or _default_lexicon()
    entries = load_entries(source)
    pool = []
    for e in entries:
        category = category_label(e.category)
        action = action_label(e.action)
        attribute = e.attributes[0] if e.attributes else "unknown"
        text = f"{e.question} {e.rewrite}".strip()
        score = lexicon.score_text(text)
        # weighted_scoreが高いほど語彙的に「わかりやすい(易)」とみなし、difficultyは反転して正規化
        # (curriculum_dpo.pyのdifficulty算出と同じ変換規則)。
        difficulty = 1.0 / (1.0 + score["weighted_score"])
        pool.append(
            AdaptedCandidate(
                entry=e,
                category=category,
                action=action,
                attribute=attribute,
                difficulty=round(float(difficulty), 4),
                embedding=hash_embedding(text),
            )
        )
    return pool


def optimize_from_dataset(
    source: str,
    n_target: int,
    w_coverage: float = 1.0,
    w_diversity: float = 1.0,
    w_difficulty_balance: float = 0.5,
    n_difficulty_bins: int = 3,
    seed: int = 42,
    lexicon: ThreatLexicon | None = None,
) -> tuple[list[AdaptedCandidate], list[AdaptedCandidate]]:
    """(元の候補プール全体, 選抜されたサブセット) を返す。"""
    pool = build_candidate_pool_from_dataset(source, lexicon=lexicon)
    if not pool:
        return pool, []

    embeddings = np.stack([c.embedding for c in pool])
    difficulties = np.array([c.difficulty for c in pool])
    coverage_keys = [f"{c.category}|{c.action}" for c in pool]
    diversity_keys = [c.attribute for c in pool]

    idx = greedy_select(
        embeddings, difficulties, coverage_keys, diversity_keys, n_target,
        w_coverage=w_coverage, w_diversity=w_diversity, w_difficulty_balance=w_difficulty_balance,
        n_difficulty_bins=n_difficulty_bins, seed=seed,
    )
    return pool, [pool[i] for i in idx]


def coverage_report(pool: list[AdaptedCandidate], selected: list[AdaptedCandidate]) -> dict:
    all_cats = {c.category for c in pool}
    all_pairs = {f"{c.category}|{c.action}" for c in pool}
    all_attrs = {c.attribute for c in pool}
    sel_cats = {c.category for c in selected}
    sel_pairs = {f"{c.category}|{c.action}" for c in selected}
    sel_attrs = {c.attribute for c in selected}
    return {
        "n_pool": len(pool),
        "n_selected": len(selected),
        "categories_covered": len(sel_cats),
        "categories_total_in_pool": len(all_cats),
        "category_action_pairs_covered": len(sel_pairs),
        "category_action_pairs_total_in_pool": len(all_pairs),
        "attributes_covered": len(sel_attrs),
        "attributes_total_in_pool": len(all_attrs),
    }


def selected_to_entries(selected: list[AdaptedCandidate]) -> list[IESBenchEntry]:
    """選抜結果を、入力データセットの実エントリのまま(合成テキストを混ぜずに)返す。"""
    return [c.entry for c in selected]


def main() -> None:
    p = argparse.ArgumentParser(
        description="既存データセット(公式IESBench等)から被覆×多様性×難易度バランスでサブセットを加工・選抜する"
    )
    p.add_argument(
        "--source", required=True,
        help="iesbench_schema.load_entries() が読める形式(annotation.jsonを含むディレクトリ、"
             "または直接 .json/.jsonl ファイル)。公式IESBenchの展開先ディレクトリを直接指定可",
    )
    p.add_argument("--n-target", type=int, default=200)
    p.add_argument("--w-coverage", type=float, default=1.0)
    p.add_argument("--w-diversity", type=float, default=1.0)
    p.add_argument("--w-difficulty-balance", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/adapted_subset.jsonl")
    p.add_argument("--report-out", default="outputs/adapted_subset_coverage_report.json")
    args = p.parse_args()

    set_seed(args.seed)
    pool, selected = optimize_from_dataset(
        args.source, args.n_target,
        w_coverage=args.w_coverage, w_diversity=args.w_diversity,
        w_difficulty_balance=args.w_difficulty_balance, seed=args.seed,
    )
    print(f"元データセット件数: {len(pool)}")
    if not pool:
        return

    entries = selected_to_entries(selected)
    save_entries(entries, args.out)
    report = coverage_report(pool, selected)
    write_json(args.report_out, report)

    print(f"選抜件数: {len(entries)} -> {args.out}")
    print(f"被覆レポート -> {args.report_out}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
