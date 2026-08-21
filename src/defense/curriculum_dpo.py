"""
Curriculum DPO: 選好ペアの「提示順序」がDPO安全アライメント学習に与える影響の実験枠組み。

## 研究動機

DPO自体の学習手法(損失関数、正則化)の研究は数多いが、
「選好ペアをどの順序でモデルに提示するか」という**カリキュラム設計**は
ほとんど検討されていない(docs/03_defense_survey.md 2.3節の指摘)。
教師あり学習では easy-to-hard カリキュラム(Bengio et al., 2009)や
自己ペース学習(Kumar et al., 2010)が収束と汎化を改善することが知られており、
継続学習(continual learning)分野では「後から学んだタスクが先に学んだタスクの
性能を劣化させる」破局的忘却(catastrophic forgetting)がよく知られた問題である。
DPOによる安全アライメントは実質的に「複数の安全カテゴリを順に(あるいは混ぜて)学習する
継続学習タスク」とみなせるため、これらの知見がそのまま応用できる可能性が高い。

## 検証する順序戦略

  - random              : ベースライン(シャッフルのみ)
  - easy_to_hard         : 語彙脅威スコア(lexicon_optimizer)が高い「わかりやすい」ペアから開始
  - hard_to_easy         : 逆順(anti-curriculum、比較対象)
  - spaced_repetition    : easy_to_hard順で1周した後、難しい後半1/3をエポック中に再注入する
                            (間隔反復: 忘れやすい難例を繰り返し再提示)
  - category_blocked     : 安全カテゴリごとにブロックして順に学習(継続学習の典型的セットアップ)
  - category_interleaved : カテゴリをラウンドロビンで交互に学習(blockedとの対比)
  - self_paced           : 各エポック開始時に現在のモデルの損失を測り、損失が低い(=既に
                            分離できている)ペアから提示する(Kumar et al. 2010 の自己ペース学習)

## 測定する指標

  - steps_to_threshold  : 損失が閾値を下回るまでのステップ数(収束速度)
  - final_loss          : 学習終了時の平均DPO損失
  - forgetting_score     : category_blocked戦略に固有。最初に学習したカテゴリのブロックについて、
                            「学習直後の損失」と「全カテゴリ学習後の損失」の差(破局的忘却の直接測定)
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.dataset.iesbench_schema import SAFETY_POLICIES
from src.dataset.lexicon_optimizer import ThreatLexicon, build_lexicon
from src.defense.train_safety_dpo import TinyCharTransformer, _tiny_sequence_logp, dpo_loss
from src.utils.io_utils import write_json
from src.utils.seed import set_seed

CurriculumStrategy = str  # "random" | "easy_to_hard" | "hard_to_easy" | "spaced_repetition"
# | "category_blocked" | "category_interleaved" | "self_paced"


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    category: str
    difficulty: float = 0.0  # 0=易(語彙シグナルが強く分離しやすい) 〜 1=難(曖昧)


def load_pairs(path: str, lexicon: ThreatLexicon) -> list[PreferencePair]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = lexicon.score_text(row["prompt"] + " " + row["rejected"])
        # weighted_scoreが高いほど「わかりやすい(易)」ペアなので、difficultyは反転して正規化
        difficulty = 1.0 / (1.0 + score["weighted_score"])
        category = _guess_category_from_prompt(row["prompt"])
        pairs.append(PreferencePair(row["prompt"], row["chosen"], row["rejected"], category, difficulty))
    return pairs


def _guess_category_from_prompt(prompt: str) -> str:
    """
    プロンプト文中から安全カテゴリ名を推定する。
    - train_safety_dpo.build_synthetic_preference_data が生成する
      "policy-category under review: <category>)" 形式に対応
    - build_dataset.py が画像に埋め込む "category=<category>" 形式にも対応
    """
    for category in SAFETY_POLICIES:
        if category in prompt:
            return category
    return "unknown"


def order_pairs(pairs: list[PreferencePair], strategy: CurriculumStrategy, rng, model=None) -> list[PreferencePair]:
    if strategy == "random":
        out = pairs.copy()
        rng.shuffle(out)
        return out

    if strategy == "easy_to_hard":
        return sorted(pairs, key=lambda p: p.difficulty)

    if strategy == "hard_to_easy":
        return sorted(pairs, key=lambda p: -p.difficulty)

    if strategy == "spaced_repetition":
        ordered = sorted(pairs, key=lambda p: p.difficulty)
        n = len(ordered)
        hard_third = ordered[2 * n // 3:]
        # 易→難で1周した後、難しい後半1/3を末尾にもう一度差し込む(間隔反復)
        return ordered + hard_third

    if strategy == "category_blocked":
        by_cat: dict[str, list[PreferencePair]] = {}
        for p in pairs:
            by_cat.setdefault(p.category, []).append(p)
        out = []
        for cat in sorted(by_cat):  # カテゴリ名でソートし決定的な順序にする
            out.extend(by_cat[cat])
        return out

    if strategy == "category_interleaved":
        by_cat: dict[str, list[PreferencePair]] = {}
        for p in pairs:
            by_cat.setdefault(p.category, []).append(p)
        queues = [by_cat[c] for c in sorted(by_cat)]
        out = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
        return out

    if strategy == "self_paced":
        if model is None:
            return sorted(pairs, key=lambda p: p.difficulty)
        scored = []
        model.eval()
        with torch.no_grad():
            for p in pairs:
                pc = _tiny_sequence_logp(model, p.prompt, p.chosen)
                pr = _tiny_sequence_logp(model, p.prompt, p.rejected)
                margin = (pc - pr).item()  # margin大 = モデルが既に分離できている(易)
                scored.append((-margin, p))
        scored.sort(key=lambda t: t[0])
        return [p for _, p in scored]

    raise ValueError(f"未知のcurriculum戦略: {strategy}")


def train_one_strategy(
    pairs: list[PreferencePair],
    strategy: CurriculumStrategy,
    epochs: int = 3,
    lr: float = 1e-3,
    beta: float = 0.1,
    loss_threshold: float = 0.05,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    torch.manual_seed(seed)
    policy = TinyCharTransformer()
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = __import__("random").Random(seed)

    step = 0
    steps_to_threshold = None
    loss_curve = []
    first_category_loss_right_after: float | None = None
    first_category = pairs[0].category if strategy == "category_blocked" else None

    for epoch in range(epochs):
        ordered = order_pairs(pairs, strategy, rng, model=policy if strategy == "self_paced" else None)
        for pair in ordered:
            pc = _tiny_sequence_logp(policy, pair.prompt, pair.chosen)
            pr = _tiny_sequence_logp(policy, pair.prompt, pair.rejected)
            with torch.no_grad():
                rc = _tiny_sequence_logp(ref, pair.prompt, pair.chosen)
                rr = _tiny_sequence_logp(ref, pair.prompt, pair.rejected)
            loss = dpo_loss(pc, pr, rc, rr, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_curve.append(loss_val)
            step += 1
            if steps_to_threshold is None and loss_val < loss_threshold:
                steps_to_threshold = step

            if strategy == "category_blocked" and pair.category == first_category and first_category_loss_right_after is None:
                pass  # 後段でブロック境界検出時に確定させる

        if strategy == "category_blocked" and epoch == 0 and first_category_loss_right_after is None:
            # 最初のカテゴリブロックの学習が終わった直後の、そのカテゴリでの損失を測定
            first_block_pairs = [p for p in pairs if p.category == first_category]
            with torch.no_grad():
                losses = []
                for p in first_block_pairs:
                    pc = _tiny_sequence_logp(policy, p.prompt, p.chosen)
                    pr = _tiny_sequence_logp(policy, p.prompt, p.rejected)
                    rc = _tiny_sequence_logp(ref, p.prompt, p.chosen)
                    rr = _tiny_sequence_logp(ref, p.prompt, p.rejected)
                    losses.append(dpo_loss(pc, pr, rc, rr, beta=beta).item())
                first_category_loss_right_after = sum(losses) / max(1, len(losses))

    forgetting_score = None
    if strategy == "category_blocked" and first_category_loss_right_after is not None:
        first_block_pairs = [p for p in pairs if p.category == first_category]
        with torch.no_grad():
            losses = []
            for p in first_block_pairs:
                pc = _tiny_sequence_logp(policy, p.prompt, p.chosen)
                pr = _tiny_sequence_logp(policy, p.prompt, p.rejected)
                rc = _tiny_sequence_logp(ref, p.prompt, p.chosen)
                rr = _tiny_sequence_logp(ref, p.prompt, p.rejected)
                losses.append(dpo_loss(pc, pr, rc, rr, beta=beta).item())
            final_first_category_loss = sum(losses) / max(1, len(losses))
        forgetting_score = final_first_category_loss - first_category_loss_right_after

    return {
        "strategy": strategy,
        "n_steps": step,
        "steps_to_threshold": steps_to_threshold,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "mean_last_10pct_loss": sum(loss_curve[-max(1, len(loss_curve) // 10):]) / max(1, len(loss_curve[-max(1, len(loss_curve) // 10):])),
        "forgetting_score": forgetting_score,
        "loss_curve": loss_curve,
    }


def compare_strategies(
    data_path: str,
    strategies: list[CurriculumStrategy],
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    lexicon_entries, _ = build_lexicon()
    lexicon = ThreatLexicon(lexicon_entries)
    pairs = load_pairs(data_path, lexicon)
    if not pairs:
        raise SystemExit(f"選好データが空です: {data_path}")

    results = {}
    for strategy in strategies:
        results[strategy] = train_one_strategy(pairs, strategy, epochs=epochs, seed=seed)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Curriculum DPO: 選好ペア提示順序の比較実験")
    p.add_argument("--data", default="data/sample/dpo_preferences.jsonl")
    p.add_argument(
        "--strategies",
        default="random,easy_to_hard,hard_to_easy,spaced_repetition,category_blocked,category_interleaved,self_paced",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/curriculum_dpo_report.json")
    args = p.parse_args()

    strategies = args.strategies.split(",")
    results = compare_strategies(args.data, strategies, epochs=args.epochs, seed=args.seed)

    print(f"{'strategy':<22} {'steps_to_thr':>13} {'final_loss':>11} {'last10%_loss':>13} {'forgetting':>11}")
    for strategy, r in results.items():
        stt = r["steps_to_threshold"] if r["steps_to_threshold"] is not None else "N/A"
        fgt = f"{r['forgetting_score']:.4f}" if r["forgetting_score"] is not None else "N/A"
        print(f"{strategy:<22} {str(stt):>13} {r['final_loss']:>11.4f} {r['mean_last_10pct_loss']:>13.4f} {fgt:>11}")

    # loss_curveはサイズが大きいので出力JSONには要約統計のみ残す
    summary = {
        s: {k: v for k, v in r.items() if k != "loss_curve"} for s, r in results.items()
    }
    write_json(args.out, summary)
    print(f"\nレポート出力: {args.out} (loss_curve全系列は省略、summaryのみ)")


if __name__ == "__main__":
    main()
