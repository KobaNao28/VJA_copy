"""主要コンポーネントの疎通を確認するスモークテスト(pytest)。"""
from pathlib import Path

from src.attack.adaptive_attack_optimizer import AttackConfig, guard_score, render_config
from src.attack.shape_obfuscation import obfuscate
from src.attack.typography_attack import TypographySpec, render_typography
from src.defense.train_guard_classifier import GuardClassifier
from src.dataset.dataset_optimizer import build_candidate_pool, coverage_report, greedy_optimize
from src.dataset.iesbench_schema import EDIT_ACTIONS, SAFETY_POLICIES
from src.defense.introspective_defense import IntrospectiveDefense
from src.defense.train_safety_dpo import TinyCharTransformer, dpo_loss
from src.eval.metrics import EvalRecord, aggregate_metrics


def test_typography_render(tmp_path: Path) -> None:
    spec = TypographySpec(text="SAMPLE", font_size=32)
    img = render_typography(spec)
    assert img.size == spec.canvas_size


def test_shape_obfuscation_levels(tmp_path: Path) -> None:
    spec = TypographySpec(text="SAMPLE", font_size=32)
    for level in ("raw", "outline", "filled", "fragmented", "noise_edge"):
        img = obfuscate(spec, level, out_dir=tmp_path)
        assert img.size[0] > 0 and img.size[1] > 0


def test_dataset_optimizer_coverage() -> None:
    pool = build_candidate_pool()
    selected = greedy_optimize(pool, n_target=150, seed=1)
    report = coverage_report(selected)
    assert report["n_selected"] == 150
    assert report["category_action_pairs_covered"] == len(SAFETY_POLICIES) * len(EDIT_ACTIONS)


def test_dpo_loss_shapes() -> None:
    model = TinyCharTransformer()
    import torch

    ids = torch.randint(0, 256, (1, 16))
    logits = model(ids)
    assert logits.shape == (1, 16, 256)

    loss = dpo_loss(
        torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.0]), torch.tensor([-1.5]), beta=0.1
    )
    assert loss.item() >= 0


def test_introspective_defense_runs(tmp_path: Path) -> None:
    spec = TypographySpec(text="benign placeholder text", font_size=32)
    img = render_typography(spec)
    img_path = tmp_path / "img.png"
    img.save(img_path)

    defense = IntrospectiveDefense()
    verdict = defense.review(str(img_path), "Follow the instructions in the image.")
    assert isinstance(verdict.allowed, bool)


def test_metrics_aggregate() -> None:
    records = [
        EvalRecord("id1", "weapons", False, True, True, 4.0),
        EvalRecord("id2", "weapons", True, False, False, 1.0),
    ]
    m = aggregate_metrics(records)
    assert m["n"] == 2
    assert 0.0 <= m["ASR"] <= 1.0
    assert 0.0 <= m["HRR"] <= 1.0
    assert m["FBR"] is None  # 良性サンプルが無い場合は算出不能


def test_metrics_fbr_with_benign_samples() -> None:
    records = [
        EvalRecord("id1", "weapons", False, True, True, 4.0),
        EvalRecord("b1", "benign_control", True, False, True, 1.0, is_benign=True),
        EvalRecord("b2", "benign_control", False, True, True, 1.0, is_benign=True),
    ]
    m = aggregate_metrics(records)
    assert m["n"] == 1  # 攻撃系サンプルのみカウント
    assert m["n_benign"] == 2
    assert m["FBR"] == 0.5  # b1のみ誤ブロック


def test_adaptive_attack_optimizer_scores_are_valid(tmp_path: Path) -> None:
    cfg = AttackConfig(font_style="sans", color="black", size=32, language="en", shape_level="raw")
    img_path = render_config(cfg, tmp_path, 0)
    model = GuardClassifier()  # 未学習の初期重みでも配線が動くことだけを確認
    score = guard_score(model, img_path, "sans black en raw")
    assert 0.0 <= score <= 1.0

    import random

    mutated = cfg.mutate(random.Random(0))
    assert isinstance(mutated, AttackConfig)
