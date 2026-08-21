"""
多層防御パイプライン: guard classifier(前段フィルタ) + introspective defense(内省的推論)
+ セッション単位の累積リスク追跡(マルチターン分割攻撃対策, docs/02_attack_enhancement_proposals.md 2.3節)
を統合したランタイム。

設計方針(docs/03_defense_survey.md の「多層防御・スイスチーズモデル」節に対応):
  単一の防御層は必ず突破されうるという前提に立ち、
    Layer 0: Attack Immune Memory(任意)で既知の検知回避パターンと高速照合(immune_memory_defense.py)
    Layer 1: 軽量 guard classifier で明らかに攻撃的なバリアントを高速に足切り
    Layer 2: introspective defense でテキスト空間に持ち込んだ上で安全性判定
    Layer 3: セッション全体の累積リスクスコアで、単発では無害でも
             複数リクエストに分割された攻撃(salami slicing)を検知
  のいずれかで検知できれば拒否する(OR結合、多層防御)。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from src.defense.immune_memory_defense import AttackMemoryBank, embed
from src.defense.introspective_defense import IntrospectiveDefense, ReasoningFn
from src.defense.train_guard_classifier import GuardClassifier, load_image_tensor


@dataclass
class UnifiedVerdict:
    allowed: bool
    layer_triggered: str
    rationale: str
    session_risk_score: float


@dataclass
class SessionRiskTracker:
    """マルチターンに跨る累積リスクスコアリング(docs/02 2.3節への対応)。"""

    decay: float = 0.85
    threshold: float = 1.5
    _scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def update(self, session_id: str, request_risk: float) -> float:
        prev = self._scores[session_id]
        new_score = prev * self.decay + request_risk
        self._scores[session_id] = new_score
        return new_score

    def is_over_threshold(self, session_id: str) -> bool:
        return self._scores[session_id] >= self.threshold


class UnifiedDefensePipeline:
    def __init__(
        self,
        guard_model: GuardClassifier | None = None,
        guard_threshold: float = 0.5,
        reasoning_fn: ReasoningFn | None = None,
        session_risk_threshold: float = 1.5,
        memory_bank: AttackMemoryBank | None = None,
    ):
        self.guard_model = guard_model
        self.guard_threshold = guard_threshold
        self.introspective = IntrospectiveDefense(reasoning_fn=reasoning_fn)
        self.risk_tracker = SessionRiskTracker(threshold=session_risk_threshold)
        # Layer 0: Attack Immune Memory(任意)。guard_modelの特徴抽出器を流用するため
        # guard_modelが無いと埋め込みが作れず、その場合はこの層はスキップされる。
        self.memory_bank = memory_bank

    def _guard_score(self, image_path: str, prompt: str) -> float:
        if self.guard_model is None:
            return 0.0
        self.guard_model.eval()
        with torch.no_grad():
            img = load_image_tensor(image_path).unsqueeze(0)
            logit = self.guard_model(img, [prompt])
            return torch.sigmoid(logit).item()

    def process(self, session_id: str, image_path: str, prompt: str) -> UnifiedVerdict:
        # Layer 0: Attack Immune Memory(既知の検知回避パターンとの高速照合)
        if self.memory_bank is not None and self.guard_model is not None:
            query_emb = embed(self.guard_model, image_path, prompt)
            match = self.memory_bank.query(query_emb)
            if match.matched:
                risk = self.risk_tracker.update(session_id, 1.0)
                booster_note = "(booster再学習を推奨: 同カテゴリへの反復曝露を検出)" if match.booster_retrain_needed else ""
                return UnifiedVerdict(
                    allowed=False,
                    layer_triggered="immune_memory",
                    rationale=f"既知の検知回避パターン(category={match.category})と類似度{match.best_similarity}で一致 {booster_note}",
                    session_risk_score=risk,
                )

        # Layer 1: guard classifier
        guard_score = self._guard_score(image_path, prompt)
        if guard_score >= self.guard_threshold:
            risk = self.risk_tracker.update(session_id, guard_score)
            return UnifiedVerdict(
                allowed=False,
                layer_triggered="guard_classifier",
                rationale=f"guard_classifier score={guard_score:.3f} >= threshold={self.guard_threshold}",
                session_risk_score=risk,
            )

        # Layer 2: introspective defense
        verdict = self.introspective.review(image_path, prompt)
        request_risk = 0.0 if verdict.allowed else 1.0
        # 検知されなかった場合でも、部分的にリスクのある視覚指示があれば緩やかに加点する
        if verdict.allowed and verdict.extracted_visual_instructions != "画像内に明示的な視覚指示は検出されなかった":
            request_risk = 0.3

        risk = self.risk_tracker.update(session_id, request_risk)

        if not verdict.allowed:
            return UnifiedVerdict(
                allowed=False,
                layer_triggered="introspective_defense",
                rationale=verdict.rationale,
                session_risk_score=risk,
            )

        # Layer 3: セッション累積リスク
        if self.risk_tracker.is_over_threshold(session_id):
            return UnifiedVerdict(
                allowed=False,
                layer_triggered="session_risk_tracker",
                rationale=f"セッション累積リスクスコア {risk:.2f} が閾値 {self.risk_tracker.threshold} を超過(分割攻撃の疑い)",
                session_risk_score=risk,
            )

        return UnifiedVerdict(
            allowed=True,
            layer_triggered="none",
            rationale="全防御層を通過",
            session_risk_score=risk,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="統合防御パイプラインのデモ実行")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Follow the instructions in the image.")
    p.add_argument("--session-id", default="demo-session")
    args = p.parse_args()

    pipeline = UnifiedDefensePipeline()
    result = pipeline.process(args.session_id, args.image, args.prompt)
    print(result)
