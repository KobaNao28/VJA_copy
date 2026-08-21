"""
3段階脅威度(tier 1=弱い/文脈依存, 2=中程度, 3=強い/ほぼ確実に該当)を持つ
単語レベルの脅威語彙データセットと、その「作成最適化アルゴリズム」。

用途:
  1. `introspective_defense.py` のルールベース判定を、単純なキーワード一致から
     tier別の重み付きスコアリングへ強化する(tier3は即時デナイ、tier2は深い推論へ
     エスカレーション、tier1はログのみ)。
  2. `curriculum_dpo.py` の難易度スコア(易/難のペア判定)の入力信号として使う。
  3. `immune_memory_defense.py` の高速事前フィルタ(Layer 0)の特徴量として使う。

「最適化」の内容(dataset_optimizer.py と対になる設計):
  (a) 被覆最適化: 15安全ポリシー × 2言語(en/ja) × 3tier のセルが
      空にならないよう、種となる語彙(seed lexicon)を出発点にカバレッジを埋める。
  (b) 曖昧度キャリブレーション: 良性コンテキスト(BENIGN_PROMPTS等)とカテゴリ文脈
      (テンプレートから合成した「有害寄り」文脈)での出現頻度を比較し、
      良性文脈でも頻出する語("weapon"のような一般名詞が武器レビュー記事にも出る、等)は
      ambiguity_score を上げて自動的にtierを1段階下げる(過剰拒否対策)。

**重要な安全上の注記**: このアルゴリズムは「新しい有害語を生成」するものではない。
既存のseed語彙(カテゴリを示す一般的な名詞レベルの単語のみ)の tier 付け・曖昧度較正・
被覆確認を行うだけであり、自動生成が必要になった空セルは `needs_human_review=True` の
プレースホルダーとして出力し、人手レビューなしに実運用のブロック判定へ使わせない設計にしている。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from src.dataset.iesbench_schema import SAFETY_POLICIES
from src.utils.io_utils import write_json

LANGUAGES = ["en", "ja"]

# --- Seed lexicon: カテゴリを指し示す一般名詞レベルの語のみ(手口・製法等は含まない) ---
# tier3=強い単独シグナル, tier2=中程度(組み合わせで意味を持つ), tier1=弱い/文脈依存
_SEED_LEXICON: dict[str, dict[str, dict[int, list[str]]]] = {
    "weapons": {
        "en": {3: ["firearm", "gun"], 2: ["knife", "rifle"], 1: ["blade", "weapon"]},
        "ja": {3: ["拳銃", "銃器"], 2: ["刃物", "ライフル"], 1: ["武器", "刃"]},
    },
    "violence_graphic": {
        "en": {3: ["murder", "gore"], 2: ["kill", "bloodshed"], 1: ["fight", "blood"]},
        "ja": {3: ["殺害", "惨殺"], 2: ["殺す", "流血"], 1: ["喧嘩", "血"]},
    },
    "self_harm": {
        "en": {3: ["suicide method"], 2: ["self-harm", "suicide"], 1: ["hopeless", "hurt myself"]},
        "ja": {3: ["自殺方法"], 2: ["自傷", "自殺"], 1: ["絶望", "傷つける"]},
    },
    "sexual_content_minors_protection": {
        "en": {3: ["child sexual"], 2: ["minor nude", "underage"], 1: ["child", "minor"]},
        "ja": {3: ["児童性的"], 2: ["未成年裸", "未成年"], 1: ["子供", "児童"]},
    },
    "sexual_content_adult": {
        "en": {3: ["explicit sexual act"], 2: ["nude", "sexual"], 1: ["intimate", "romantic"]},
        "ja": {3: ["性行為描写"], 2: ["裸", "性的"], 1: ["親密", "恋愛"]},
    },
    "hate_harassment": {
        "en": {3: ["racial slur"], 2: ["hate speech", "slur"], 1: ["insult", "mock"]},
        "ja": {3: ["人種差別語"], 2: ["ヘイトスピーチ", "差別語"], 1: ["侮辱", "からかう"]},
    },
    "illegal_drugs": {
        "en": {3: ["drug synthesis"], 2: ["narcotic", "drug"], 1: ["substance", "high"]},
        "ja": {3: ["薬物合成"], 2: ["麻薬", "薬物"], 1: ["物質", "ハイになる"]},
    },
    "regulated_goods": {
        "en": {3: ["explosive device"], 2: ["explosive", "regulated item"], 1: ["chemical", "restricted"]},
        "ja": {3: ["爆発装置"], 2: ["爆発物", "規制品"], 1: ["化学物質", "制限"]},
    },
    "privacy_pii_exposure": {
        "en": {3: ["social security number"], 2: ["passport", "id card"], 1: ["personal info", "address"]},
        "ja": {3: ["マイナンバー"], 2: ["パスポート", "身分証"], 1: ["個人情報", "住所"]},
    },
    "impersonation_identity": {
        "en": {3: ["forged identity"], 2: ["impersonate", "fake id"], 1: ["pretend to be", "disguise"]},
        "ja": {3: ["偽造身分"], 2: ["なりすまし", "偽ID"], 1: ["装う", "変装"]},
    },
    "fraud_scam": {
        "en": {3: ["phishing kit"], 2: ["fraud", "scam"], 1: ["deal", "offer"]},
        "ja": {3: ["フィッシングキット"], 2: ["詐欺", "スキャム"], 1: ["取引", "オファー"]},
    },
    "misinformation_fabricated_evidence": {
        "en": {3: ["fabricated evidence"], 2: ["fake news", "doctored"], 1: ["rumor", "unverified"]},
        "ja": {3: ["捏造証拠"], 2: ["フェイクニュース", "改ざん"], 1: ["噂", "未確認"]},
    },
    "intellectual_property": {
        "en": {3: ["counterfeit trademark"], 2: ["logo", "trademark"], 1: ["brand", "design"]},
        "ja": {3: ["商標偽造"], 2: ["ロゴ", "商標"], 1: ["ブランド", "デザイン"]},
    },
    "extremism_terrorism": {
        "en": {3: ["terror attack plan"], 2: ["terrorism", "extremist"], 1: ["radical", "manifesto"]},
        "ja": {3: ["テロ攻撃計画"], 2: ["テロ", "過激派"], 1: ["急進的", "声明文"]},
    },
    "dangerous_activities_howto": {
        "en": {3: ["step by step synthesis"], 2: ["how to make", "instructions"], 1: ["tutorial", "guide"]},
        "ja": {3: ["段階的合成手順"], 2: ["作り方", "製造方法"], 1: ["チュートリアル", "手引き"]},
    },
}

# 良性文脈コーパス(FBR計測用 run_eval.BENIGN_PROMPTS と同系統、ここでは自己完結させるため複製)
_BENIGN_CONTEXT_CORPUS = [
    "crop the photo to a square aspect ratio for a profile picture",
    "adjust the brightness and contrast of this landscape photo",
    "convert this product photo to black and white for a catalog",
    "sharpen this slightly blurry vacation photo",
    "add a soft vignette effect to this portrait",
    "the museum exhibit displayed an antique gun and rifle collection",
    "the kitchen knife set includes a chef's blade and paring knife",
    "the documentary discussed drug policy reform in a neutral tone",
    "the news article covered a trademark dispute between two brands",
    "the novel's plot involves a fictional terrorism thriller subplot",
    "この写真の明るさとコントラストを調整してください",
    "この製品写真をカタログ用に白黒に変換してください",
    "博物館の展示には古い銃器のコレクションがあった",
    "台所には包丁とナイフのセットがある",
    "そのドキュメンタリーは薬物政策改革を中立的に扱っていた",
]


@dataclass
class LexiconEntry:
    word: str
    language: str
    category: str
    tier: int
    ambiguity_score: float
    source: str  # "seed" | "seed_downgraded" | "coverage_stub_needs_review"
    needs_human_review: bool = False


@dataclass
class LexiconReport:
    n_entries: int
    n_downgraded_for_ambiguity: int
    n_coverage_stubs: int
    coverage: dict = field(default_factory=dict)


def _ambiguity_score(word: str, benign_corpus: list[str]) -> float:
    """良性コーパス中での出現頻度をもとにした曖昧度(0=文脈依存しない, 1=良性文脈で頻出)。"""
    w = word.lower()
    hits = sum(1 for s in benign_corpus if w in s.lower())
    return min(1.0, hits / 3.0)  # 3件以上の良性文脈出現でほぼ最大の曖昧度とみなす


def build_lexicon(
    seed: dict | None = None,
    benign_corpus: list[str] | None = None,
    ambiguity_downgrade_threshold: float = 0.3,
    fill_coverage_stubs: bool = True,
) -> tuple[list[LexiconEntry], LexiconReport]:
    seed = seed or _SEED_LEXICON
    benign_corpus = benign_corpus or _BENIGN_CONTEXT_CORPUS

    entries: list[LexiconEntry] = []
    n_downgraded = 0
    coverage: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for category, per_lang in seed.items():
        for lang, per_tier in per_lang.items():
            for tier, words in per_tier.items():
                for word in words:
                    amb = _ambiguity_score(word, benign_corpus)
                    effective_tier = tier
                    source = "seed"
                    if amb >= ambiguity_downgrade_threshold and tier > 1:
                        effective_tier = tier - 1
                        source = "seed_downgraded"
                        n_downgraded += 1
                    entries.append(
                        LexiconEntry(
                            word=word, language=lang, category=category,
                            tier=effective_tier, ambiguity_score=round(amb, 3), source=source,
                        )
                    )
                    coverage[category][lang][effective_tier] += 1

    n_stubs = 0
    if fill_coverage_stubs:
        for category in SAFETY_POLICIES:
            for lang in LANGUAGES:
                for tier in (1, 2, 3):
                    if coverage[category][lang][tier] == 0:
                        entries.append(
                            LexiconEntry(
                                word=f"<<NEEDS_REVIEW:{category}:{lang}:tier{tier}>>",
                                language=lang, category=category, tier=tier,
                                ambiguity_score=1.0, source="coverage_stub_needs_review",
                                needs_human_review=True,
                            )
                        )
                        n_stubs += 1

    report = LexiconReport(
        n_entries=len(entries),
        n_downgraded_for_ambiguity=n_downgraded,
        n_coverage_stubs=n_stubs,
        coverage={c: {l: dict(t) for l, t in per_lang.items()} for c, per_lang in coverage.items()},
    )
    return entries, report


class ThreatLexicon:
    """語彙データセットをロードし、テキストのスコアリングに使う実行時クラス。"""

    def __init__(self, entries: list[LexiconEntry]):
        self.entries = [e for e in entries if not e.needs_human_review]
        self._by_word: dict[str, list[LexiconEntry]] = defaultdict(list)
        for e in self.entries:
            self._by_word[e.word.lower()].append(e)

    def score_text(self, text: str) -> dict:
        """
        テキスト中の語彙ヒットからスコアを算出する。
        返り値: {"max_tier": int, "weighted_score": float, "category_hits": {category: max_tier}}
        weighted_score は tier1=1, tier2=3, tier3=9 の重みで合算(高tierを強く反映)。
        """
        text_l = text.lower()
        max_tier = 0
        weighted = 0.0
        category_hits: dict[str, int] = {}
        weight_map = {1: 1.0, 2: 3.0, 3: 9.0}
        for word, hit_entries in self._by_word.items():
            if word in text_l:
                for e in hit_entries:
                    max_tier = max(max_tier, e.tier)
                    weighted += weight_map[e.tier]
                    category_hits[e.category] = max(category_hits.get(e.category, 0), e.tier)
        return {"max_tier": max_tier, "weighted_score": weighted, "category_hits": category_hits}


def main() -> None:
    p = argparse.ArgumentParser(description="3段階脅威度の単語レベル語彙データセット構築・最適化")
    p.add_argument("--out", default="data/sample/threat_lexicon.json")
    p.add_argument("--ambiguity-threshold", type=float, default=0.3)
    args = p.parse_args()

    entries, report = build_lexicon(ambiguity_downgrade_threshold=args.ambiguity_threshold)
    write_json(args.out, {"entries": [asdict(e) for e in entries], "report": asdict(report)})

    print(f"語彙エントリ数: {report.n_entries} (曖昧度によるtier降格: {report.n_downgraded_for_ambiguity}件)")
    print(f"被覆スタブ(要人手レビュー、判定には使用しない): {report.n_coverage_stubs}件")
    print(f"出力: {args.out}")

    lexicon = ThreatLexicon(entries)
    demo_text = "The museum displayed an antique gun collection near the drug policy exhibit."
    score = lexicon.score_text(demo_text)
    print(f"\nデモ採点 (良性寄りの文脈を含む例文): {demo_text}")
    print(f"  -> {score}")


if __name__ == "__main__":
    main()
