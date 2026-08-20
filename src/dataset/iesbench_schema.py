"""
IESBench 互換スキーマ定義とローダー/バリデータ。

公式 IESBench(CSU-JPG/IESBench)のアノテーションJSON形式をそのまま読み込めるよう、
フィールド名を公式スキーマに合わせている:
  image_id, image_path, question, attributes, action, category, rewrite

このモジュールは合成データ(本リポジトリで生成するプレースホルダーデータセット)と
実データ(公式配布物)の両方を同じインターフェースで扱うために使う。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.utils.io_utils import read_json, read_jsonl, write_jsonl

# 15の安全ポリシーカテゴリ(一般的なAI安全ポリシー分類を参考にしたプレースホルダー粒度のラベル)
SAFETY_POLICIES: list[str] = [
    "violence_graphic",
    "weapons",
    "self_harm",
    "sexual_content_adult",
    "sexual_content_minors_protection",  # 常に最優先で拒否すべきカテゴリ
    "hate_harassment",
    "illegal_drugs",
    "regulated_goods",
    "privacy_pii_exposure",
    "impersonation_identity",
    "fraud_scam",
    "misinformation_fabricated_evidence",
    "intellectual_property",
    "extremism_terrorism",
    "dangerous_activities_howto",
]

# 9つの編集アクション
EDIT_ACTIONS: list[str] = [
    "add_object",
    "remove_object",
    "replace_object",
    "style_transfer",
    "inpaint_region",
    "outpaint_extend",
    "recolor",
    "identity_swap",
    "background_change",
]

# 属性(公式は116種。ここでは代表的なカテゴリからサンプルとして生成する)
_ATTRIBUTE_BASE = [
    "facial_expression", "clothing", "body_pose", "skin_tone", "age_appearance",
    "background_scene", "text_overlay", "logo_brand", "weapon_object", "vehicle",
    "accessory", "hairstyle", "makeup", "lighting", "camera_angle", "location_signage",
    "document_text", "id_card_field", "currency_object", "medical_object",
]
_ATTRIBUTE_MODIFIERS = ["primary", "secondary", "background", "foreground", "partial", "full"]

ATTRIBUTES: list[str] = sorted(
    {f"{mod}_{base}" for base in _ATTRIBUTE_BASE for mod in _ATTRIBUTE_MODIFIERS}
)[:116]


@dataclass
class IESBenchEntry:
    image_id: str
    image_path: str
    question: str
    attributes: list[str]
    action: str
    category: str
    rewrite: str = ""
    extra: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = []
        if self.action not in EDIT_ACTIONS:
            errors.append(f"未知のaction: {self.action}")
        if self.category not in SAFETY_POLICIES:
            errors.append(f"未知のcategory: {self.category}")
        if not self.attributes:
            errors.append("attributesが空です")
        if not self.image_id:
            errors.append("image_idが空です")
        return errors


def load_entries(path: str | Path) -> list[IESBenchEntry]:
    path = Path(path)
    if path.is_dir():
        json_path = path / "annotations.json"
        raw = read_json(json_path) if json_path.exists() else []
    elif path.suffix == ".jsonl":
        raw = list(read_jsonl(path))
    else:
        raw = read_json(path)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]

    entries = []
    for row in raw:
        entries.append(
            IESBenchEntry(
                image_id=row.get("image_id", ""),
                image_path=row.get("image_path", row.get("image-path", "")),
                question=row.get("question", ""),
                attributes=row.get("attributes", []),
                action=row.get("action", ""),
                category=row.get("category", ""),
                rewrite=row.get("rewrite", ""),
                extra={k: v for k, v in row.items() if k not in
                       {"image_id", "image_path", "image-path", "question", "attributes", "action", "category", "rewrite"}},
            )
        )
    return entries


def save_entries(entries: list[IESBenchEntry], out_path: str | Path) -> None:
    rows = []
    for e in entries:
        d = asdict(e)
        extra = d.pop("extra", {})
        d.update(extra)
        rows.append(d)
    write_jsonl(out_path, rows)


def main() -> None:
    p = argparse.ArgumentParser(description="IESBench互換データのロード・検証")
    p.add_argument("--load", required=True)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    entries = load_entries(args.load)
    print(f"読み込みエントリ数: {len(entries)}")
    if args.validate:
        n_bad = 0
        for e in entries:
            errs = e.validate()
            if errs:
                n_bad += 1
                print(f"  [NG] {e.image_id}: {errs}")
        print(f"検証結果: {len(entries) - n_bad}/{len(entries)} 件がスキーマ準拠")


if __name__ == "__main__":
    main()
