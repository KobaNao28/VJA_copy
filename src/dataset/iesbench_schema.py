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

# 公式IESBenchの実際のcategoryコード(I1〜I15)の正式名称・定義。
# 出典: 公式実装 CSU-JPG/VJA の src/models/qwen_image_edit_safe.py 内の risk_category 定数
# (github.com/CSU-JPG/VJA, 2026年時点でclone・確認済み。VJA論文が提案する防御パイプライン
# QwenImageEditSafePipeline が安全判定のプロンプトに埋め込んで使っている定義そのもの)。
# 本リポジトリの SAFETY_POLICIES(下記、独自の合成ラベル体系)とは無関係で、
# 実データのcategoryコードを人間可読な形でレポートに表示するための参照用途。
OFFICIAL_CATEGORY_NAMES: dict[str, str] = {
    "I1": "Violence and Harm Scene Manipulation",
    "I2": "Privacy Violation",
    "I3": "Self-Harm and Suicide Scene Generation",
    "I4": "Child Abuse",
    "I5": "Animal Abuse",
    "I6": "False Medical and Health Imagery",
    "I7": "Commercial Document Tampering",
    "I8": "Visual Misinformation and Disinformation",
    "I9": "Sexual and Non-consensual Intimate Imagery",
    "I10": "Dangerous Act Instructional Visualization",
    "I11": "Copyright and Authorship Tampering",
    "I12": "Hidden Information Reconstruction",
    "I13": "Evidence Tampering",
    "I14": "Discrimination-based Visual Editing",
    "I15": "Aversive Visual Manipulation",
}

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


def _as_list(value) -> list[str]:
    """公式データはaction/categoryが複数値を許容するリストの場合がある(1件が複数ポリシーに
    抵触しうるため)。本リポジトリの合成データ生成器は単一文字列で渡すため、両対応する。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _labels_to_str(values: str | list[str]) -> str:
    items = _as_list(values)
    if not items:
        return "unknown"
    return "+".join(sorted(str(v) for v in items))


def category_label(category: str | list[str]) -> str:
    """category(str または list[str])を、集計・辞書キーとして使える単一文字列に正規化する。
    複数ポリシーに抵触する場合(公式データはリスト)は '+' 区切りで結合する
    (例: ["I2", "I13"] -> "I2+I13")。単一文字列はそのまま返す。"""
    return _labels_to_str(category)


def action_label(action: str | list[str]) -> str:
    """action(str または list[str])を category_label() と同じ規則で単一文字列に正規化する
    (例: ["add", "remove"] -> "add+remove")。"""
    return _labels_to_str(action)


def describe_category(category: str | list[str]) -> str:
    """categoryコードを OFFICIAL_CATEGORY_NAMES で人間可読な形に展開する
    (例: ["I2", "I13"] -> "I2 (Privacy Violation) + I13 (Evidence Tampering)")。
    未知のコード(本リポジトリの合成ラベル等)はそのまま返す。"""
    parts = []
    for c in _as_list(category):
        name = OFFICIAL_CATEGORY_NAMES.get(c)
        parts.append(f"{c} ({name})" if name else str(c))
    return " + ".join(parts) if parts else "unknown"


@dataclass
class IESBenchEntry:
    image_id: str
    image_path: str
    question: str
    attributes: list[str]
    action: str | list[str]
    category: str | list[str]
    rewrite: str = ""
    extra: dict = field(default_factory=dict)

    def validate(self, check_taxonomy: bool = True) -> list[str]:
        """
        check_taxonomy=True: action/categoryが本リポジトリの合成ラベル体系
          (EDIT_ACTIONS/SAFETY_POLICIES)に含まれるかを厳密に検証する
          (本リポジトリの合成データ向け)。
        check_taxonomy=False: 値が空でないかのみを構造的に検証する
          (公式IESBenchはaction=自由記述動詞、category='I1'〜'I15'のコード名を
          使っており、本リポジトリの推測ラベルとは一致しないため)。
        """
        errors = []
        actions = _as_list(self.action)
        categories = _as_list(self.category)

        if check_taxonomy:
            unknown_actions = [a for a in actions if a not in EDIT_ACTIONS]
            if unknown_actions:
                errors.append(f"未知のaction: {unknown_actions}")
            unknown_categories = [c for c in categories if c not in SAFETY_POLICIES]
            if unknown_categories:
                errors.append(f"未知のcategory: {unknown_categories}")
        else:
            if not actions or not any(actions):
                errors.append("actionが空です")
            if not categories or not any(categories):
                errors.append("categoryが空です")

        if not self.attributes:
            errors.append("attributesが空です")
        if not self.image_id:
            errors.append("image_idが空です")
        return errors

    def looks_like_official_labels(self) -> bool:
        """categoryが 'I1'〜'I15' のような公式コード名パターンに一致するかを判定する。"""
        import re

        return any(re.fullmatch(r"I\d{1,2}", c) for c in _as_list(self.category))


# annotationファイルの名前候補(公式配布での正確なファイル名が確認できていないため、
# 複数の一般的な名前を優先順位付きで試す。存在しないファイル名を決め打ちして
# 黙って0件を返す事故を防ぐため、最終的に見つからなければ明示的にエラーにする)
_ANNOTATION_FILENAME_CANDIDATES = [
    "annotations.json", "annotation.json", "data.json", "metadata.json",
    "iesbench.json", "IESBench.json", "test.json", "eval.json",
    "annotations.jsonl", "data.jsonl", "iesbench.jsonl",
]


def _find_annotation_file(dir_path: Path) -> Path:
    for name in _ANNOTATION_FILENAME_CANDIDATES:
        candidate = dir_path / name
        if candidate.exists():
            return candidate

    # 既知の名前で見つからない場合、直下 → 1階層下の順に *.json/*.jsonl を総当たりで探す。
    # img/assets等の画像フォルダを除外し、候補が1件に絞れれば自動採用する。
    exclude_dirs = {"img", "images", "assets", "image"}
    for depth_glob in ("*.json", "*.jsonl", "*/*.json", "*/*.jsonl"):
        candidates = [
            p for p in dir_path.glob(depth_glob)
            if p.is_file() and not (len(p.relative_to(dir_path).parts) > 1 and p.relative_to(dir_path).parts[0] in exclude_dirs)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"{dir_path} 内でannotationファイルの候補が複数見つかり自動判定できません: "
                f"{[str(p.relative_to(dir_path)) for p in candidates]}\n"
                f"--load に直接ファイルパスを指定してください(例: --load {candidates[0]})"
            )

    all_files = sorted(p.name for p in dir_path.iterdir())[:30] if dir_path.exists() else []
    raise FileNotFoundError(
        f"{dir_path} 内にannotationファイル(json/jsonl)が見つかりませんでした。\n"
        f"既知の候補名: {_ANNOTATION_FILENAME_CANDIDATES}\n"
        f"ディレクトリ直下の内容(先頭30件): {all_files}\n"
        f"実際のファイル名が分かる場合は --load {dir_path}/<実際のファイル名> のように直接指定してください。"
    )


def load_entries(path: str | Path) -> list[IESBenchEntry]:
    path = Path(path)
    if path.is_dir():
        json_path = _find_annotation_file(path)
        base_dir = path
        raw = list(read_jsonl(json_path)) if json_path.suffix == ".jsonl" else read_json(json_path)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]
    elif path.suffix == ".jsonl":
        raw = list(read_jsonl(path))
        base_dir = path.parent
    else:
        raw = read_json(path)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]
        base_dir = path.parent

    entries = []
    for row in raw:
        raw_image_path = row.get("image_path", row.get("image-path", ""))
        resolved_image_path = raw_image_path
        if raw_image_path and not Path(raw_image_path).is_absolute():
            candidate = base_dir / raw_image_path
            if candidate.exists():
                resolved_image_path = str(candidate)
        entries.append(
            IESBenchEntry(
                image_id=row.get("image_id", ""),
                image_path=resolved_image_path,
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
    p.add_argument(
        "--check-taxonomy", choices=["auto", "on", "off"], default="auto",
        help="action/categoryを本リポジトリの合成ラベル体系(EDIT_ACTIONS/SAFETY_POLICIES)と"
             "照合するか。autoなら公式IESBenchの'I1'〜'I15'形式のcategoryを検出した場合に自動でoffにする",
    )
    args = p.parse_args()

    entries = load_entries(args.load)
    print(f"読み込みエントリ数: {len(entries)}")
    if not entries:
        return

    if args.check_taxonomy == "auto":
        check_taxonomy = not entries[0].looks_like_official_labels()
        if not check_taxonomy:
            print(
                "[情報] category が 'I1'〜'I15' 形式(公式IESBenchのラベル)と判定したため、"
                "本リポジトリの合成ラベル体系(SAFETY_POLICIES/EDIT_ACTIONS)との照合はスキップし、"
                "構造的な検証(空チェック)のみ行います。"
            )
    else:
        check_taxonomy = args.check_taxonomy == "on"

    if args.validate:
        n_bad = 0
        for e in entries:
            errs = e.validate(check_taxonomy=check_taxonomy)
            if errs:
                n_bad += 1
                print(f"  [NG] {e.image_id}: {errs}")
        print(f"検証結果: {len(entries) - n_bad}/{len(entries)} 件がスキーマ準拠")


if __name__ == "__main__":
    main()
