# 10. 公式データセット(IESBench等)の取得・加工・実験手順

本ドキュメントは、本リポジトリのコード(合成生成器)だけで完結する実験から一歩進めて、
**公式IESBench、または同スキーマに変換した他の配布データセットを実際に使う場合**の
手順をまとめる。ここに書く手順・コマンドは、ユーザー自身の環境(egress制限のないGoogle
Colab等)で実行することを前提にしている。このリポジトリの開発セッション自体は
`huggingface.co` 等へのegressがブロックされているため、以下は**実データを使わず
公式スキーマ通りに模擬した最小データで動作確認済み**であり、実データそのものでの
実行確認はユーザー側で行う必要がある(詳細は5節「本セッションでの検証範囲」)。

## 1. 全体の流れ

```
[1] 取得         huggingface-cli download で公式データをダウンロード
      ↓
[2] 検証         iesbench_schema.py --validate でスキーマ準拠を確認
      ↓
[3a] そのまま使う   run_eval.py / train_safety_dpo.py 等に --dataset / --build-from-iesbench で直接渡す
[3b] 加工して使う   dataset_adapter.py で被覆×多様性×難易度バランスの取れた小サブセットに間引く
      ↓
[4] 実験         防御学習・評価パイプラインを実行
```

## 2. [1] 取得

### IESBench(本リポジトリが前提とする公式データセット)

```bash
pip install huggingface_hub
huggingface-cli download CSU-JPG/IESBench --repo-type dataset --local-dir ./iesbench_official
```

展開後のディレクトリには `annotation.json`(または類似の名前のアノテーションファイル)と
`img/` 以下に画像が入っている。正確なファイル名はダウンロード内容により変わりうるため、
本リポジトリのローダーは以下の候補名を優先順位付きで自動探索し、見つからない場合は
ディレクトリ直下を1〜2階層 `*.json`/`*.jsonl` で総当たりして自動判定する
(`src/dataset/iesbench_schema.py::_find_annotation_file()`):

`annotations.json`, `annotation.json`, `data.json`, `metadata.json`, `iesbench.json`,
`IESBench.json`, `test.json`, `eval.json`, `annotations.jsonl`, `data.jsonl`, `iesbench.jsonl`

見つからない/複数候補で曖昧な場合はエラーメッセージにディレクトリ内容が表示されるので、
`--load <dir>/<実際のファイル名>` のように直接ファイルを指定すればよい。

### IESBench以外の公式・配布データセットを使う場合

本リポジトリの内部表現(`IESBenchEntry`)は以下のフィールドを持つ:

| フィールド | 型 | 説明 |
|---|---|---|
| `image_id` | str | 一意なID |
| `image_path` / `image-path` | str | 画像への相対または絶対パス |
| `question` | str | 編集指示文(プロンプト) |
| `attributes` | list[str] | 対象属性 |
| `action` | str または list[str] | 編集アクション(公式IESBenchはリスト。複数アクションに該当しうる) |
| `category` | str または list[str] | 安全ポリシーカテゴリ(公式IESBenchは `"I1"`〜`"I15"` のリスト) |
| `rewrite` | str | 安全な書き換え案(存在する場合) |

他のデータセット(例: 他大学が配布するjailbreak画像編集ベンチマーク、または独自に
アノテーションした社内データ)を使いたい場合は、**上記のフィールド名に変換したJSON配列
またはJSONLに整形するだけ**で、本リポジトリの全パイプライン(検証・加工・評価・DPO学習)に
そのまま載せられる。`action`/`category` は単一文字列でもリストでもどちらでも受け付ける
(`src/dataset/iesbench_schema.py::_as_list()` が両対応)。

## 3. [2] 検証

```bash
python -m src.dataset.iesbench_schema --load ./iesbench_official --validate
```

- `category` が `"I1"`〜`"I15"` 形式(公式IESBenchのラベル)だと自動判定した場合、
  本リポジトリの合成ラベル体系(`SAFETY_POLICIES`/`EDIT_ACTIONS`)との照合はスキップし、
  「空でないか」等の構造チェックのみ行う(`--check-taxonomy auto`、既定)。
- 本リポジトリ自身の合成データセット(`build_dataset.py`が生成したもの)を検証したい場合は
  `--check-taxonomy on` で厳密な語彙照合を強制できる。

## 4. [3] 実データの使い方: そのまま使う vs 加工して使う

### 4.1 そのまま使う(全件を対象に実験する)

評価・DPO選好データ合成のどちらも、ディレクトリ/JSON/JSONLを直接指定できる
(内部で `load_entries()` を使うため、IESBench公式ディレクトリでも本リポジトリの合成
データでも同じインターフェース):

```bash
# 攻撃×防御の一括評価(ASR/HS/EV/HRR/FBR)
python -m src.eval.run_eval --dataset ./iesbench_official --compare-all --out outputs/eval_report.json

# 実データの question/rewrite から DPO選好ペア(chosen/rejected)を自動合成
python -m src.defense.train_safety_dpo --build-from-iesbench ./iesbench_official \
    --data outputs/dpo_preferences_from_official.jsonl --mock --epochs 5
```

### 4.2 加工して使う(被覆×多様性×難易度バランスを保ったまま間引く)

公式IESBenchは1,054件と決して巨大ではないが、`docs/09_resource_requirements.md` で
述べた計算資源制約(32GB VRAM級の環境での学習・評価の反復)を考えると、**15ポリシー×
アクション種別の被覆をできるだけ保ったまま、より小さいサブセットで素早く反復したい**
場面がある。これを行うのが新規モジュール `src/dataset/dataset_adapter.py`:

```bash
python -m src.dataset.dataset_adapter --source ./iesbench_official \
    --n-target 200 --out outputs/iesbench_subset_200.jsonl \
    --report-out outputs/iesbench_subset_200_report.json
```

出力される `iesbench_subset_200.jsonl` は元データの実エントリ(実画像パス・実プロンプト)
そのものを保持したサブセットで、`run_eval.py --dataset` や
`train_safety_dpo.py --build-from-iesbench` にそのまま渡せる。

## 5. 「1から作成」と「既存データセットから加工」の使い分け

本リポジトリのデータセット構築アルゴリズムは、目的の異なる2つのモジュールに分かれている。
どちらも被覆×多様性×難易度バランスの貪欲法(`src/dataset/coverage_optimizer.py::greedy_select()`、
劣モジュラ関数の貪欲最大化、Nemhauser–Wolsey–Fisher 1978 の近似保証)という同じコアエンジンを
共有しており、「候補プールの作り方」だけが異なる:

| | `src/dataset/dataset_optimizer.py`(1から作成) | `src/dataset/dataset_adapter.py`(既存データセットから加工) |
|---|---|---|
| 候補の出所 | `SAFETY_POLICIES × ATTRIBUTES × EDIT_ACTIONS` の直積(合成・架空) | `load_entries()` で読み込んだ実データセットの実エントリ |
| 出力される画像・プロンプト | プレースホルダー文言 + `typography_attack`等で新規レンダリング | 元データセットの実画像・実プロンプトをそのまま保持 |
| 外部データ依存 | なし(完全にゼロから生成可能) | あり(IESBench等、事前に取得済みのデータセットが必要) |
| 用途 | 実データが無い/取得できない環境でもパイプライン全体を検証したい | 実データはあるが、計算資源・時間の制約で件数を絞りたい/偏りを抑えたサブセットが欲しい |
| CLI | `python -m src.dataset.build_dataset --n-target 200` | `python -m src.dataset.dataset_adapter --source <dir> --n-target 200` |

`dataset_adapter.py` の難易度(difficulty)は人手評価が無い場合のプロキシとして
`lexicon_optimizer.py` の脅威語彙スコアを使っている(語彙的に明白な指示文ほど「易」と
みなす仮定)。実運用では実測ASRやより精緻な難易度指標に置き換えることを推奨する。

## 6. 本セッションでの検証範囲(重要な注意)

このリポジトリの開発・修正を行っているセッションは、組織のegressポリシーにより
`huggingface.co` 等へのアクセスがブロックされている(`docs/09_resource_requirements.md`
3.2節で確認済み)。そのため、本ドキュメントに記載した以下の項目は、
**公式IESBenchのスキーマを模擬した最小データ(実画像なし・実カテゴリコード"I1"〜"I15"
形式・複数カテゴリ/複数アクションを含む)で動作確認済み**であり、実データそのものでの
動作確認ではない:

- `iesbench_schema.py --validate` によるスキーマ検証(実データ形式を模擬)
- `dataset_adapter.py` による加工・選抜(120件の模擬データから40件を選抜し、
  被覆レポート・出力JSONLの内容を確認)
- `run_eval.py --compare-all` による評価(加工後サブセットに対しASR/HS/EV/HRR/FBRが
  クラッシュせず算出されることを確認)
- `train_safety_dpo.py --build-from-iesbench <dir> --mock` によるDPO選好データ自動合成
  + 学習(損失が減少することを確認)

**ユーザー側の環境(Colab等)で実際にIESBenchをダウンロードした後は、上記と全く同じ
コマンドがそのまま動く設計**になっているが、実データ特有の想定外のフィールド値
(未知のエンコーディング、欠損値等)が存在する可能性はゼロではない。実行してエラーが
出た場合はエラーメッセージ(特にどのフィールドで失敗したか)を共有してもらえれば
追加の修正を行う。
