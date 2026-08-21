# Visual-to-Visual攻撃の拡張: 新手法・新シナリオとその実装

画像編集モデルに限定せず、「意図を非テキストの視覚情報だけで伝える」という
VJAの本質的な攻撃構造がどこまで一般化するかを検討し、2つの具体的シナリオを
実装・実証した。

## 1. 新しい攻撃手法(エンコーディングチャネル)の見取り図

VJAの矢印・丸囲み以外にも、以下のようなチャネルが考えられる。それぞれについて
Web検索で確認できた関連研究を参考文献列に付す(注記が無い項目は本リポジトリ独自の
着想であり、確認できた既存文献は無い)。

| # | 手法 | メカニズム | 本ドキュメントでの実装状況 | 関連研究 |
|---|---|---|---|---|
| 1 | 空間配置セマンティクス | オブジェクトの相対位置そのものが指示を表す | 未実装(今後の拡張候補) | — |
| 2 | マルチ画像分割(視覚的サラミスライシング) | 複数画像に指示を分割し、モデルのマルチ画像文脈推論だけが全体を再構成する | 未実装。`SessionRiskTracker`が部分的な受け皿 | [SIVA (arXiv:2602.08136)](https://arxiv.org/pdf/2602.08136), [MIDAS (arXiv:2603.00565)](https://arxiv.org/pdf/2603.00565), [Multi-Modal Linkage (arXiv:2412.00473)](https://arxiv.org/abs/2412.00473), [Jailbreak in Pieces (arXiv:2307.14539)](https://arxiv.org/pdf/2307.14539) |
| **3** | **時間軸エンコーディング(動画)** | 単一フレームは無害、物体の移動軌跡が指示を描く | **本ドキュメントで実装(2節)** | [TempJail (arXiv:2608.19737)](https://arxiv.org/abs/2608.19737), [Two Frames Matter (arXiv:2603.07028)](https://arxiv.org/html/2603.07028v1), [VideoJail-Pro (OpenReview)](https://openreview.net/pdf?id=fSAIDcPduZ), [Multi-Clip Video Jailbreak (arXiv:2606.02111)](https://arxiv.org/html/2606.02111v1) |
| 4 | 図式・アイコノグラフィ言語 | 化学装置図/配線図等の図表記法を悪用 | 未実装 | — |
| 5 | プライベート視覚コード/ステガノグラフィ | 会話前半で私製の図形→意味対応を"教える"、または画像に不可視エンコード | 未実装 | [Invisible Injections (arXiv:2507.22304)](https://arxiv.org/abs/2507.22304), [Hiding in Plain Sight (arXiv:2505.16765)](https://arxiv.org/pdf/2505.16765) |
| 6 | 学習済みトリガーパッチ/バックドア | ブラックボックス探索で最適化したステッカー状パッチ | `adaptive_attack_optimizer.py`と着想は共通、パッチ単位の最適化は未実装 | [Test-Time Backdoor on MLLMs (arXiv:2402.08577)](https://arxiv.org/pdf/2402.08577), [BackdoorVLM (arXiv:2511.18921)](https://arxiv.org/html/2511.18921) |
| 7 | 姿勢・表情の参照画像 | 参照ポーズ画像自体が有害行為の再現指示になる | 未実装 | — |
| 8 | 深度・陰影エンコーディング | 3D/深度認識モデル向けの影パターン | 未実装 | — |
| 9 | 音声-映像相関チャネル | 個別に無害な映像と音声の組み合わせのみが指示になる | 未実装 | — |
| 10 | 物理世界マーカー | 印刷物・環境内オブジェクト配置(ロボット/AR、VLA基盤モデル) | 未実装(高リスク・物理安全領域のため概念レベルに留める) | [VLA Adversarial Vulnerabilities (arXiv:2411.13587)](https://arxiv.org/abs/2411.13587), [Partially Observable Patch Attacks on VLA (arXiv:2606.03556)](https://arxiv.org/html/2606.03556v1), [VLA-Hijack (arXiv:2605.28083)](https://arxiv.org/pdf/2605.28083), [When Robots Obey the Patch (arXiv:2511.21192)](https://arxiv.org/abs/2511.21192), [VLA Safety Survey (arXiv:2604.23775)](https://arxiv.org/html/2604.23775v1) |

## 2. 実装1: 時系列軌跡エンコーディング攻撃(技術#3, シナリオS3)

### 2.1 脅威モデル

動画理解モデルに対し、**単一フレームは無害**(小さな点がわずかに動いているだけ)
だが、フレーム間の移動軌跡そのものが指示を描く攻撃。フレームサンプリング型の
モデレーション(代表フレームだけを検査する方式)や、静止画ベースの安全フィルタ
(本リポジトリの`mark_detector.py`を含む)を原理的に回避する。

### 2.2 実装

- `src/attack/temporal_trajectory_attack.py`: 5種類の「意味のある軌跡」
  (arrow_sweep, circle_then_tap, zigzag_alert, underline_sweep)と
  2種類の「良性な動き」(random_noise, static)を、合成シーン上のマーカー移動
  として生成。単一フレームを見ても両者は識別不能([ここに例]参照)。
- `src/attack/trajectory_variant_generator.py`: バリアント一括生成(270系列)
- `src/defense/trajectory_detector.py`:
  - `naive_single_frame_flag()`: 最も寛容な単一フレーム基準(マーカー存在の有無のみ)
  - `TrajectoryClassifierMLP`: フレーム間のマーカー座標系列から
    正規化座標・総移動距離・正味変位・累積回頭角(曲率)を特徴量化し、
    系列全体を見て判別する軽量MLP

### 2.3 実測結果(270系列、80/20分割)

| 検出方式 | 指標 |
|---|---|
| 素朴な単一フレーム基準 | 攻撃(意味のある軌跡)検知率 **1.000** / 良性動作での誤検知率 **1.000** (差 = 0.000) |
| 系列認識MLP(`TrajectoryClassifierMLP`) | 検証is_semantic精度 **0.907** |

単一フレーム基準は攻撃・良性のどちらも100%「マーカーあり」としか言えず、
**判別力が完全にゼロ**であることが数値で確認された。これは
`docs/07_vja_faithful_defense_gap.md`で示したOCRベース防御のVJA型攻撃に対する
盲点(検知率0%)と同型の構造的弱点であり、「静止画のみを検査する防御は
本質的に時間軸に沿った攻撃を検出できない」という一般原則を裏付ける。

再現コマンド:

```bash
python -m src.attack.trajectory_variant_generator --out data/sample/trajectories
python -m src.defense.trajectory_detector --data data/sample/trajectories/manifest.jsonl \
    --save outputs/trajectory_detector.pt
```

### 2.4 関連研究

動画/時間軸方向のjailbreakは静止画攻撃と比べ研究が薄い領域だが、2026年に入り
複数の関連研究が発表されている。

- **[TempJail: Temporal Jailbreak Attack against Large Vision-Language Models via Subtitle Scheduling](https://arxiv.org/abs/2608.19737)**
  (2026)。ブラックボックスで、対話調の字幕列を構築しその**時間的なスケジューリング
  (提示タイミング・長さ)** を最適化してLVLMの安全性を回避する。「情報の意味だけでなく
  提示タイミングも脱獄成功率を左右する」という知見は、本リポジトリの
  「単一フレームでは無害/系列全体で意味が生じる」という主張と同じ問題意識を
  異なる角度(字幕の時間割当 vs マーカー軌跡)から扱っている。
- **[Two Frames Matter: A Temporal Attack for Text-to-Video Model Jailbreaking](https://arxiv.org/html/2603.07028v1)**
  (2026)。T2V(動画生成)モデルに対し、開始・終了フレームなど疎な境界条件のみを
  指定し、中間の遷移をモデル自身に補間させることで有害な中間フレームを誘発する
  「時間的軌跡補間の脆弱性」を指摘。本リポジトリの攻撃(理解モデル向け・完全な
  軌跡を明示的に与える)とは逆方向(生成モデル向け・軌跡の一部だけを与えて残りを
  補間させる)だが、「時間軸上のギャップに攻撃面がある」という点で構造的に対応する。
- **[VideoJail-Pro (フレーム再配列によるjailbreak, OpenReview)](https://openreview.net/pdf?id=fSAIDcPduZ)**、
  **[Jailbreaking MLLMs using Multi-Clip Video (arXiv:2606.02111)](https://arxiv.org/html/2606.02111v1)**。
  フレーム順序の並べ替えや複数クリップへの分割によって単一クリップ/フレーム検査を
  回避する点で、本リポジトリの「単一フレーム基準は判別力ゼロ」という実測結果
  (2.3節)を裏付ける独立の証左になっている。

**本実装との違い**: 上記はいずれもテキスト応答生成側(LVLM/T2V)の脆弱性検証が
主眼であり、**検出器側**(単一フレーム基準 vs 軌跡認識MLPの定量比較、
`naive_single_frame_flag()`の判別力がゼロであることの直接測定)を提示した
研究は本ドキュメント作成時点では見当たらなかった。この検出器側の実測比較
(2.3節)が本実装の独自の貢献点である。

## 3. 実装2: GUI/エージェント・ハイジャック攻撃(シナリオS1)

### 3.1 脅威モデル

ブラウザ/OS操作AIエージェントが受け取るスクリーンショットに、正規のUIには
存在しない**偽のダイアログ/ボタン**を描画し、エージェントに「システムからの
指示」であるかのように誤認させ、削除・送金・権限付与等の行動を誘発する
("visual-to-action" injection)。テキストチャット欄へのプロンプト注入とは
異なるチャネルであり、既存のテキストベースのプロンプトインジェクション検知
(ユーザー入力テキストのみを検査する方式)では原理的に検知できない。

### 3.2 実装

- `src/attack/ui_injection_attack.py`: 完全に汎用的な図形(実在のOS/ブラウザの
  外観は模倣しない)でスクリーンショット様の合成画像を生成し、4種類のスタイル
  (banner_top, modal_center, toast_bottom_right, inline_button)の偽ダイアログを
  重畳する。ボタンラベルは常に一般的なプレースホルダー("Confirm"/"Proceed"等)。
- `src/attack/ui_injection_variant_generator.py`: バリアント一括生成
  (背景プール再利用で交絡因子を統制)
- `src/defense/ui_injection_detector.py`: RGB軽量CNN(`mark_detector.py`と
  同じBatchNorm+AdaptiveMaxPool構成)で偽ダイアログの有無を判定

### 3.3 実測結果

72バリアント(注入あり36・クリーン36)で学習したところ、**検証精度100%**
(10エポック以内に到達)。ダイアログはマークより占有面積が大きく強い信号のため、
`mark_detector.py`のスパース信号問題(6.4.3節参照、`docs/07`)は生じなかった。

再現コマンド:

```bash
python -m src.attack.ui_injection_variant_generator --out data/sample/ui_injection
python -m src.defense.ui_injection_detector --data data/sample/ui_injection/manifest.jsonl \
    --save outputs/ui_injection_detector.pt
```

### 3.4 関連研究

GUI/コンピュータ操作エージェントに対する視覚的注入攻撃は、エージェント化の
急速な進展を受けて2024年後半から2026年にかけて急速に研究が進んでいる領域である。

- **[Attacking Vision-Language Computer Agents via Pop-ups (Wu et al., ACL 2025, arXiv:2411.02391)](https://arxiv.org/abs/2411.02391)**
  ([コード: SALT-NLP/PopupAttack](https://github.com/SALT-NLP/PopupAttack))。
  本ドキュメントの実装に**最も近い先行研究**。人間なら無視できる広告風ポップアップに
  "Attention Hook"(ユーザーの元クエリの要約等)を仕込み、GPT-4系・Gemini・Claude
  等のフロンティアVLMエージェントを誘導する。GT/HKU/Stanfordの共同研究で、
  ASR平均86%、タスク成功率を47%低下させたと報告。本リポジトリの
  `ui_injection_attack.py`(banner_top/modal_center/toast_bottom_right/inline_button
  の4スタイル)は、このPopupAttackが示した「エージェントは人間なら無視するUI要素に
  過剰に反応する」という脆弱性クラスを、実在ブランド/OS模倣を避けた完全合成の
  スキーマとして再現したものと位置づけられる。
- **[MIRAGE: Context-Aware Prompt Injection against Mobile GUI Agents (arXiv:2605.28116)](https://arxiv.org/abs/2605.28116)**
  (2026)。ユーザー生成コンテンツ領域(レビュー欄等)に文脈適応型ペイロードを
  自然に溶け込ませる手法。1,111サンプル・10アプリ・11攻撃意図のベンチマークで
  ASR 23-30%を報告。本リポジトリの偽ダイアログ(構造的に異質なUI要素の注入)とは
  異なり、**正規のUI領域そのものを悪用する**点が対照的。
- **[AgentRAE: Notification-based Visual Backdoors against Screenshot-based Mobile GUI Agents (arXiv:2603.23007)](https://arxiv.org/pdf/2603.23007)**。
  通知/ダイアログへの誘目性UI要素埋め込みでエージェントを不正なポップアップ
  クリックに誘導する点で、本リポジトリの脅威モデルと直接的に一致する。
- **[MIRAGE: Stealthy Visual Prompt Injection for Web Agents (arXiv:2606.20717)](https://arxiv.org/html/2606.20717)**
  (同名だが別チーム・別手法)。拡散モデルによる制約付き視覚合成とスパースな
  暗ピクセル摂動を組み合わせ、空間制約下で次アクションを乗っ取る、より
  高度で検知回避に最適化された手法。本リポジトリの`obfuscation_opacity`
  パラメータ(ダイアログを背景に溶け込ませる)は、この方向性の簡易版に相当する。

**本実装との違い**: PopupAttack等は実際のエージェント実行環境(ブラウザ/OS
シミュレータ)に対する攻撃成功率(ASR)を測定するのに対し、本リポジトリは
**静止画検出器での検知精度**(3.3節、検証精度100%)に焦点を当てている。
実運用では両者は補完関係にあり、検出器(本リポジトリ)は「エージェントに
画像を渡す前段」で偽UIをフィルタする防御レイヤーとして、PopupAttack等が
示すエージェント自体の脆弱性を軽減する位置づけになる。

## 4. この2実装から得られる一般的な教訓

1. **モダリティ非対称性はVJA固有の問題ではなく構造的な問題である**。
   「静止画・単一フレーム・テキストのみを検査する防御は、その検査対象外の
   次元(時間軸、UI構造、空間配置等)に意図をエンコードする攻撃には
   原理的に無力」という一般原則が、非テキスト視覚指示(docs/07)と
   時系列軌跡(本docs)の両方で数値的に確認された。
2. **検出器は攻撃のエンコーディング次元に合わせて設計する必要がある**。
   静止画CNN(mark_detector, ui_injection_detector)は空間パターンには強いが
   時間パターンには無力であり、逆に軌跡MLPは時間パターンに強いが単一フレームの
   空間的な悪意(例: 有害な静止画そのもの)は見ない。**単一の検出器で
   全エンコーディング次元をカバーすることはできず、`unified_defense_pipeline.py`
   のような多層防御に、次元ごとの専用検出器を追加していくアプローチが必要**。
3. **エージェント化により被害の質が変わる**(GUI注入の事例)。画像編集の
   誤動作は「不適切な画像が生成される」だが、エージェント文脈でのUI注入は
   「意図しない実世界の行動(送金・削除等)が実行される」に直結し、
   リスクの深刻度が一段階上がる。

## 5. 今後の拡張ポイント

- `unified_defense_pipeline.py` に `trajectory_detector`/`ui_injection_detector` を
  追加レイヤーとして統合する
- マルチ画像分割攻撃(#2)を`SessionRiskTracker`と組み合わせて実装する。
  実装時は [SIVA](https://arxiv.org/pdf/2602.08136) の「安全アライメントは単一画像
  にしか行われていない」という指摘と、その対策として提案されている選好最適化への
  分割画像の組み込みを参考にできる
- 学習済みトリガーパッチ(#6)を`adaptive_attack_optimizer.py`のピクセル最適化
  版として実装し、`immune_memory_defense.py`の記憶登録元にする。
  [BackdoorVLM](https://arxiv.org/html/2511.18921) 等のベンチマーク設計を
  参考に、評価プロトコルを揃えられる
- ロボティクス/VLA領域への展開(#10)は、`docs/01_threat_scenarios.md`で
  「物理世界マーカー」として概念化した脅威が既に活発な研究領域
  ([VLA-Hijack](https://arxiv.org/pdf/2605.28083), [When Robots Obey the Patch](https://arxiv.org/abs/2511.21192))
  であることが確認できたため、本リポジトリでの実装は物理的安全性への配慮から
  概念整理に留め、実攻撃コードの構築は行わない方針を維持する
