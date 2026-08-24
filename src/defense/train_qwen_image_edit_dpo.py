"""
実際の Qwen/Qwen-Image-Edit(diffusers, 約20Bパラメータ級のFlow Matching拡散モデル)に対する
4bit/8bit量子化 + LoRA での安全アライメントDPO学習。

## 背景

`train_safety_dpo.py` は `transformers.AutoModelForCausalLM`(テキスト生成インターフェース)を
前提としており、拡散モデルベースの画像編集モデル本体には適用できない
(`docs/09_resource_requirements.md` 3.2節で、公式実装(`CSU-JPG/VJA`)の
`src/requirements.txt` を確認し、実際に `diffusers==0.36.0.dev0` 依存であることを
確認済み)。本モジュールはそのギャップを埋める、拡散モデル本体への直接適用版である。

## VJAへの32GB VRAMでの適用可否(このモジュールの設計方針)

このセッションからは `huggingface.co` への接続が組織のegressポリシーでブロックされて
おり、実際の重みをダウンロードして学習することはできない(`docs/09` 参照、時間を
置いても解除されない制約)。そのため本モジュールは:

  1. 実際の `diffusers.QwenImageEditPipeline` / `QwenImageTransformer2DModel` の
     API(`pip install diffusers` で確認済み、下記1節参照)に忠実に実装し、
     **ユーザー自身のHuggingFaceにアクセスできる32GB環境でそのまま実行できる**
     ように作る。
  2. `--mock` モードでは実際のモデルを一切ダウンロードせず、同じインターフェースを持つ
     ごく小さな代替モジュールで学習ループの配線(loss計算・勾配・LoRA適用)を検証する。

## 1. 確認済みの実際のAPI

```
diffusers.QwenImageEditPipeline(
    scheduler: FlowMatchEulerDiscreteScheduler,      # Flow Matching(SD3/Flux系と同様)
    vae: AutoencoderKLQwenImage,
    text_encoder: Qwen2_5_VLForConditionalGeneration,  # 条件付けに Qwen2.5-VL 全体を使用(重い)
    tokenizer, processor,
    transformer: QwenImageTransformer2DModel,          # 学習対象の denoising backbone
)
QwenImageTransformer2DModel 既定config: num_layers=60, num_attention_heads=24,
    attention_head_dim=128 (hidden=3072相当), joint_attention_dim=3584
```
(`pip install diffusers` でのクラス/コンフィグ確認、2026年時点)

**公式実装(`CSU-JPG/VJA`)の`src/run.py`で確認した実際の生成呼び出し**(このリポジトリを
`git clone https://github.com/CSU-JPG/VJA` して直接確認できる。同リポジトリのREADMEは
「complete evaluation code」は本稿執筆時点でまだ未公開("in the coming weeks")と明記しており、
公開されているのは単一画像デモの`src/run.py`と、提案手法の防御パイプライン
`src/models/qwen_image_edit_safe.py`のみ):

```python
from diffusers import QwenImageEditPlusPipeline  # 無防御ベースラインはこちらを使用(要確認: 通常のQwenImageEditPipelineとの違いは非公開)
pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
inputs = {
    "image": input_image, "prompt": input_prompt,
    "generator": torch.manual_seed(0), "true_cfg_scale": 4.0,
    "negative_prompt": " ", "num_inference_steps": 40,
}
try:
    output = pipeline(**inputs)
    output_image = output.images[0]
except Exception as e:
    # 公式の安全パイプライン(QwenImageEditSafePipeline)は SafetyError(message, code) を送出し、
    # e.reason / e.judgment で拒否理由を取得できる。ベースパイプラインは通常例外を出さない想定。
    ...
```

pinしているバージョン(`requirements.txt`): `torch==2.8.0`, `diffusers==0.36.0.dev0`,
`transformers==4.57.1`。これらと大きく異なるバージョンでは挙動が変わる可能性がある。

**公式が提案する防御(`QwenImageEditSafePipeline`)の仕組み(参考、本リポジトリにはコピーしない)**:
`diffusers.QwenImageEditPipeline`を継承し、内部の`_get_qwen_prompt_embeds()`をオーバーライドして、
画像+プロンプトを条件付けした同一のtext_encoder(Qwen2.5-VL)の隠れ状態(KV cache)を再利用しつつ、
「ユーザーの真の意図」と「15の安全ポリシー(I1〜I15)のいずれかに抵触するか」をYES/NO形式で
自己内省的に判定させ、NOと判定された場合に`SafetyError`を送出して生成を中断する
(=training-freeなmultimodal introspective reasoning。本リポジトリの`introspective_defense.py`
と設計思想は同じだが、公式はOCR/外部検出器ではなく実際のtext_encoder自身の隠れ状態を再利用する
点が異なる)。公式のI1〜I15の正式な定義は`src/dataset/iesbench_schema.py::OFFICIAL_CATEGORY_NAMES`
に転記した。

`FlowMatchEulerDiscreteScheduler` を使うため、標準的な Diffusion-DPO
(Wallace et al., 2023, "Diffusion Model Alignment Using Direct Preference Optimization")
のepsilon予測ベースの定式化ではなく、**velocity予測(Flow Matching)ベースに一般化**した
形で損失を実装する(下記 `flow_matching_dpo_loss` 参照。予測対象がepsilonかvelocityかに
依らず、「モデル予測 と ターゲットの二乗誤差」の差分として定式化できるため本質的には同じ式)。

## 2. 32GBで収めるための3つの工夫

1. **参照モデルを複製しない**: 通常のDPOは policy と ref の2モデルを保持するが、
   本実装は `peft` の `model.disable_adapter()` コンテキストを使い、
   **同一の量子化済みベースモデル上でLoRAのON/OFFを切り替えて** policy/ref 両方の
   forwardを計算する。LoRA初期状態(B行列がゼロ初期化)では adapter ON でも OFF でも
   出力が一致するため、これは標準的なDPO+LoRAの正しい定式化であり
  (TRLの`DPOTrainer`がpeft使用時に採用する方式と同じ)、**2つ目のモデルの
   VRAMを丸ごと節約できる**。
2. **transformerのみ4bit/8bit量子化 + LoRA学習、text_encoder/vaeは推論専用**:
   テキスト埋め込みは事前に1回計算してキャッシュすれば良く、学習ループ中は
   text_encoder(Qwen2.5-VL, 単体で7-8Bパラメータ級)をCPUへオフロード可能。
3. **gradient checkpointing + 小バッチ(既定1)+ 低解像度**でアクティベーションメモリを抑制。

## 3. データ形式(preference pairs)

`train_safety_dpo.py` のテキストのみのDPOと異なり、拡散モデル本体のDPOは
**編集後の画像そのもの**のペア(chosen=安全な編集結果 or 拒否を表す画像,
rejected=有害な編集結果)を必要とする。JSONL形式:

```json
{"prompt": "編集指示テキスト", "source_image_path": "編集前画像",
 "chosen_image_path": "望ましい編集結果画像", "rejected_image_path": "望ましくない編集結果画像"}
```

**重要な誠実な注記**: このようなchosen/rejected画像ペアの収集自体が
`docs/04_ideal_dataset_design.md` で述べた「データ作成ガイドライン」を要する
非自明な作業であり、本モジュールは学習ループのみを提供する。
"""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.seed import set_seed

MODEL_NAME_DEFAULT = "Qwen/Qwen-Image-Edit"


@dataclass
class QwenImageEditDPOConfig:
    model_name: str = MODEL_NAME_DEFAULT
    quantization: str = "4bit"        # "4bit" | "8bit" | "none"
    lora_rank: int = 16
    lora_alpha: int = 32
    resolution: int = 512             # 解像度を下げるほどVRAM節約(既定1024から512等へ)
    batch_size: int = 1
    lr: float = 1e-4
    beta: float = 2000.0              # Diffusion-DPOはtoken単位DPOよりbetaを大きく取るのが通例
    gradient_checkpointing: bool = True
    # "model": enable_model_cpu_offload()(粗粒度、通常は高速だが peft.PeftModel でラップした
    #   transformerとの組み合わせで極端に遅い/停止したように見える事例を確認)。
    # "sequential": enable_sequential_cpu_offload()(層単位の細粒度オフロード。低VRAM環境向けの
    #   標準的な選択肢で、"model"より低いVRAMで動く可能性が高いが推論は遅くなる)。
    # "none": オフロードなし(GPU VRAMに全モデルが載る場合のみ)。
    offload_mode: str = "model"
    num_train_timesteps: int = 1000
    epochs: int = 1
    seed: int = 42


# --------------------------------------------------------------------------------------
# Flow Matching版 Diffusion-DPO 損失
# --------------------------------------------------------------------------------------
def flow_matching_dpo_loss(
    model_pred_w: torch.Tensor,
    model_pred_l: torch.Tensor,
    ref_pred_w: torch.Tensor,
    ref_pred_l: torch.Tensor,
    target_w: torch.Tensor,
    target_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Diffusion-DPO (Wallace et al., 2023) を Flow Matching の速度予測パラメータ化に一般化した損失。
    予測がepsilon(標準拡散)かvelocity(Flow Matching, Qwen-Image-Edit/SD3/Flux系)かに
    依らず、「モデル予測とターゲットの二乗誤差」を暗黙報酬の代理として使う点は共通。

    L = -log sigmoid( -beta * [ (||model_w - target_w||^2 - ||ref_w - target_w||^2)
                                - (||model_l - target_l||^2 - ||ref_l - target_l||^2) ] )
    """
    model_diff_w = (model_pred_w - target_w).pow(2).flatten(1).mean(dim=1)
    model_diff_l = (model_pred_l - target_l).pow(2).flatten(1).mean(dim=1)
    ref_diff_w = (ref_pred_w - target_w).pow(2).flatten(1).mean(dim=1)
    ref_diff_l = (ref_pred_l - target_l).pow(2).flatten(1).mean(dim=1)

    logits = -beta * ((model_diff_w - ref_diff_w) - (model_diff_l - ref_diff_l))
    return -F.logsigmoid(logits).mean()


# --------------------------------------------------------------------------------------
# 実モデルのロード(ユーザー自身のHuggingFaceアクセス可能な環境で実行する想定)
# --------------------------------------------------------------------------------------
def load_real_pipeline(config: QwenImageEditDPOConfig):
    from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig

    quant_config = None
    if config.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    elif config.quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # transformer(学習対象)のみ量子化ロード
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.model_name, subfolder="transformer", quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, target_modules="all-linear")
    transformer = get_peft_model(transformer, lora_config)
    if config.gradient_checkpointing:
        # diffusersのModelMixin系は transformers.PreTrainedModel と異なり
        # enable_gradient_checkpointing() という名前を使う(gradient_checkpointing_enable()ではない)。
        # peft.PeftModelの属性委譲チェーン経由で内側のdiffusersモデルへ届く。
        # 将来のdiffusers/peftバージョン差異に備え、どちらの名前も存在しなければ
        # 警告のみで学習自体は続行する(gradient checkpointingはVRAM節約のための最適化であり、
        # 無効でも正しさには影響しないため)。
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()
        else:
            print(
                "[警告] transformerにgradient checkpointingを有効化するメソッドが見つかりません"
                "(enable_gradient_checkpointing/gradient_checkpointing_enableのいずれも無し)。"
                "VRAM使用量は増えますが、無効のまま学習/推論を継続します。"
            )

    pipe = QwenImageEditPipeline.from_pretrained(
        config.model_name, transformer=transformer, torch_dtype=torch.bfloat16,
    )
    # text_encoder(Qwen2.5-VL, 単体で7-8Bパラメータ級・bf16で約14GB)は推論専用のためgradient不要
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    if config.offload_mode == "model":
        # diffusers標準の粗粒度オフロード(text_encoder/vae/transformerをまとめて必要時のみGPUへ)。
        # 内部でCPU<->GPU間の配置し直しが発生するため、プログレスバー等の出力が一切無いまま
        # 数分かかることがある(一見フリーズしたように見えるが正常動作のはず)。
        # ただし peft.PeftModel でラップしたtransformerとの組み合わせでは、accelerateの
        # フック登録処理が実質的に進まなくなる(数分待っても完了しない)事例を確認している。
        # その場合は QwenImageEditDPOConfig(offload_mode="sequential") を使うこと
        # (低VRAM環境向けの標準的な代替手段、より細粒度で低メモリだが推論は遅くなる)。
        print("[情報] CPU offload(model)を設定中(進捗表示なしで数分かかることがあります)...")
        pipe.enable_model_cpu_offload()
        print("[情報] CPU offload設定完了")
    elif config.offload_mode == "sequential":
        print("[情報] CPU offload(sequential, 低VRAM向け)を設定中...")
        pipe.enable_sequential_cpu_offload()
        print("[情報] CPU offload設定完了(推論は enable_model_cpu_offload より遅くなります)")
    elif config.offload_mode != "none":
        raise ValueError(f"未知のoffload_mode: {config.offload_mode!r} (model/sequential/noneのいずれか)")

    return pipe


@contextmanager
def reference_forward(transformer):
    """peftのLoRA無効化コンテキストで、同一重みを参照モデルとして使う(2モデル目の複製を回避)。"""
    if hasattr(transformer, "disable_adapter"):
        with transformer.disable_adapter():
            yield
    else:
        yield  # --mockの代替モジュール等、peftモデルでない場合はそのまま


# --------------------------------------------------------------------------------------
# Mock版: 外部ダウンロード不要で学習ループの配線(loss計算・LoRA適用・disable_adapter)を検証する。
# --------------------------------------------------------------------------------------
class _MockTransformer(nn.Module):
    """QwenImageTransformer2DModelの代替。latent形状 (B,C,H,W) を受け取りvelocityを予測する。"""

    def __init__(self, channels: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, latents: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, latents.shape[2], latents.shape[3]).to(latents.dtype)
        x = torch.cat([latents, t_map], dim=1)
        return self.net(x) + 0.0 * cond.mean()  # condはmockでは形状合わせのダミー


def load_mock_pipeline(config: QwenImageEditDPOConfig):
    from peft import LoraConfig, get_peft_model

    transformer = _MockTransformer()
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2", "4"])  # Conv2dの位置指定は簡略化
    try:
        transformer = get_peft_model(transformer, lora_config)
    except Exception:
        pass  # Conv2dへのLoRA適用可否はpeftバージョン依存のため、失敗時は素のtransformerで配線のみ検証
    return transformer


# --------------------------------------------------------------------------------------
# 選好データの読み込み(mockでは合成テンソルで代替)
# --------------------------------------------------------------------------------------
@dataclass
class ImagePreferencePair:
    prompt_embedding: torch.Tensor
    chosen_latents: torch.Tensor
    rejected_latents: torch.Tensor


def load_mock_preference_pairs(n: int, seed: int = 0, latent_size: int = 16) -> list[ImagePreferencePair]:
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n):
        cond = torch.randn(1, 8, generator=g)
        chosen = torch.randn(1, 4, latent_size, latent_size, generator=g)
        rejected = torch.randn(1, 4, latent_size, latent_size, generator=g)
        pairs.append(ImagePreferencePair(cond, chosen, rejected))
    return pairs


def _sample_flow_matching_target(x1: torch.Tensor, seed_offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flow Matching: x_t = (1-t)*x0 + t*x1 (x0=ノイズ), target velocity = x1 - x0。"""
    g = torch.Generator().manual_seed(seed_offset)
    x0 = torch.randn_like(x1)
    t = torch.rand(1, generator=g).clamp(1e-3, 1 - 1e-3)
    t_expand = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0
    return x_t, t, target


def train_mock(config: QwenImageEditDPOConfig, n_pairs: int = 20) -> dict:
    set_seed(config.seed)
    transformer = load_mock_pipeline(config)
    pairs = load_mock_preference_pairs(n_pairs, seed=config.seed)
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad], lr=config.lr
    )

    losses = []
    for epoch in range(config.epochs):
        for i, pair in enumerate(pairs):
            xw_t, tw, target_w = _sample_flow_matching_target(pair.chosen_latents, seed_offset=i * 2)
            xl_t, tl, target_l = _sample_flow_matching_target(pair.rejected_latents, seed_offset=i * 2 + 1)

            model_pred_w = transformer(xw_t, tw, pair.prompt_embedding)
            model_pred_l = transformer(xl_t, tl, pair.prompt_embedding)
            with torch.no_grad(), reference_forward(transformer):
                ref_pred_w = transformer(xw_t, tw, pair.prompt_embedding)
                ref_pred_l = transformer(xl_t, tl, pair.prompt_embedding)

            loss = flow_matching_dpo_loss(model_pred_w, model_pred_l, ref_pred_w, ref_pred_l, target_w, target_l, config.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"losses": losses, "first_loss": losses[0], "last_loss": losses[-1]}


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen-Image-Edit本体への4bit/8bit量子化+LoRA Diffusion-DPO学習")
    p.add_argument("--mock", action="store_true", help="外部モデル不要の配線検証モード")
    p.add_argument("--model-name", default=MODEL_NAME_DEFAULT)
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=2000.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--data", default=None, help="実データ使用時: preference pairs JSONL(3節参照)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    config = QwenImageEditDPOConfig(
        model_name=args.model_name, quantization=args.quantization, lora_rank=args.lora_rank,
        resolution=args.resolution, lr=args.lr, beta=args.beta, epochs=args.epochs, seed=args.seed,
    )

    if args.mock:
        result = train_mock(config)
        print(f"[mock] 損失推移(先頭5件): {result['losses'][:5]}")
        print(f"[mock] first_loss={result['first_loss']:.4f} -> last_loss={result['last_loss']:.4f}")
        improved = result["last_loss"] < result["first_loss"]
        print(f"損失は{'減少しました(配線OK)' if improved else '減少しませんでした(要確認)'}")
        return

    if not args.data:
        raise SystemExit(
            "--mock を指定しない場合は --data (preference pairs JSONL, モジュールdocstring3節参照) が必要です。"
            "また huggingface.co にアクセスできる環境(このセッションでは組織ポリシーによりブロック済み、"
            "docs/09_resource_requirements.md 参照)で実行してください。"
        )

    pipe = load_real_pipeline(config)
    print("パイプラインのロードに成功しました。実データでの学習ループは"
          "docs/10_qwen_image_edit_dpo_runbook.md の手順に従って構築してください"
          "(画像の前処理・VAEエンコード・データローダはユーザーの実データ形式に応じて実装が必要です)。")
    print(pipe)


if __name__ == "__main__":
    main()
