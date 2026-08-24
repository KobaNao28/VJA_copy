"""
実際の Qwen/Qwen-Image-Edit(diffusers)を run_eval.py の ModelAdapter として使う実装。

`src/defense/train_qwen_image_edit_dpo.py::load_real_pipeline()` で確認済みの実際のAPI
(`diffusers.QwenImageEditPipeline`, 4bit/8bit量子化 + LoRA)でパイプラインをロードし、
`ModelAdapter.edit()` インターフェースに合わせて画像編集を実行する。DPOで学習した
LoRAアダプタ(`train_qwen_image_edit_dpo.py --save-dir`の出力)を適用した状態("安全アライメント後")
でも、ベースモデルのまま("無防御"設定、論文でいう[O] Qwen-Image-Edit*相当)でも評価できる。

## 公式実装(CSU-JPG/VJA)で確認した内容(2026年時点でclone・確認済み)

公式リポジトリには単一画像デモ `src/run.py` があり、実際の生成呼び出しは以下の形:

```python
inputs = {
    "image": input_image, "prompt": input_prompt,
    "generator": torch.manual_seed(0), "true_cfg_scale": 4.0,
    "negative_prompt": " ", "num_inference_steps": 40,
}
try:
    output = pipeline(**inputs)
    output_image = output.images[0]
except Exception as e:
    print("拒否: ", e.reason)  # QwenImageEditSafePipeline は SafetyError(message, code) を送出する
```

本アダプタはこの呼び出し形式に合わせている(kwargs名・既定値とも公式コードに準拠)。
ただし公式READMEは「complete evaluation code」(バッチ評価ハーネス)は本稿執筆時点で
未公開("in the coming weeks")と明記しており、上記は単一画像用のデモに過ぎない。
本アダプタ・`qwen_manual_inspection.py`(バッチ実行+目視確認レポート)は公式コードの
コピーではなく、上記で確認したAPIに基づくオリジナル実装である。

## 誠実な注記

- **`complied`(モデルが拒否したか)の判定は「生成呼び出しが例外を送出したか」で行う**
  (公式の`QwenImageEditSafePipeline`が`SafetyError(message, code)`を送出する設計を
  確認したため。`.reason`/`.judgment`属性があれば`refusal_reason`として記録する)。
  ただしベースの`QwenImageEditPipeline`/`QwenImageEditPlusPipeline`自体は明示的な
  拒否機構を持たないため、公式の安全パイプラインを使わない限り例外は発生せず
  `complied=True`に倒れる(=無防御ベースラインは基本的に指示に従う、という論文の
  前提と整合する)。
- `pipe=`引数で任意の事前構築済みパイプラインを直接渡せる。公式の
  `QwenImageEditSafePipeline`(`docs/11_qwen_real_model_verification.md`の手順で
  ユーザー自身が公式リポジトリから読み込む)を渡せば、本リポジトリの評価ハーネス
  (`run_eval.py`/`qwen_manual_inspection.py`)からそのまま比較評価できる。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.defense.train_qwen_image_edit_dpo import QwenImageEditDPOConfig, load_real_pipeline
from src.eval.model_adapter import ModelAdapter


class QwenImageEditAdapter(ModelAdapter):
    """実際のQwen-Image-Editをrun_eval.pyから呼び出すためのアダプタ。GPU + 実際の重みが必要。"""

    def __init__(
        self,
        quantization: str = "4bit",
        lora_dir: Optional[str] = None,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int = 0,
        out_dir: str = "outputs/qwen_image_edit_eval",
        offload_mode: str = "model",
        pipe=None,
    ):
        if pipe is not None:
            # 事前構築済みパイプライン(例: ユーザーが公式リポジトリから読み込んだ
            # QwenImageEditSafePipeline)をそのまま使う。quantization/lora_dirは無視される。
            self.pipe = pipe
        else:
            # gradient checkpointingは逆伝播(学習)時のVRAM節約用の最適化であり、
            # 推論のみのこの用途では不要(diffusers/peftバージョン差異による
            # gradient_checkpointing_enable系の非互換も推論経路では踏まずに済む)。
            # offload_mode: "model"(既定, 通常は高速だがpeft.PeftModelとの組み合わせで
            # 停止したように見える事例あり) / "sequential"(低VRAM環境向け、遅いが安定)/
            # "none"(GPU VRAMに全モデルが載る場合のみ。16GB級のGPUではまず入らない
            # ―text_encoderだけでbf16約14GB、transformerが4bit量子化でも約10GB前後)。
            config = QwenImageEditDPOConfig(
                quantization=quantization, gradient_checkpointing=False,
                offload_mode=offload_mode,
            )
            # lora_dir未指定(=ベースライン評価、DPO学習済みLoRAを使わない)なら、
            # そもそもpeft.PeftModelでラップしない(load_real_pipeline()のapply_lora参照)。
            # 不要なラップを避けることで型不一致の警告と、それに起因すると見られる
            # CPU offload処理の停止の両方を回避できる。
            self.pipe = load_real_pipeline(config, apply_lora=lora_dir is not None)
            if lora_dir:
                # train_qwen_image_edit_dpo.py --save-dir で保存したLoRAアダプタ(安全アライメント後の重み)を適用する。
                self.pipe.transformer.load_adapter(lora_dir, adapter_name="dpo_safety")
                self.pipe.transformer.set_adapter("dpo_safety")

        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def edit(self, image_path: str, prompt: str) -> dict:
        source = Image.open(image_path).convert("RGB")
        self._counter += 1

        # 公式run.pyと同じ呼び出し形。QwenImageEditSafePipeline等、安全パイプラインを
        # 使っている場合はここで例外(SafetyErrorなど)が送出されうる。
        try:
            output = self.pipe(
                image=source,
                prompt=prompt,
                generator=torch.manual_seed(self.seed),
                true_cfg_scale=self.true_cfg_scale,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
            )
        except Exception as e:
            reason = getattr(e, "reason", str(e))
            return {"complied": False, "output_path": None, "refusal_reason": reason}

        output_image = output.images[0]
        out_path = self.out_dir / f"edit_{self._counter:05d}.png"
        output_image.save(out_path)
        return {"complied": True, "output_path": str(out_path)}
