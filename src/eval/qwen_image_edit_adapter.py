"""
実際の Qwen/Qwen-Image-Edit(diffusers)を run_eval.py の ModelAdapter として使う実装。

`src/defense/train_qwen_image_edit_dpo.py::load_real_pipeline()` で確認済みの実際のAPI
(`diffusers.QwenImageEditPipeline`, 4bit/8bit量子化 + LoRA)でパイプラインをロードし、
`ModelAdapter.edit()` インターフェースに合わせて画像編集を実行する。DPOで学習した
LoRAアダプタ(`train_qwen_image_edit_dpo.py --save-dir`の出力)を適用した状態("安全アライメント後")
でも、ベースモデルのまま("無防御"設定、論文でいう[O] Qwen-Image-Edit*相当)でも評価できる。

## 誠実な注記(重要)

- `load_real_pipeline()`が確認しているのは「モデル/コンフィグのロードAPI」までであり、
  実際の生成呼び出し `pipe(image=..., prompt=..., ...)` の正確なkwargs名は、
  本セッションが重みをダウンロードできない制約上、diffusersの画像編集系パイプライン
  (Qwen-Image-Edit公式のREADME/使用例)の一般的な慣例に基づいて実装している。
  実際に動かす環境でエラーが出た場合は `help(pipe.__call__)` や公式の使用例で
  正確な引数名を確認し、`_call_pipe()` を調整すること。
- `complied`(モデルが指示に従ったか)の判定は本アダプタでは常に `True` とする。
  拡散モデル本体は(安全アライメント済みの特別な拒否機構が無い限り)基本的に
  何らかの画像を生成してしまうため、「拒否したかどうか」を画像そのものから判定するのは
  本質的にjudge(LLM-as-judge等)の仕事である(`src/eval/judge.py::LLMJudge`)。
  もし使用するチェックポイントに既知の拒否シグナル(特定のウォーターマーク画像、
  変化のない出力等)があるなら、`refusal_detector` にコールバックとして渡すことで
  上書きできる。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.defense.train_qwen_image_edit_dpo import QwenImageEditDPOConfig, load_real_pipeline
from src.eval.model_adapter import ModelAdapter

RefusalDetector = Callable[[Image.Image, Image.Image, str], bool]  # (source, output, prompt) -> 拒否したか


class QwenImageEditAdapter(ModelAdapter):
    """実際のQwen-Image-Editをrun_eval.pyから呼び出すためのアダプタ。GPU + 実際の重みが必要。"""

    def __init__(
        self,
        quantization: str = "4bit",
        lora_dir: Optional[str] = None,
        num_inference_steps: int = 30,
        true_cfg_scale: float = 4.0,
        out_dir: str = "outputs/qwen_image_edit_eval",
        refusal_detector: Optional[RefusalDetector] = None,
    ):
        # gradient checkpointingは逆伝播(学習)時のVRAM節約用の最適化であり、
        # run_eval.py経由の推論のみの用途では不要(かつ無効にしておけば
        # diffusers/peftバージョン差異によるgradient_checkpointing_enable系の
        # 非互換を推論経路では踏まずに済む)。
        config = QwenImageEditDPOConfig(quantization=quantization, gradient_checkpointing=False)
        self.pipe = load_real_pipeline(config)
        if lora_dir:
            # train_qwen_image_edit_dpo.py --save-dir で保存したLoRAアダプタ(安全アライメント後の重み)を適用する。
            self.pipe.transformer.load_adapter(lora_dir, adapter_name="dpo_safety")
            self.pipe.transformer.set_adapter("dpo_safety")
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.refusal_detector = refusal_detector
        self._counter = 0

    def _call_pipe(self, image: Image.Image, prompt: str) -> Image.Image:
        """実際の生成呼び出し。kwargs名は要検証(このファイル冒頭の注記参照)。"""
        result = self.pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.true_cfg_scale,
        )
        return result.images[0]

    def edit(self, image_path: str, prompt: str) -> dict:
        source = Image.open(image_path).convert("RGB")
        output = self._call_pipe(source, prompt)

        self._counter += 1
        out_path = self.out_dir / f"edit_{self._counter:05d}.png"
        output.save(out_path)

        complied = True
        if self.refusal_detector is not None:
            complied = not self.refusal_detector(source, output, prompt)

        return {"complied": complied, "output_path": str(out_path) if complied else None}
