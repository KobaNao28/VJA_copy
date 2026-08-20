"""
評価対象モデルの抽象アダプタ。

実運用では本クラスを継承し、`edit()` を実際の画像編集VLM呼び出し(商用API or ローカル重み)に
差し替えることで、`run_eval.py` から同一インターフェースで任意モデルを評価できる。

例:
  class GPTImageAdapter(ModelAdapter):
      def edit(self, image_path, prompt):
          response = openai_client.images.edit(image=open(image_path, "rb"), prompt=prompt)
          ...
          return {"complied": True, "output_path": out_path}
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def edit(self, image_path: str, prompt: str) -> dict:
        """{'complied': bool, 'output_path': str | None} を返す。"""
        raise NotImplementedError


class DummyWeaklyAlignedAdapter(ModelAdapter):
    """
    論文で報告される「弱アライメントモデル」の挙動を模した合成アダプタ。
    実モデルへの接続なしにパイプライン全体(defense -> model -> judge -> metrics)を
    検証するために使う。base_compliance は「安全フィルタなしなら指示に従う確率」。
    """

    def __init__(self, base_compliance: float = 0.85, seed: int = 0):
        self.base_compliance = base_compliance
        self.rng = random.Random(seed)

    def edit(self, image_path: str, prompt: str) -> dict:
        complied = self.rng.random() < self.base_compliance
        return {"complied": complied, "output_path": image_path if complied else None}


class DummyWellAlignedAdapter(ModelAdapter):
    """テキストプロンプトのみを検査する素朴な安全フィルタを内蔵した合成モデル。"""

    _REFUSAL_KEYWORDS = ["weapon", "kill", "武器", "殺", "nude", "explosive"]

    def __init__(self, base_compliance: float = 0.4, seed: int = 0):
        self.base_compliance = base_compliance
        self.rng = random.Random(seed)

    def edit(self, image_path: str, prompt: str) -> dict:
        if any(k.lower() in prompt.lower() for k in self._REFUSAL_KEYWORDS):
            return {"complied": False, "output_path": None}
        complied = self.rng.random() < self.base_compliance
        return {"complied": complied, "output_path": image_path if complied else None}
