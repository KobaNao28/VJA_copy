"""
DPO (Direct Preference Optimization) によるマルチモーダル安全アライメント学習。

想定データ形式(JSONL): 各行が以下を持つ選好ペア
  {
    "image_path": "path/to/image.png"  (省略可: テキストのみDPOにも対応),
    "prompt": "ユーザー入力(画像内の視覚指示を含む文脈記述)",
    "chosen": "安全なモデル応答(例: 編集を拒否し理由を説明する)",
    "rejected": "安全でないモデル応答(例: 有害編集をそのまま実行してしまう)"
  }

本番運用では `--model-name` に実際のVLM(例: Qwen2-VL, LLaVA-Next 等)のHFモデルIDを
指定し、`transformers` + `peft`(LoRA)+ 実データで学習する。
ネットワーク/GPUが無い環境でも配線を検証できるよう、`--mock` モードでは外部モデルを
一切ダウンロードせず、リポジトリ内で完結する小さな Transformer で
DPO損失が実際に低下することを確認できる。

DPO損失(Rafailov et al., 2023):
  L = -log σ( β * [ (logπ(chosen|x) - logπ_ref(chosen|x))
                   - (logπ(rejected|x) - logπ_ref(rejected|x)) ] )
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.seed import set_seed


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


# --------------------------------------------------------------------------------------
# 実モデル(HF Transformers + PEFT LoRA)向けのアダプタ。ネットワーク接続がある環境で使用する。
# --------------------------------------------------------------------------------------
def build_hf_policy_and_ref(model_name: str, use_lora: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    ref = AutoModelForCausalLM.from_pretrained(model_name)
    for p in ref.parameters():
        p.requires_grad_(False)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        policy = get_peft_model(policy, lora_cfg)

    return tokenizer, policy, ref


def sequence_logp(model, tokenizer, prompt: str, response: str, device: str) -> torch.Tensor:
    full = prompt + response
    enc = tokenizer(full, return_tensors="pt").to(device)
    prompt_len = len(tokenizer(prompt)["input_ids"])
    out = model(**enc)
    logits = out.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)
    resp_mask = torch.zeros_like(labels, dtype=torch.bool)
    resp_mask[:, prompt_len - 1:] = True
    return (token_logp * resp_mask).sum(dim=-1)


def train_with_hf(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, policy, ref = build_hf_policy_and_ref(args.model_name, use_lora=not args.no_lora)
    policy.to(device)
    ref.to(device)
    ref.eval()

    pairs = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for row in pairs:
            prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
            pc = sequence_logp(policy, tokenizer, prompt, chosen, device)
            pr = sequence_logp(policy, tokenizer, prompt, rejected, device)
            with torch.no_grad():
                rc = sequence_logp(ref, tokenizer, prompt, chosen, device)
                rr = sequence_logp(ref, tokenizer, prompt, rejected, device)
            loss = dpo_loss(pc, pr, rc, rr, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[epoch {epoch}] avg DPO loss = {total_loss / max(1, len(pairs)):.4f}")

    if args.save_dir:
        policy.save_pretrained(args.save_dir)
        print(f"saved LoRA adapter -> {args.save_dir}")


# --------------------------------------------------------------------------------------
# Mockモデル: 外部ダウンロード不要で DPO の配線(損失計算・勾配更新)を検証するための
# ごく小さな文字レベル Transformer。実運用のモデル規模を模擬するものではない。
# --------------------------------------------------------------------------------------
class TinyCharTransformer(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos_ids)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(input_ids.device)
        x = self.encoder(x, mask=causal_mask)
        return self.head(x)


def _encode_bytes(text: str, max_len: int) -> list[int]:
    b = list(text.encode("utf-8")[:max_len])
    return b + [0] * (max_len - len(b))


def _tiny_sequence_logp(
    model: TinyCharTransformer, prompt: str, response: str, prompt_budget: int = 64, response_budget: int = 64
) -> torch.Tensor:
    """
    プロンプト部とレスポンス部を別々にバイト数上限内へ収めてから連結する。
    (プロンプトが長い場合にレスポンス全体が切り詰められてしまい、
     選好シグナルが消える=DPO損失が学習不能になる問題を避けるため)
    """
    prompt_ids = _encode_bytes(prompt, prompt_budget)
    response_bytes = response.encode("utf-8")[:response_budget]
    response_ids = list(response_bytes) + [0] * (response_budget - len(response_bytes))
    full = torch.tensor(prompt_ids + response_ids, dtype=torch.long).unsqueeze(0)

    logits = model(full)[:, :-1, :]
    labels = full[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    resp_start = prompt_budget  # labels は full[1:] なので prompt_budget 番目から response 領域
    resp_len = len(response_bytes)
    mask[:, resp_start:resp_start + resp_len] = True
    return (token_logp * mask).sum(dim=-1)


def train_mock(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    pairs = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pairs:
        raise SystemExit(f"学習データが空です: {args.data}")

    policy = TinyCharTransformer()
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    losses = []
    for epoch in range(args.epochs):
        total = 0.0
        for row in pairs:
            pc = _tiny_sequence_logp(policy, row["prompt"], row["chosen"])
            pr = _tiny_sequence_logp(policy, row["prompt"], row["rejected"])
            with torch.no_grad():
                rc = _tiny_sequence_logp(ref, row["prompt"], row["chosen"])
                rr = _tiny_sequence_logp(ref, row["prompt"], row["rejected"])
            loss = dpo_loss(pc, pr, rc, rr, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(pairs)
        losses.append(avg)
        print(f"[mock epoch {epoch}] avg DPO loss = {avg:.4f}")

    improved = losses[-1] < losses[0]
    print(f"損失は{'減少しました(配線OK)' if improved else '減少しませんでした(要確認)'}: {losses[0]:.4f} -> {losses[-1]:.4f}")


@dataclass
class SafetyPreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    image_path: str | None = None


def build_synthetic_preference_data(iesbench_jsonl: str, out_path: str, n: int = 60) -> None:
    """
    src/dataset/build_dataset.py が生成したIESBench互換データから、
    「危険な編集をそのまま実行する応答(rejected)」対「安全ポリシーに触れて拒否/代替案を
    提示する応答(chosen)」の選好ペアを合成する(プレースホルダー文言)。
    """
    rows = [json.loads(l) for l in Path(iesbench_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    out_rows = []
    for row in rows:
        prompt = f"[image with embedded visual instruction] {row['question']}"
        chosen = (
            f"このリクエストは安全ポリシー『{row['category']}』に抵触する可能性があるため、"
            f"編集を実行せず、理由を説明した上で安全な代替案を提示します。"
        )
        rejected = f"了解しました。{row['action']} を {row['attributes'][0]} に対して直ちに実行します。"
        out_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"選好データ {len(out_rows)} 件を生成 -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="マルチモーダルDPOによる安全アライメント学習")
    p.add_argument("--data", default="data/sample/dpo_preferences.jsonl")
    p.add_argument("--build-from-iesbench", default=None, help="IESBench互換jsonlから選好データを自動合成")
    p.add_argument("--mock", action="store_true", help="外部モデル不要のミニTransformerで配線検証")
    p.add_argument("--model-name", default=None, help="HF上の実VLM/LLMモデルID(--mock未指定時に使用)")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    if args.build_from_iesbench:
        build_synthetic_preference_data(args.build_from_iesbench, args.data)

    if args.mock:
        train_mock(args)
    else:
        if not args.model_name:
            raise SystemExit("--mock を指定しない場合は --model-name (HFモデルID) が必要です")
        train_with_hf(args)


if __name__ == "__main__":
    main()
