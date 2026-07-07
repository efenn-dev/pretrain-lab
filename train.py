"""Minimal GPT pretraining — single GPU, from scratch.

Two data modes, auto-detected from --data:
  * a .txt file  -> char-level (vocab built from the file; great for smoke tests)
  * a directory  -> token-level, expects train.bin / val.bin of uint16 GPT-2
                    tokens (produced by prepare_fineweb.py)

Example (smoke test, ~2 min on an RTX 5070 Ti):
  python train.py --data data/shakespeare.txt --out-dir runs/shakespeare-nano \
      --preset nano --max-iters 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("train")

# rough param counts assume the GPT-2 BPE vocab (50304); char-level is smaller
PRESETS: dict[str, dict[str, int]] = {
    "nano": {"n_layer": 6, "n_head": 6, "n_embd": 384},        # ~10M params
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},      # ~124M params
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # ~350M params
}


@dataclass
class GPTConfig:
    """Architecture hyperparameters for the GPT model."""

    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    arch: str = "gpt2"  # "gpt2" (learned pos, LayerNorm, GELU) or "modern" (RoPE, RMSNorm, SwiGLU)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, both residual."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.attn_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp_up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.mlp_down = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn_qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(y)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        return x


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q/k by position-dependent angles (RoPE, half-split layout)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class ModernBlock(nn.Module):
    """Llama-style block: RMSNorm, rotary positions, SwiGLU MLP."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = 128 * round(8 * cfg.n_embd / 3 / 128)  # SwiGLU sized to match GELU param count
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.attn_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.mlp_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.mlp_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.mlp_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        cos, sin = rope
        q, k, v = self.attn_qkv(self.norm1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(y)
        h = self.norm2(x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model (GPT-2 family shape)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.arch == "modern":
            self.wpe = None
            head_dim = cfg.n_embd // cfg.n_head
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            freqs = torch.outer(torch.arange(cfg.block_size).float(), inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)
            self.blocks = nn.ModuleList(ModernBlock(cfg) for _ in range(cfg.n_layer))
            self.ln_f: nn.Module = nn.RMSNorm(cfg.n_embd)
        else:
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
            self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # residual-path projections: GPT-2 scales init with depth; modern zero-inits
        for name, p in self.named_parameters():
            if name.endswith(("attn_proj.weight", "mlp_down.weight")):
                if cfg.arch == "modern":
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        """Parameter count excluding position embeddings (reporting convention)."""
        n = sum(p.numel() for p in self.parameters())
        return n - (self.wpe.weight.numel() if self.wpe is not None else 0)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x = self.wte(idx)
        rope = None
        if self.wpe is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)
        else:
            rope = (self.rope_cos[:T], self.rope_sin[:T])
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, rope)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 200
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def load_data(data_path: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load training data; returns (train_tokens, val_tokens, vocab_size).

    Char mode writes meta.json (the char vocab) next to the checkpoint so
    sample.py can decode.
    """
    if data_path.is_file():
        text = data_path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        (out_dir / "meta.json").write_text(
            json.dumps({"mode": "char", "itos": chars}), encoding="utf-8"
        )
        arr = np.array([stoi[c] for c in text], dtype=np.uint16)
        n = int(0.9 * len(arr))
        log.info("char mode: %d chars, vocab %d", len(arr), len(chars))
        return arr[:n], arr[n:], len(chars)

    train = np.memmap(data_path / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data_path / "val.bin", dtype=np.uint16, mode="r")
    (out_dir / "meta.json").write_text(json.dumps({"mode": "gpt2"}), encoding="utf-8")
    log.info("token mode: %d train tokens, %d val tokens", len(train), len(val))
    return train, val, 50304


def get_batch(
    arr: np.ndarray, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(arr) - block_size - 1, size=batch_size)
    x = torch.from_numpy(np.stack([arr[i : i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([arr[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix]))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(
    model: GPT, splits: dict[str, np.ndarray], block_size: int, batch_size: int,
    device: str, eval_iters: int,
) -> dict[str, float]:
    model.eval()
    out = {}
    for name, arr in splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(arr, block_size, batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it: int, max_iters: int, lr: float, warmup: int) -> float:
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help=".txt file or dir with train.bin/val.bin")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="nano")
    parser.add_argument("--arch", choices=["gpt2", "modern"], default="gpt2",
                        help="modern = RoPE + RMSNorm + SwiGLU (train ~same cost, better loss)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", help="torch.compile (needs Triton; off by default on Windows)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, vocab_size = load_data(args.data, args.out_dir)
    cfg = GPTConfig(
        block_size=args.block_size, vocab_size=vocab_size,
        dropout=args.dropout, arch=args.arch, **PRESETS[args.preset],
    )
    model = GPT(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    log.info("model: %s on %s — %.2fM params", args.preset, device, model.num_params() / 1e6)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )

    best_val = float("inf")
    tokens_per_iter = args.batch_size * args.block_size * args.grad_accum
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(it, args.max_iters, args.lr, args.warmup_iters)

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(
                model, {"train": train_data, "val": val_data},
                args.block_size, args.batch_size, device, args.eval_iters,
            )
            log.info("iter %d: train loss %.4f, val loss %.4f", it, losses["train"], losses["val"])
            if losses["val"] < best_val:
                best_val = losses["val"]
                raw = getattr(model, "_orig_mod", model)
                torch.save(
                    {"model": raw.state_dict(), "config": asdict(cfg),
                     "iter": it, "val_loss": best_val},
                    args.out_dir / "ckpt.pt",
                )
        if it == args.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_iter * args.log_interval / dt if it > 0 else 0
            log.info("iter %d: loss %.4f, %.0f tok/s", it, loss.item(), tps)

    log.info("done — best val loss %.4f, checkpoint at %s", best_val, args.out_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
