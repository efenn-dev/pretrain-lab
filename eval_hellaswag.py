"""HellaSwag benchmark for pretrain-lab checkpoints (token-mode only).

Scores each of 4 candidate sentence endings by the model's loss on the ending
tokens and picks the lowest. Reports acc (summed loss) and acc_norm
(length-normalized). Random = 25%, original GPT-2 124M ~= 29-31% acc_norm,
frontier models > 95%.

  python eval_hellaswag.py --ckpt-dir runs/fw-124m --max-examples 1000

Writes results to <ckpt-dir>/hellaswag.json (picked up by the dashboard).
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from train import GPT, GPTConfig

log = logging.getLogger("hellaswag")

DATA_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=1000, help="10042 = full val set")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads((args.ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["mode"] != "gpt2":
        raise SystemExit("HellaSwag needs a token-mode (gpt2) checkpoint")

    data_path = args.data_dir / "hellaswag_val.jsonl"
    if not data_path.exists():
        log.info("downloading HellaSwag val set...")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)

    ckpt = torch.load(args.ckpt_dir / "ckpt.pt", map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("loaded %s (iter %d, val loss %.3f)", args.ckpt_dir, ckpt["iter"], ckpt["val_loss"])

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    correct = correct_norm = n = 0
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            ctx_t = enc.encode_ordinary(ex["ctx"])
            rows: list[list[int]] = []
            end_lens: list[int] = []
            for ending in ex["endings"]:
                end_t = enc.encode_ordinary(" " + ending)
                row = (ctx_t + end_t)[-cfg.block_size :]
                rows.append(row)
                end_lens.append(min(len(end_t), len(row) - 1))

            max_len = max(len(r) for r in rows)
            x = torch.zeros(4, max_len, dtype=torch.long, device=device)
            for i, r in enumerate(rows):
                x[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                logits, _ = model(x, x)  # pass targets so forward returns full-sequence logits
            logp = F.log_softmax(logits.float(), dim=-1)

            loss_sum, loss_avg = [], []
            for i, r in enumerate(rows):
                length, e = len(r), end_lens[i]
                tgt = torch.tensor(r[length - e :], dtype=torch.long, device=device)
                lp = logp[i, length - 1 - e : length - 1].gather(1, tgt[:, None]).squeeze(1)
                loss_sum.append(-lp.sum().item())
                loss_avg.append(-lp.mean().item())

            label = int(ex["label"])
            correct += int(min(range(4), key=lambda i: loss_sum[i]) == label)
            correct_norm += int(min(range(4), key=lambda i: loss_avg[i]) == label)
            n += 1
            if n % 200 == 0:
                log.info("%d examples: acc %.4f, acc_norm %.4f", n, correct / n, correct_norm / n)
            if n >= args.max_examples:
                break

    result = {"n": n, "acc": round(correct / n, 4), "acc_norm": round(correct_norm / n, 4)}
    (args.ckpt_dir / "hellaswag.json").write_text(json.dumps(result), encoding="utf-8")
    log.info("final: %s (random = 0.25)", result)


if __name__ == "__main__":
    main()
