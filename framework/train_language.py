"""
Language Modeling Experiment: QuatBlock vs Givens vs Diagonal SSM (Issue #31)

Trains three SSM language models on WikiText-2, comparing next-token prediction
perplexity. Tests whether the geometric (spinor) inductive bias that helped on
toy Markov data also helps on real language.

Models:
  1. QuatBlock SSM: 256 blocks of SU(2) sandwich product (D=768, block_size=3)
  2. Givens SSM: 384 paired rotations (D=768, block_size=2)
  3. Diagonal SSM: per-dimension decay (Mamba-style baseline)

Usage:
    cd framework && python3 train_language.py [--d_model 768] [--n_layers 2]
                                               [--epochs 10] [--batch_size 8]
                                               [--seq_len 256]
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from models.ssm_cuda import SSMLanguageModelCuda as SSMLanguageModel
    print("Using CUDA-fused SSM kernels", flush=True)
except Exception as e:
    from models.ssm_cells import SSMLanguageModel
    print(f"CUDA kernels unavailable ({e}), falling back to Python SSM", flush=True)


# ============================================================================
# 1. Data
# ============================================================================

class WikiTextDataset(Dataset):
    """WikiText-2 dataset for causal language modeling."""

    def __init__(self, split="train", seq_len=256, tokenizer=None):
        from datasets import load_dataset

        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split,
                           trust_remote_code=True)

        # Concatenate all text
        text = "\n".join(line for line in raw["text"] if line.strip())

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Tokenize the full corpus
        tokens = tokenizer.encode(text, add_special_tokens=False)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        # Number of full sequences we can extract
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        input_ids = self.tokens[start:start + self.seq_len]
        labels = self.tokens[start + 1:start + self.seq_len + 1]
        return input_ids, labels


# ============================================================================
# 2. Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                log_interval=50):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_tokens = labels.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            tok_per_sec = total_tokens / elapsed
            print(f"  epoch {epoch} | batch {batch_idx+1}/{len(dataloader)} | "
                  f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                  f"{tok_per_sec:.0f} tok/s", flush=True)

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1))

        total_tokens += labels.numel()
        total_loss += loss.item() * labels.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20))


# ============================================================================
# 3. Main
# ============================================================================

def run_model(ssm_type, args, tokenizer, train_dataset, val_dataset, device):
    """Train and evaluate one SSM type."""
    vocab_size = tokenizer.vocab_size

    model = SSMLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        ssm_type=ssm_type,
    ).to(device)

    total_params = model.count_parameters()
    ssm_params = model.count_ssm_parameters()

    print(f"\n{'='*70}", flush=True)
    print(f"Model: {ssm_type}", flush=True)
    print(f"  Total params: {total_params:,}", flush=True)
    print(f"  SSM params:   {ssm_params:,}", flush=True)
    print(f"  Other params: {total_params - ssm_params:,}", flush=True)
    print(f"{'='*70}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    best_val_ppl = float("inf")
    train_losses = []
    val_ppls = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, epoch, log_interval=args.log_interval)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_ppls.append(val_ppl)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        print(f"  epoch {epoch} DONE | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | "
              f"best_ppl={best_val_ppl:.1f}", flush=True)

    elapsed = time.time() - t0

    return {
        "ssm_type": ssm_type,
        "total_params": total_params,
        "ssm_params": ssm_params,
        "best_val_ppl": best_val_ppl,
        "final_val_ppl": val_ppls[-1],
        "final_train_loss": train_losses[-1],
        "train_time": elapsed,
        "val_ppls": val_ppls,
    }


def main():
    parser = argparse.ArgumentParser(description="SSM Language Modeling (Issue #31)")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--models", nargs="+",
                        default=["quatblock", "givens", "diagonal"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_total:.1f} GB", flush=True)

    print(f"\nConfig: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"seq_len={args.seq_len}", flush=True)
    print(f"        epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}", flush=True)
    print(f"        models={args.models}", flush=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading WikiText-2...", flush=True)
    train_dataset = WikiTextDataset("train", seq_len=args.seq_len,
                                    tokenizer=tokenizer)
    val_dataset = WikiTextDataset("validation", seq_len=args.seq_len,
                                  tokenizer=tokenizer)
    print(f"  Train: {len(train_dataset)} sequences ({len(train_dataset) * args.seq_len:,} tokens)", flush=True)
    print(f"  Val:   {len(val_dataset)} sequences ({len(val_dataset) * args.seq_len:,} tokens)", flush=True)

    # Run each model
    all_results = []
    for ssm_type in args.models:
        # Clear GPU memory between models
        torch.cuda.empty_cache()
        result = run_model(ssm_type, args, tokenizer, train_dataset,
                           val_dataset, device)
        all_results.append(result)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Model':>12} | {'Total Params':>13} | {'SSM Params':>11} | "
          f"{'Best PPL':>9} | {'Final PPL':>10} | {'Time':>7}", flush=True)
    print(f"{'-'*12}-+-{'-'*13}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}",
          flush=True)

    for r in all_results:
        print(f"{r['ssm_type']:>12} | {r['total_params']:>13,} | "
              f"{r['ssm_params']:>11,} | {r['best_val_ppl']:>9.1f} | "
              f"{r['final_val_ppl']:>10.1f} | {r['train_time']:>6.0f}s",
              flush=True)
    print(f"{'-'*12}-+-{'-'*13}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}",
          flush=True)

    # Pairwise comparison
    if len(all_results) >= 2:
        baseline = next((r for r in all_results if r["ssm_type"] == "diagonal"),
                        all_results[-1])
        print(f"\nComparison vs Diagonal (best_ppl={baseline['best_val_ppl']:.1f}):",
              flush=True)
        for r in all_results:
            if r["ssm_type"] != baseline["ssm_type"]:
                delta = r["best_val_ppl"] - baseline["best_val_ppl"]
                pct = delta / baseline["best_val_ppl"] * 100
                direction = "better" if delta < 0 else "worse"
                print(f"  {r['ssm_type']:>12}: {delta:+.1f} ppl ({pct:+.1f}%, {direction})",
                      flush=True)

    # Learning curves
    print(f"\nLearning curves (val PPL per epoch):", flush=True)
    header = f"{'Epoch':>6}"
    for r in all_results:
        header += f" | {r['ssm_type']:>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for epoch in range(args.epochs):
        row = f"{epoch+1:>6}"
        for r in all_results:
            row += f" | {r['val_ppls'][epoch]:>12.1f}"
        print(row, flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
