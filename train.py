"""
Training Engine for Mini-GPT.
Supports:
1. Pre-training on raw text (e.g. Shakespeare or web text).
2. Supervised Fine-Tuning (SFT / Chat tuning) on conversational datasets with loss masking.
Features:
- AdamW optimizer with cosine learning rate decay and warmup.
- Gradient clipping (1.0).
- Automatic checkpointing to checkpoint.pt.
"""

import argparse
import math
import os
import sys
import time
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

from model import GPT, GPTConfig
from tokenizer import get_tokenizer
from dataset import TextDataset, ChatDataset


def get_lr(
    it: int,
    max_iters: int,
    warmup_iters: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate decay with linear warmup."""
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini-GPT")
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "chat"],
        help="Training mode: pretrain on raw text or chat for instruction tuning",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (defaults to input.txt for pretrain, data/conversations.json for chat)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=["bpe", "char"],
        help="Tokenizer type: 'bpe' (GPT-2 compatible tiktoken) or 'char' (toy character)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--block_size", type=int, default=256, help="Maximum context length"
    )
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of Transformer blocks"
    )
    parser.add_argument(
        "--n_head", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_embd", type=int, default=384, help="Embedding dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Total training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=50, help="Linear warmup iterations"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Loss evaluation interval"
    )
    parser.add_argument(
        "--eval_iters", type=int, default=20, help="Evaluation iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target compute device",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoint.pt",
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Optional path to existing checkpoint to fine-tune from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"[*] Running on device: {device}")

    # Set default data path if not provided
    if args.data is None:
        args.data = (
            "input.txt"
            if args.mode == "pretrain"
            else os.path.join("data", "conversations.json")
        )

    # Load tokenizer
    corpus_sample = None
    if args.tokenizer == "char" and os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            corpus_sample = f.read(50000)
    tokenizer = get_tokenizer(args.tokenizer, corpus_sample)
    print(
        f"[*] Initialized {args.tokenizer.upper()} tokenizer (vocab size: {tokenizer.vocab_size:,})"
    )

    # Load dataset
    print(f"[*] Loading data from: {args.data} (mode: {args.mode})")
    if args.mode == "pretrain":
        dataset = TextDataset(
            args.data, tokenizer, train_ratio=0.9, is_file=True
        )
    else:
        dataset = ChatDataset(
            args.data,
            tokenizer,
            block_size=args.block_size,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Initialize or load model
    if args.init_from and os.path.exists(args.init_from):
        print(f"[*] Loading checkpoint from {args.init_from}...")
        model = GPT.from_checkpoint(args.init_from, device=device)
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=False,
        )
        model = GPT(config).to(device)

    total_params = model.get_num_params()
    print(f"[*] Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Configure optimizer
    min_lr = args.lr * 0.1
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device,
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        if args.mode == "pretrain":
            assert isinstance(dataset, TextDataset)
            for split in ["train", "val"]:
                losses = torch.zeros(args.eval_iters)
                for k in range(args.eval_iters):
                    X, Y = dataset.get_batch(
                        split, args.batch_size, args.block_size, device=device
                    )
                    _, loss = model(X, Y)
                    if loss is not None:
                        losses[k] = loss.item()
                out[split] = losses.mean().item()
        else:
            assert isinstance(dataset, ChatDataset)
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = dataset.get_batch(args.batch_size, device=device)
                _, loss = model(X, Y)
                if loss is not None:
                    losses[k] = loss.item()
            out["train"] = losses.mean().item()
        model.train()
        return out

    print("[*] Starting training...")
    start_time = time.time()

    for iter_num in range(args.max_iters + 1):
        # Update learning rate with cosine decay
        lr = get_lr(
            iter_num, args.max_iters, args.warmup_iters, args.lr, min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate periodically
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            if args.mode == "pretrain":
                print(
                    f"step {iter_num:4d} | train loss: {losses['train']:.4f} | val loss: {losses.get('val', 0.0):.4f} | lr: {lr:.2e} | elapsed: {elapsed:.1f}s"
                )
            else:
                print(
                    f"step {iter_num:4d} | chat loss: {losses['train']:.4f} | lr: {lr:.2e} | elapsed: {elapsed:.1f}s"
                )

        if iter_num == args.max_iters:
            break

        # Fetch batch
        if args.mode == "pretrain":
            assert isinstance(dataset, TextDataset)
            xb, yb = dataset.get_batch(
                "train", args.batch_size, args.block_size, device=device
            )
        else:
            assert isinstance(dataset, ChatDataset)
            xb, yb = dataset.get_batch(args.batch_size, device=device)

        # Forward, backward, and optimizer step
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        assert loss is not None
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Save final model checkpoint
    model.save_checkpoint(args.save_path)
    print(f"[OK] Training complete! Saved checkpoint to: {args.save_path}")

    # Generate quick sample
    print("\n--- Model Output Generation Sample ---")
    model.eval()
    if args.mode == "pretrain":
        prompt_text = "The king doth"
        ctx = torch.tensor(
            tokenizer.encode(prompt_text), dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        sample_msg = [
            {"role": "user", "content": "Who are you?"}
        ]
        prompt_str = tokenizer.apply_chat_template(
            sample_msg, add_generation_prompt=True
        )
        assert isinstance(prompt_str, str)
        ctx = torch.tensor(
            tokenizer.encode(prompt_str), dtype=torch.long, device=device
        ).unsqueeze(0)

    generated = model.generate(
        ctx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(generated[0].tolist()))
    print("--------------------------------------\n")


if __name__ == "__main__":
    main()
