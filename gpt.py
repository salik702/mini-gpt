"""
Mini-GPT: Generative Pre-trained Transformer in Pure PyTorch.
Features:
- Fast Causal Multi-Head FlashAttention (F.scaled_dot_product_attention).
- Unified QKV projection in a single weight matrix.
- Pre-LayerNorm Transformer blocks with GELU non-linearity.
- Weight tying between token embeddings and LM projection head.
- Supervised Fine-Tuning (SFT) support with ChatML prompt loss masking.
- Interactive ChatGPT Terminal Interface: run with `python gpt.py --chat`.
"""

import argparse
import sys
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-GPT Engine")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive ChatGPT terminal chat session",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "chat"],
        help="Training mode: 'pretrain' on text or 'chat' for instruction tuning",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data file (default: input.txt for pretrain, data/conversations.json for chat)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=["bpe", "char"],
        help="Tokenizer type: 'bpe' (Byte-Pair Encoding, GPT-2 compatible) or 'char'",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--block_size", type=int, default=256, help="Context length"
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
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help="Path to save or load model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.chat:
        # Launch interactive chat mode
        import chat

        sys.argv = [
            "chat.py",
            "--checkpoint",
            args.checkpoint,
            "--tokenizer",
            args.tokenizer,
            "--device",
            args.device,
        ]
        chat.main()
        return

    # Standard training mode
    import train

    sys.argv = [
        "train.py",
        "--mode",
        args.mode,
        "--tokenizer",
        args.tokenizer,
        "--batch_size",
        str(args.batch_size),
        "--block_size",
        str(args.block_size),
        "--n_layer",
        str(args.n_layer),
        "--n_head",
        str(args.n_head),
        "--n_embd",
        str(args.n_embd),
        "--dropout",
        str(args.dropout),
        "--max_iters",
        str(args.max_iters),
        "--lr",
        str(args.lr),
        "--save_path",
        args.checkpoint,
        "--device",
        args.device,
    ]
    if args.data is not None:
        sys.argv.extend(["--data", args.data])

    train.main()


if __name__ == "__main__":
    main()
