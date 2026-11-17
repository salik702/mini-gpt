"""
Interactive Terminal ChatGPT Interface for Mini-GPT.
Features:
- Real-time token streaming (typewriter effect).
- Multi-turn conversation history management.
- Dynamic ChatML template formatting.
- Customizable system prompt, temperature, and top-p sampling.
"""

import argparse
import os
import sys
import torch

from model import GPT, GPTConfig
from tokenizer import get_tokenizer

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass


# ANSI Color Codes for aesthetic terminal display
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Mini-GPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=["bpe", "char"],
        help="Tokenizer type ('bpe' recommended)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are Mini-GPT, a helpful, intelligent AI assistant built from scratch in pure PyTorch by Salik Ahmad.",
        help="System prompt instructing model persona",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more deterministic, higher = more creative)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K token filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-P (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        help="Maximum generated tokens per response",
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
    device = args.device

    print(f"\n{COLOR_BOLD}{COLOR_MAGENTA}=============================================================={COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}  🤖 MINI-GPT: Interactive Terminal Chat Assistant{COLOR_RESET}")
    print(f"{COLOR_YELLOW}  Built with obsession by Salik Ahmad | Pure PyTorch Transformer{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_MAGENTA}=============================================================={COLOR_RESET}\n")

    # Load tokenizer
    corpus_sample = None
    if args.tokenizer == "char" and os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            corpus_sample = f.read(50000)
    tokenizer = get_tokenizer(args.tokenizer, corpus_sample)

    # Load model
    if os.path.exists(args.checkpoint):
        print(f"[*] Loading model checkpoint from: {COLOR_GREEN}{args.checkpoint}{COLOR_RESET}")
        model = GPT.from_checkpoint(args.checkpoint, device=device)
    else:
        print(
            f"[*] Checkpoint '{args.checkpoint}' not found. Initializing fresh model with default weights."
        )
        print(f"    {COLOR_YELLOW}(Hint: Run 'python train.py --mode chat' to train with conversational data){COLOR_RESET}")
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=256,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
        )
        model = GPT(config).to(device)

    model.eval()
    total_params = model.get_num_params()
    print(f"[*] Model Active: {total_params:,} parameters ({total_params/1e6:.2f}M) on {device}")
    print(f"[*] Commands: type {COLOR_BOLD}'exit'{COLOR_RESET} or {COLOR_BOLD}'quit'{COLOR_RESET} to end, {COLOR_BOLD}'clear'{COLOR_RESET} to reset memory.\n")

    messages = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input(f"{COLOR_BOLD}{COLOR_CYAN}You: {COLOR_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print(f"{COLOR_YELLOW}Goodbye! Happy coding!{COLOR_RESET}\n")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": args.system}]
            print(f"{COLOR_YELLOW}[Memory Cleared]{COLOR_RESET}\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # Format ChatML template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        assert isinstance(formatted_prompt, str)

        prompt_tokens = tokenizer.encode(formatted_prompt)

        # Context length management: keep system prompt and most recent tokens
        max_context = model.config.block_size - args.max_tokens
        if len(prompt_tokens) > max_context and len(prompt_tokens) > 10:
            prompt_tokens = prompt_tokens[-max_context:]

        ctx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

        sys.stdout.write(f"{COLOR_BOLD}{COLOR_GREEN}Mini-GPT: {COLOR_RESET}")
        sys.stdout.flush()

        response_tokens = []

        def token_streamer(token_id: int):
            if token_id != tokenizer.eos_token_id:
                response_tokens.append(token_id)
                char_text = tokenizer.decode([token_id])
                sys.stdout.write(char_text)
                sys.stdout.flush()

        model.generate(
            ctx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            streamer=token_streamer,
        )

        full_reply = tokenizer.decode(response_tokens).strip()
        messages.append({"role": "assistant", "content": full_reply})
        sys.stdout.write("\n\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
