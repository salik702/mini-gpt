"""
Dataset loaders and preprocessing for Mini-GPT.
Supports:
1. Raw Text Pre-training Dataset (e.g. Tiny Shakespeare or web corpus).
2. Instruction / SFT Conversational Dataset with Loss Masking (ChatML).
"""

import json
from typing import List, Tuple
import torch


class TextDataset:
    """Pre-training dataset for raw continuous text."""

    def __init__(
        self,
        text_or_path: str,
        tokenizer,
        train_ratio: float = 0.9,
        is_file: bool = True,
    ):
        if is_file:
            with open(text_or_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = text_or_path

        tokens = tokenizer.encode(raw_text)
        self.data = torch.tensor(tokens, dtype=torch.long)

        split_idx = int(len(self.data) * train_ratio)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size + 1:
            raise ValueError(
                f"Data split length ({len(data)}) must be larger than block_size ({block_size})."
            )

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


class ChatDataset:
    """
    Supervised Fine-Tuning (SFT) Dataset with Loss Masking.
    Masks user and system prompt tokens with -100 so the cross-entropy loss
    is calculated strictly on the assistant's response tokens.
    """

    def __init__(
        self,
        conversations_path: str,
        tokenizer,
        block_size: int = 256,
        pad_token_id: int = 50259,
    ):
        with open(conversations_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.block_size = block_size
        self.pad_token_id = pad_token_id

        for conversation in raw_data:
            input_ids: List[int] = []
            labels: List[int] = []

            for turn in conversation:
                role = turn["role"]
                content = turn["content"]
                turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                turn_tokens = tokenizer.encode(turn_str)

                input_ids.extend(turn_tokens)

                if role == "assistant":
                    prefix = f"<|im_start|>{role}\n"
                    prefix_tokens = tokenizer.encode(prefix)
                    # Mask prefix tag, supervise assistant response
                    labels.extend([-100] * len(prefix_tokens))
                    labels.extend(turn_tokens[len(prefix_tokens) :])
                else:
                    # User and system prompts are masked from gradient loss
                    labels.extend([-100] * len(turn_tokens))

            # Truncate to block_size + 1
            if len(input_ids) > block_size + 1:
                input_ids = input_ids[: block_size + 1]
                labels = labels[: block_size + 1]

            # Shift x and y for autoregressive prediction
            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(labels[1:], dtype=torch.long)

            # Pad sequences if shorter than block_size
            if len(x) < block_size:
                pad_len = block_size - len(x)
                x = torch.cat(
                    [x, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
                )
                y = torch.cat(
                    [y, torch.full((pad_len,), -100, dtype=torch.long)]
                )

            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self.samples), (batch_size,))
        x = torch.stack([self.samples[i][0] for i in indices])
        y = torch.stack([self.samples[i][1] for i in indices])
        return x.to(device), y.to(device)
