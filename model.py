"""
Core Decoder-Only Transformer (GPT) Architecture.
Includes:
- Unified QKV projection with native PyTorch FlashAttention (scaled_dot_product_attention).
- Pre-LayerNorm residual stream architecture with GELU activation.
- Weight tying between token embeddings and language model head.
- Supervised Fine-Tuning (SFT) support with loss masking (ignore_index=-100).
- Autoregressive generation with Temperature, Top-K, Top-P (nucleus), and EOS stopping.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50260
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with PyTorch FlashAttention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Unified linear projection for query, key, and value in a single matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute query, key, and value for all heads in parallel: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch FlashAttention (Hardware-accelerated scaled dot-product with causal mask)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Re-assemble multi-head outputs: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-Forward Network with 4x hidden expansion and GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer Block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections for gradient stability
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Full Generative Pre-trained Transformer Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie token embedding weights with output projection weights (Press & Wolf 2017)
        self.wte.weight = self.lm_head.weight

        # Initialize all model weights
        self.apply(self._init_weights)

        # Special scaled initialization for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        _, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Sequence length {T} exceeds maximum context size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Forward through embeddings and transformer blocks
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if targets is not None:
            # Full sequence projection for cross-entropy loss computation
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # ignore_index=-100 ensures prompt tokens in SFT / Chat are not penalized
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            # Inference optimization: compute logits only for the final token position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with proper parameter decay grouping:
        - 2D weight tensors (Linear, Embedding) receive weight decay.
        - 1D tensors (LayerNorm weights/biases, Linear biases) receive NO weight decay.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        streamer: Optional[Callable[[int], None]] = None,
    ) -> torch.Tensor:
        """
        Autoregressive next-token sampling with:
        - Temperature scaling
        - Top-K filtering
        - Top-P (nucleus) filtering
        - Early stopping on EOS token
        - Real-time token streaming callback
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context window to maximum allowed block size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # focus on last step

            if temperature > 0.0:
                logits = logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (argmax)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            next_token = int(idx_next[0, 0].item())

            if streamer is not None and idx.size(0) == 1:
                streamer(next_token)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and idx.size(0) == 1 and next_token == eos_token_id:
                break

        return idx

    def save_checkpoint(self, path: str):
        """Save model weights and configuration."""
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "GPT":
        """Load model weights and configuration from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = GPTConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
