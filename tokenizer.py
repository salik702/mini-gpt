"""
Tokenization subsystem for Mini-GPT.
Supports:
1. BPE (Byte-Pair Encoding) with ChatML special tokens via tiktoken (GPT-2 compatible).
2. Fallback Character-level tokenizer for toy Shakespeare experiments.
"""

from typing import List, Dict, Union, Optional, Literal, Set

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer with native ChatML special token support."""

    SPECIAL_TOKENS = {
        "<|endoftext|>": 50256,
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
        "<|pad|>": 50259,
    }

    def __init__(self):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for BPETokenizer. Install via `pip install tiktoken`."
            )

        base_enc = tiktoken.get_encoding("gpt2")
        self._enc = tiktoken.Encoding(
            name="gpt2_chatml",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=self.SPECIAL_TOKENS,
        )

        self.vocab_size = self._enc.n_vocab
        self.eos_token_id = self.SPECIAL_TOKENS["<|im_end|>"]
        self.pad_token_id = self.SPECIAL_TOKENS["<|pad|>"]
        self.bos_token_id = self.SPECIAL_TOKENS["<|im_start|>"]
        self.im_start_id = self.SPECIAL_TOKENS["<|im_start|>"]
        self.im_end_id = self.SPECIAL_TOKENS["<|im_end|>"]

    def encode(self, text: str, allowed_special: Union[Literal["all"], Set[str]] = "all") -> List[int]:
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        return self._enc.decode(token_ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """
        Formats a list of messages into ChatML structure:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        if tokenize:
            return self.encode(formatted)
        return formatted


class CharTokenizer:
    """Character-level tokenizer fallback with ChatML special token handling."""

    SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|pad|>", "<|endoftext|>"]

    def __init__(self, text_corpus: str = ""):
        chars = sorted(list(set(text_corpus)))
        for sp in self.SPECIAL_TOKENS:
            if sp not in chars:
                chars.append(sp)

        self.chars = chars
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.im_start_id = self.stoi["<|im_start|>"]
        self.im_end_id = self.stoi["<|im_end|>"]
        self.pad_token_id = self.stoi["<|pad|>"]
        self.eos_token_id = self.im_end_id
        self.bos_token_id = self.im_start_id

    def encode(self, text: str, allowed_special: Union[Literal["all"], Set[str], str] = "all") -> List[int]:
        _ = allowed_special
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for sp in self.SPECIAL_TOKENS:
                if text.startswith(sp, i):
                    tokens.append(self.stoi[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                char = text[i]
                if char in self.stoi:
                    tokens.append(self.stoi[char])
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.itos[idx] for idx in token_ids if idx in self.itos])

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        if tokenize:
            return self.encode(formatted)
        return formatted


def get_tokenizer(tokenizer_type: str = "bpe", text_corpus: Optional[str] = None):
    """Factory helper to obtain a tokenizer instance."""
    if tokenizer_type.lower() == "bpe":
        try:
            return BPETokenizer()
        except ImportError:
            print("[Warning] tiktoken not found. Falling back to character-level tokenizer.")
            return CharTokenizer(text_corpus or "")
    else:
        return CharTokenizer(text_corpus or "")
