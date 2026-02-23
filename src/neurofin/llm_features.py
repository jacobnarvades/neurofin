from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LLMFeatureExtractor:
    model_name: str
    layer_indices: list[int]
    device: str = "cuda"
    use_4bit: bool = False
    max_tokens: int = 4096
    context_tokens: int = 1024
    batch_size: int = 32
    smoke_test: bool = False

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model_kwargs: dict = {
            "output_hidden_states": True,
            "trust_remote_code": False,
        }
        if self.use_4bit and self.device.startswith("cuda"):
            self._model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self._model_kwargs["device_map"] = "auto"
        else:
            self._model_kwargs["torch_dtype"] = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = None  # lazy — loaded on first extract_word_features call

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        import sys
        print("Loading LLM weights into GPU...", flush=True, file=sys.stderr)
        self.model = AutoModel.from_pretrained(self.model_name, **self._model_kwargs)
        self.model.eval()
        if not self.use_4bit:
            self.model.to(self.device)

    def extract_word_features(self, words: list[str]) -> np.ndarray:
        """
        Returns array with shape (n_words, n_layers, hidden_dim).
        """
        self._ensure_model_loaded()
        if not words:
            return np.empty((0, len(self.layer_indices), self.model.config.hidden_size), dtype=np.float32)

        if self.smoke_test:
            # Integration-only mode: scientifically invalid features.
            return self._extract_word_features_single_pass(words, max_tokens=512)

        joined_text = " ".join(words)
        enc = self.tokenizer(
            joined_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].cpu().numpy()  # (seq, 2)
        input_ids = enc["input_ids"][0]  # (seq,)
        total_tokens = int(input_ids.shape[0])
        if total_tokens == 0:
            return np.zeros((len(words), len(self.layer_indices), self.model.config.hidden_size), dtype=np.float32)

        if total_tokens > self.max_tokens:
            input_ids = input_ids[: self.max_tokens]
            offsets = offsets[: self.max_tokens]
            total_tokens = int(input_ids.shape[0])

        spans = _word_char_spans(words)
        target_token_idx = _last_token_per_word(spans, offsets)
        hidden_size = int(self.model.config.hidden_size)
        word_features = np.zeros((len(spans), len(self.layer_indices), hidden_size), dtype=np.float32)

        windows: list[tuple[int, torch.Tensor]] = []
        for w_idx, tok_idx in enumerate(target_token_idx):
            if tok_idx is None or tok_idx >= total_tokens:
                continue
            start = max(0, tok_idx - self.context_tokens + 1)
            windows.append((w_idx, input_ids[start : tok_idx + 1]))

        if not windows:
            return word_features

        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        bsz = max(1, int(self.batch_size))
        for i in range(0, len(windows), bsz):
            batch = windows[i : i + bsz]
            max_len = max(int(seq.shape[0]) for _, seq in batch)
            batch_input = torch.full((len(batch), max_len), int(pad_id), dtype=input_ids.dtype)
            batch_attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for row, (_, seq) in enumerate(batch):
                l = int(seq.shape[0])
                batch_input[row, max_len - l :] = seq
                batch_attn[row, max_len - l :] = 1

            model_inputs = {"input_ids": batch_input, "attention_mask": batch_attn}
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            with torch.no_grad():
                out = self.model(**model_inputs, use_cache=False)

            # hidden_states[0] is embedding output; layer indices target transformer blocks.
            block_states = torch.stack(out.hidden_states[1:], dim=0)  # (n_blocks, batch, seq, hidden)
            selected = block_states[self.layer_indices, :, -1, :]  # (n_layers, batch, hidden)
            selected_np = selected.detach().cpu().float().numpy().astype(np.float32)
            for row, (w_idx, _) in enumerate(batch):
                word_features[w_idx] = selected_np[:, row, :]
        return word_features

    def _extract_word_features_single_pass(self, words: list[str], max_tokens: int) -> np.ndarray:
        joined_text = " ".join(words)
        enc = self.tokenizer(
            joined_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].cpu().numpy()  # (seq, 2)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            out = self.model(**model_inputs, use_cache=False)

        # hidden_states[0] is embedding output; layer indices target transformer blocks.
        block_states = torch.stack(out.hidden_states[1:], dim=0)[:, 0]  # (n_blocks, seq, hidden)
        selected_all = block_states[self.layer_indices]  # (n_layers, seq, hidden)
        seq_len = int(selected_all.shape[1])

        spans = _word_char_spans(words)
        target_token_idx = _last_token_per_word(spans, offsets)
        hidden_size = int(self.model.config.hidden_size)
        word_features = np.zeros((len(spans), len(self.layer_indices), hidden_size), dtype=np.float32)

        for w_idx, tok_idx in enumerate(target_token_idx):
            if tok_idx is None or tok_idx >= seq_len:
                continue
            word_features[w_idx] = selected_all[:, tok_idx, :].detach().cpu().float().numpy().astype(np.float32)
        return word_features


def _word_char_spans(words: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = cursor
        end = start + len(word)
        spans.append((start, end))
        cursor = end + 1
    return spans


def _last_token_per_word(
    word_spans: list[tuple[int, int]],
    token_offsets: np.ndarray,
) -> list[int | None]:
    result: list[int | None] = []
    for w_start, w_end in word_spans:
        overlap_idxs: list[int] = []
        for tok_idx, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_end <= tok_start:
                continue
            overlap = min(tok_end, w_end) - max(tok_start, w_start)
            if overlap > 0:
                overlap_idxs.append(tok_idx)
        if overlap_idxs:
            result.append(overlap_idxs[-1])
        else:
            # Fallback to nearest previous token boundary.
            prior = [i for i, (_, tok_end) in enumerate(token_offsets) if tok_end <= w_end]
            result.append(prior[-1] if prior else None)
    return result
