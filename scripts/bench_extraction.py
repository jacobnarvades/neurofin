from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic extraction throughput benchmark.")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B-Base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--context-tokens", type=int, default=256)
    p.add_argument("--passes", type=int, default=50)
    p.add_argument("--batch-sizes", type=str, default="1,32")
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def load_model(model_name: str, device: str, use_4bit: bool) -> AutoModel:
    kwargs: dict = {
        "output_hidden_states": True,
        "trust_remote_code": False,
    }
    if use_4bit and device.startswith("cuda"):
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    if not use_4bit:
        model.to(device)
    return model


def benchmark(
    model: AutoModel,
    batch_size: int,
    context_tokens: int,
    passes: int,
    warmup: int,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)

    # Warmup passes to stabilize timings.
    for _ in range(max(0, warmup)):
        input_ids = torch.randint(0, vocab_size, (batch_size, context_tokens), dtype=torch.long, device=device)
        attn = torch.ones((batch_size, context_tokens), dtype=torch.long, device=device)
        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(passes):
        input_ids = torch.randint(0, vocab_size, (batch_size, context_tokens), dtype=torch.long, device=device)
        attn = torch.ones((batch_size, context_tokens), dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            # Match extraction workload: gather last-token hidden states for all blocks.
            _ = torch.stack(out.hidden_states[1:], dim=0)[:, :, -1, :]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_words = float(batch_size * passes)
    words_per_sec = total_words / max(elapsed, 1e-9)
    return elapsed, words_per_sec


def main() -> None:
    args = parse_args()
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if not batch_sizes:
        raise ValueError("No batch sizes provided.")

    model = load_model(args.model_name, args.device, args.use_4bit)
    print(
        f"Benchmark model={args.model_name} device={next(model.parameters()).device} "
        f"context_tokens={args.context_tokens} passes={args.passes}"
    )
    for bsz in batch_sizes:
        elapsed, wps = benchmark(
            model=model,
            batch_size=bsz,
            context_tokens=args.context_tokens,
            passes=args.passes,
            warmup=args.warmup,
        )
        print(f"batch_size={bsz} elapsed_sec={elapsed:.2f} words_per_sec={wps:.2f}")


if __name__ == "__main__":
    main()

