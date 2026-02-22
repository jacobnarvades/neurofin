from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .llm_features import LLMFeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single-comment inference to ROI features.")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--comment", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-4bit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.package.open("rb") as f:
        package = pickle.load(f)

    model_name = package["model_name"]
    layer_indices = package["layer_indices"]
    delays = package["delays"]

    extractor = LLMFeatureExtractor(
        model_name=model_name,
        layer_indices=layer_indices,
        device=args.device,
        use_4bit=args.use_4bit,
        context_tokens=1024,
    )
    words = args.comment.strip().split()
    hidden = extractor.extract_word_features(words)  # (n_words, n_layers, hidden)
    pooled = hidden.mean(axis=0).reshape(-1)  # (n_layers*hidden,)
    delayed = np.concatenate([pooled for _ in delays], axis=0).reshape(1, -1).astype(np.float32)

    pca = package["pca"]
    weights = package["weights"]
    pred = pca.transform(delayed) @ weights
    pred = pred[0]

    corr_mask = package["voxel_mask_after_corr"]
    pred_masked = pred[corr_mask]
    summary = {
        "mean_predicted_voxel_activation": float(pred_masked.mean()) if pred_masked.size else 0.0,
        "std_predicted_voxel_activation": float(pred_masked.std()) if pred_masked.size else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
