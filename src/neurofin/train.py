from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import TrainingConfig
from .data import discover_story_runs, load_bold_matrix
from .llm_features import LLMFeatureExtractor
from .modeling import fit_encoding_model, select_alpha_with_validation_story, voxelwise_correlation
from .roi import (
    aggregate_parcels_to_rois,
    aggregate_voxels_to_parcels,
    fetch_schaefer_1000,
    set_roi_mapping_context,
)
from .temporal import align_features_to_tr
from .textgrid_utils import load_word_timings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train voxelwise encoding model for neurofin.")
    p.add_argument("--data-root", type=Path, default=Path("ds003020"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B-Base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--pca-components", type=int, default=500)
    p.add_argument("--max-runs", type=int, default=0, help="0 means all discovered runs.")
    p.add_argument("--subjects", type=str, default="", help="Comma-separated subject IDs, e.g. UTS01,UTS02")
    p.add_argument("--context-tokens", type=int, default=256, help="Sliding context window size in tokens.")
    p.add_argument("--batch-size", type=int, default=32, help="Sliding-window LLM batch size.")
    p.add_argument(
        "--layer-indices",
        type=str,
        default="",
        help="Comma-separated layer indices, e.g. '40,41,42,...,55'. Overrides config default.",
    )
    p.add_argument("--feature-cache-dir", type=Path, default=Path("feature_cache"))
    p.add_argument("--smoke-test", action="store_true", help="Integration-only mode; disables sliding context extraction.")
    p.add_argument("--n-test-stories", type=int, default=4)
    p.add_argument("--test-stories", type=str, default="", help="Comma-separated story IDs (task names).")
    p.add_argument("--dry-split", action="store_true", help="Print train/val/test story split and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layer_indices_override = (
        [int(x.strip()) for x in args.layer_indices.split(",") if x.strip()]
        if args.layer_indices.strip()
        else None
    )
    cfg = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        use_4bit=args.use_4bit,
        pca_components=args.pca_components,
        context_tokens=args.context_tokens,
    )
    if layer_indices_override is not None:
        cfg.layer_indices = layer_indices_override
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_story_runs(cfg.derivatives_root, cfg.stimuli_root)
    if not runs:
        raise RuntimeError("No runs found. Check ds003020 path and downloaded derivatives/stimuli.")
    if args.subjects.strip():
        allowed_subjects = {s.strip() for s in args.subjects.split(",") if s.strip()}
        runs = [r for r in runs if r.subject in allowed_subjects]
        if not runs:
            raise RuntimeError(f"No runs found for subjects: {sorted(allowed_subjects)}")
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    story_order = unique_story_order(runs)
    if args.dry_split:
        train_story_ids, val_story_id, test_story_ids = compute_story_splits(
            story_order=story_order,
            test_stories_arg=args.test_stories,
            n_test_stories=args.n_test_stories,
        )
        split_report = {
            "n_runs": len(runs),
            "n_unique_stories": len(story_order),
            "train_stories": train_story_ids,
            "validation_story": val_story_id,
            "test_stories": test_story_ids,
            "n_train_stories": len(train_story_ids),
            "n_test_stories": len(test_story_ids),
        }
        print(json.dumps(split_report, indent=2))
        return

    extractor = LLMFeatureExtractor(
        model_name=cfg.model_name,
        layer_indices=cfg.layer_indices,
        device=cfg.device,
        use_4bit=cfg.use_4bit,
        max_tokens=cfg.max_tokens,
        context_tokens=cfg.context_tokens,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
    )

    story_blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    story_order: list[str] = []
    nonconstant_mask_flat: np.ndarray | None = None
    spatial_shape: tuple[int, int, int] | None = None

    for run in tqdm(runs, desc="Processing runs"):
        try:
            timings = load_word_timings(run.textgrid_path)
        except Exception as e:
            tqdm.write(f"Warning: skipping {run.task} ({run.textgrid_path.name}): {e}")
            continue
        words = [x.word for x in timings]
        word_times = np.array([x.start for x in timings], dtype=np.float32)
        if len(words) == 0:
            continue

        cache_key = {
            "story_id": run.task,
            "subject_id": run.subject,
            "context_tokens": int(cfg.context_tokens),
            "layer_indices": list(cfg.layer_indices),
            "model_name": cfg.model_name,
        }
        cache_path = feature_cache_path(args.feature_cache_dir, cache_key)
        word_features = load_cached_word_features(cache_path, cache_key, len(words))
        if word_features is None:
            word_features = extractor.extract_word_features(words)
            save_cached_word_features(cache_path, cache_key, word_features, len(words))
        x_run = align_features_to_tr(
            word_features=word_features,
            word_times=word_times,
            tr=run.tr,
            n_trs=run.n_trs,
            delays=cfg.delays,
            interpolation_method=cfg.interpolation_method,
            lanczos_window=cfg.lanczos_window,
        )

        y_run, run_spatial_shape = load_bold_matrix(run.bold_path)
        if spatial_shape is None:
            spatial_shape = run_spatial_shape
        elif spatial_shape != run_spatial_shape:
            raise RuntimeError("Runs have mismatched spatial shapes; align/resample first.")

        y_run_z, run_nonconstant_mask = zscore_run_per_voxel(y_run, eps=1e-6)
        if nonconstant_mask_flat is None:
            nonconstant_mask_flat = run_nonconstant_mask.copy()
        else:
            nonconstant_mask_flat &= run_nonconstant_mask

        n = min(x_run.shape[0], y_run_z.shape[0])
        story_id = run.task
        if story_id not in story_blocks:
            story_order.append(story_id)
        story_blocks[story_id].append((x_run[:n].astype(np.float32), y_run_z[:n].astype(np.float32)))

    if not story_blocks:
        raise RuntimeError("No usable runs after preprocessing.")
    if nonconstant_mask_flat is None:
        raise RuntimeError("Could not compute nonconstant voxel mask.")

    for story_id in list(story_blocks.keys()):
        updated: list[tuple[np.ndarray, np.ndarray]] = []
        for x_blk, y_blk in story_blocks[story_id]:
            updated.append((x_blk, y_blk[:, nonconstant_mask_flat]))
        story_blocks[story_id] = updated

    train_story_ids, val_story_id, test_story_ids = compute_story_splits(
        story_order=story_order,
        test_stories_arg=args.test_stories,
        n_test_stories=args.n_test_stories,
    )
    train_candidate_story_ids = train_story_ids + [val_story_id]
    if set(train_story_ids).intersection(test_story_ids):
        raise RuntimeError("Train and test story splits overlap.")
    if val_story_id in test_story_ids or val_story_id in train_story_ids:
        raise RuntimeError("Validation story must be distinct from train and test stories.")

    x_train_core, y_train_core = concat_stories(story_blocks, train_story_ids)
    x_val, y_val = concat_stories(story_blocks, [val_story_id])
    x_train_all, y_train_all = concat_stories(story_blocks, train_candidate_story_ids)
    x_test, y_test = concat_stories(story_blocks, test_story_ids)

    if x_train_all.shape[1] % len(cfg.delays) != 0:
        raise RuntimeError("Feature dimension is not divisible by number of delays.")
    feature_dim_per_delay = x_train_all.shape[1] // len(cfg.delays)

    alpha_grid = np.logspace(cfg.ridge_alphas_log10_min, cfg.ridge_alphas_log10_max, cfg.ridge_alpha_count)
    selected_alpha, alpha_scores = select_alpha_with_validation_story(
        x_train_core=x_train_core,
        y_train_core=y_train_core,
        x_val_story=x_val,
        y_val_story=y_val,
        pca_components=min(cfg.pca_components, x_train_core.shape[0], x_train_core.shape[1]),
        alpha_grid=alpha_grid,
    )
    model = fit_encoding_model(
        x_train=x_train_all,
        y_train=y_train_all,
        pca_components=min(cfg.pca_components, x_train_all.shape[0], x_train_all.shape[1]),
        alpha=selected_alpha,
    )
    y_pred = model.predict(x_test)
    corr = voxelwise_correlation(y_pred, y_test)
    keep_voxels = corr >= cfg.voxel_corr_threshold

    first_hf5_path = next((r.bold_path for r in runs if r.bold_path.suffix.lower() == ".hf5"), None)
    set_roi_mapping_context(first_hf5_path, cfg.derivatives_root)
    atlas = fetch_schaefer_1000()
    schaefer_data = np.asarray(atlas.parcel_image.get_fdata(), dtype=np.int32)

    if spatial_shape is None or nonconstant_mask_flat is None:
        raise RuntimeError("Missing spatial metadata for ROI aggregation.")

    full_keep_flat = np.zeros(nonconstant_mask_flat.shape[0], dtype=bool)
    full_keep_flat[nonconstant_mask_flat] = keep_voxels
    full_keep_3d = full_keep_flat.reshape(spatial_shape)

    mean_pred_voxels = y_pred.mean(axis=0)[keep_voxels]
    parcel_values = aggregate_voxels_to_parcels(
        voxel_values=mean_pred_voxels,
        voxel_mask_3d=full_keep_3d,
        schaefer_data=schaefer_data,
    )
    if parcel_values.ndim == 2:
        parcel_values_for_roi = parcel_values.mean(axis=0)
    else:
        parcel_values_for_roi = parcel_values
    roi_values = aggregate_parcels_to_rois(parcel_values_for_roi, atlas.parcel_labels)

    package = {
        "weights": model.weights,
        "scaler": model.scaler,
        "pca": model.pca,  # alias kept for backward compat
        "voxel_mask_after_var": nonconstant_mask_flat,
        "voxel_mask_after_corr": keep_voxels,
        "spatial_shape": spatial_shape,
        "schaefer_labels": atlas.parcel_labels,
        "roi_definitions": atlas.roi_definitions,
        "layer_indices": cfg.layer_indices,
        "delays": cfg.delays,
        "tr": float(cfg.target_tr),
        "sentence_rate_hz": float(cfg.sentence_rate_hz),
        "interpolation_method": cfg.interpolation_method,
        "hrf_model": cfg.hrf_model,
        "pca_n_components": int(model.pca.n_components_),
        "feature_dim_per_delay": int(feature_dim_per_delay),
        "context_tokens": int(cfg.context_tokens),
        "model_name": cfg.model_name,
        "correlation_map": corr,
        "alpha": model.alpha,
        "alpha_scores": alpha_scores,
        "train_stories": train_story_ids,
        "validation_story": val_story_id,
        "test_stories": test_story_ids,
        "example_roi_values": roi_values,
    }

    out_pkl = cfg.output_dir / "encoding_model_inference_package.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(package, f)

    mean_positive_corr = float(corr[corr > 0].mean()) if (corr > 0).any() else 0.0
    top5_thresh = float(np.percentile(corr, 95))
    mean_top5pct_corr = float(corr[corr >= top5_thresh].mean())

    metrics = {
        "n_runs": len(runs),
        "n_train_trs": int(x_train_all.shape[0]),
        "n_test_trs": int(x_test.shape[0]),
        "n_features": int(x_train_all.shape[1]),
        "n_voxels_after_variance_mask": int(nonconstant_mask_flat.sum()),
        "n_voxels_after_corr_mask": int(keep_voxels.sum()),
        "mean_corr": float(corr.mean()),
        "mean_positive_corr": mean_positive_corr,
        "mean_top5pct_corr": mean_top5pct_corr,
        "alpha": None if model.alpha is None else float(model.alpha),
        "train_stories": train_story_ids,
        "validation_story": val_story_id,
        "test_stories": test_story_ids,
        "example_roi_values": roi_values,
    }
    with (cfg.output_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved package: {out_pkl}")
    print(f"Mean voxelwise r (all): {metrics['mean_corr']:.4f}")
    print(f"Mean voxelwise r (positive): {metrics['mean_positive_corr']:.4f}")
    print(f"Mean voxelwise r (top 5%): {metrics['mean_top5pct_corr']:.4f}")
    print(f"Train stories: {train_story_ids}")
    print(f"Validation story: {val_story_id}")
    print(f"Test stories: {test_story_ids}")


def zscore_run_per_voxel(y_run: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score each voxel within a run across timepoints.
    Returns z-scored run and a boolean mask of nonconstant voxels.
    Computation done in float64 to avoid float32 rounding drift on large
    voxel arrays (81K voxels × T TRs accumulates enough error to push
    float32 mean away from 0 by more than float32 machine epsilon).
    """
    y64 = y_run.astype(np.float64)
    mean = y64.mean(axis=0)
    std = y64.std(axis=0)
    nonconstant = std > eps
    std_safe = std.copy()
    std_safe[~nonconstant] = 1.0
    z = (y64 - mean) / std_safe
    return z.astype(np.float32), nonconstant


def concat_stories(
    story_blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    story_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for story_id in story_ids:
        for x_blk, y_blk in story_blocks[story_id]:
            x_parts.append(x_blk)
            y_parts.append(y_blk)
    if not x_parts:
        raise RuntimeError(f"No blocks found for stories: {story_ids}")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def unique_story_order(runs: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for run in runs:
        if run.task not in seen:
            seen.add(run.task)
            out.append(run.task)
    return out


def compute_story_splits(
    story_order: list[str],
    test_stories_arg: str,
    n_test_stories: int,
) -> tuple[list[str], str, list[str]]:
    if test_stories_arg.strip():
        test_story_ids = [s.strip() for s in test_stories_arg.split(",") if s.strip()]
        missing = [s for s in test_story_ids if s not in story_order]
        if missing:
            raise RuntimeError(f"Requested test stories not found: {missing}")
    else:
        if len(story_order) <= n_test_stories + 1:
            raise RuntimeError("Not enough stories to create train, validation, and test splits.")
        test_story_ids = story_order[-n_test_stories:]

    train_candidate_story_ids = [s for s in story_order if s not in test_story_ids]
    if len(train_candidate_story_ids) < 2:
        raise RuntimeError("Need at least 2 non-test stories to create train and validation splits.")

    val_story_id = train_candidate_story_ids[-1]
    train_story_ids = train_candidate_story_ids[:-1]
    return train_story_ids, val_story_id, test_story_ids


def feature_cache_path(cache_dir: Path, cache_key: dict) -> Path:
    key_json = json.dumps(cache_key, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()[:16]
    story = cache_key["story_id"]
    subject = cache_key["subject_id"]
    return cache_dir / f"{subject}__{story}__{key_hash}.pkl"


def load_cached_word_features(cache_path: Path, cache_key: dict, n_words: int) -> np.ndarray | None:
    # LLM features depend only on text, not on which subject heard it.
    # If the subject-specific file doesn't exist, scan for any subject's
    # cache for the same story/model/context so all subjects share UTS01's
    # already-extracted features without re-running the LLM.
    path = cache_path
    if not path.exists():
        path = _find_cross_subject_cache(cache_path.parent, cache_key) or cache_path
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        # Validate subject-invariant fields only.
        stored = payload.get("key", {})
        for field in ("story_id", "context_tokens", "layer_indices", "model_name"):
            if stored.get(field) != cache_key.get(field):
                return None
        if int(payload.get("n_words", -1)) != int(n_words):
            return None
        arr = np.asarray(payload["word_features"], dtype=np.float32)
        if arr.ndim != 3:
            return None
        return arr
    except Exception:
        return None


def _find_cross_subject_cache(cache_dir: Path, cache_key: dict) -> Path | None:
    """Return any subject's cache file for the same story and model params."""
    story = cache_key["story_id"]
    for candidate in cache_dir.glob(f"*__{story}__*.pkl"):
        return candidate
    return None


def save_cached_word_features(cache_path: Path, cache_key: dict, word_features: np.ndarray, n_words: int) -> None:
    payload = {
        "key": cache_key,
        "n_words": int(n_words),
        "word_features": np.asarray(word_features, dtype=np.float32),
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
