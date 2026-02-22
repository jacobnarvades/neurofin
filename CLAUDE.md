# CLAUDE.md

## Project Overview
`neurofin` is a neuroscience-informed trading signal pipeline.

Core goal:
- Predict brain activation patterns from Reddit text using a pretrained fMRI encoding model.
- Aggregate predicted neural activations into ROI features.
- Use ROI features as covariates for financial forecasting.

Core hypothesis:
- Predicted activations in reward/salience/language/default-mode-related regions capture sentiment and cognition dimensions that correlate with stock returns.

---

## Architecture

### LLM Feature Extractor
- Model: `Qwen/Qwen3-14B-Base`
- Layers: `20..28` (inclusive), interpreted as transformer block indices (embedding output excluded)
- Hidden size: `5120`
- Sliding context window: `256` tokens
- Extraction batch size: `32`
- `use_cache=False` on model forward passes

### Training Data
- Dataset: HuthLab Narratives (`OpenNeuro ds003020`)
- Current scope: `UTS01` first
- Story inventory currently discovered: up to `84` story IDs (dataset release includes more than earlier 27-story core subset references)

### Temporal Encoding Pipeline
1. Word-level contextual embeddings
2. Lanczos interpolation to TR grid
3. HRF convolution (SPM canonical double-gamma)
4. FIR delay stacking with delays `[1, 2, 3, 4]` TRs

### Regression Pipeline
- PCA to `500` components
- Ridge model with alpha selected on held-out validation story
- Final fit on train+validation stories
- Evaluate on held-out test stories
- Target: voxelwise prediction

### Atlas + ROI
- Schaefer atlas: `1000` parcels
- Atlas resampled to native mask space when needed (`nearest` interpolation)
- Native mask target for UTS01 flat voxel data: `mask_thick.nii.gz` with `81126` nonzero voxels
- ROI groups tracked in code include default_mode, frontoparietal, salience, dorsal_attention, limbic, language

### Financial Pipeline (downstream)
Reddit comments -> encoding model -> ROI neural features -> Granger causality filter -> Chronos-2 forecasting

---

## Key Implementation Decisions (and Why)

1. Story-level split (not row-level split)
- Prevents temporal autocorrelation leakage across adjacent fMRI timepoints.

2. Validation story separate from test stories
- Alpha selection uses validation-only signal, avoids optimistic bias.

3. Per-run per-voxel z-scoring before concatenation
- Removes run-specific baseline/scanner drift.
- Applied independently per run, including test runs (fMRI preprocessing rationale, not train-derived scaling).

4. Feature cache key
- Keyed by `(story_id, subject_id, context_tokens, layer_indices, model_name)`.
- Avoids expensive re-extraction when rerunning PCA/ridge/splits.

5. Local Lanczos implementation
- `ridge_utils` may be unavailable; local `np.sinc` Lanczos is implemented and used when needed.

6. Mask discovery by voxel-count match
- For flat `.hf5` data, mask is selected by matching nonzero voxel count to flat voxel dimension.
- This avoids incorrect mask selection in mixed-space directories.

7. Schaefer indexing correctness
- Parcel labels are conceptually 1-indexed; arrays are 0-indexed.
- Index handling guards ensure no off-by-one parcel indexing errors.

8. `use_cache=False` in all forward passes
- No useful cross-pass KV reuse in independent sliding windows.
- Prevents unnecessary memory overhead.

---

## Current Status

- Local smoke test path passes end-to-end with `--smoke-test`.
- Full scientific training is compute-intensive and intended for cloud (Lambda Labs H100 workflow prepared).
- `bootstrap.sh` exists in repo root for Lambda setup and UTS01 training bootstrap.

---

## Expected Validation Targets

- Language-region held-out correlations should typically be in roughly `0.15 - 0.35`.
- `mean_test_r < 0.05` strongly suggests pipeline failure.
- `0.05 - 0.15` may be plausible but warrants investigation of model/data/alignment settings.

---

## Inference Status (Important)

`infer.py` has a known FIR-related design issue:
- It currently duplicates pooled static vectors across delays rather than using genuine temporal delay features.
- This is not faithful to training-time delayed feature structure.

Planned fix:
- Sentence pseudo-time inference branch is planned, not implemented yet.

Current instruction:
- Do **not** treat current `infer.py` outputs as final production neural features until pseudo-time inference is implemented.

---

## Known Issues / Pending Work

1. Inference pseudo-time pipeline not implemented.
2. ROI aggregation output not fully wired into inference production path.
3. Multi-subject training expansion pending (UTS01 first).
4. Lanczos warning can appear when optional helper is absent; local Lanczos path is expected and valid.

---

## Running Training

```bash
# Local smoke test
python -m neurofin.train \
  --data-root ds003020 \
  --output-dir artifacts_uts01_smoke \
  --use-4bit \
  --subjects UTS01 \
  --max-runs 5 \
  --n-test-stories 2 \
  --smoke-test

# Full training on Lambda
python -m neurofin.train \
  --data-root ds003020 \
  --output-dir artifacts_uts01 \
  --use-4bit \
  --subjects UTS01 \
  --context-tokens 256 \
  --n-test-stories 4 \
  --feature-cache-dir feature_cache \
  --batch-size 32
```

---

## Notes for Fresh Sessions

- Use `PYTHONPATH=src` when running from source tree if editable install is unavailable.
- Dataset layout currently used in this project is HuthLab-style:
  - `ds003020/derivative/preprocessed_data/<SUBJECT>/*.hf5`
  - `ds003020/derivative/TextGrids/*.TextGrid`
- For cloud runs, download directly from OpenNeuro S3 on the instance; do not upload local datasets.
