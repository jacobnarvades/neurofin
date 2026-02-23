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
- Model: `Qwen/Qwen3-14B-Base` (40 transformer layers)
- Layers: `20..28` (inclusive), interpreted as transformer block indices (embedding output excluded)
- Hidden size: `5120`
- Sliding context window: `256` tokens
- Extraction batch size: `16` (not 32 — H100 with float16, no 4-bit quant)
- `use_cache=False` on model forward passes
- Feature std at layers 20-28: ~4-6 (healthy; confirmed on Lambda)

### Training Data
- Dataset: HuthLab Narratives (`OpenNeuro ds003020`)
- 8 subjects: UTS01–UTS08
- Story inventory: 84 story IDs per subject
- TR: **2.0045s** (not 1.5s — hardcoded correctly in `data.py` fallback)
- hf5 shape: `(n_trs, 81126)` — time-first, 81126 voxels for UTS01
- hf5 files do NOT store TR in attributes; fallback to 2.0045 is correct

### Temporal Encoding Pipeline
1. Word-level contextual embeddings (Lanczos-interpolated to TR grid)
2. FIR delay stacking with delays `[1, 2, 3, 4]` TRs
- **No HRF convolution** — FIR delays implicitly model the HRF (Huth et al. 2016).
  Applying both HRF convolution AND FIR delays makes all delay channels nearly
  identical (HRF spans ~10-15 TRs; 1-4 TR shift barely changes it), causing
  ill-conditioned ridge regression and near-zero predictions. FIR only.

### Regression Pipeline
- PCA to `500` components, then **per-component z-scoring** before ridge
  (PCA produces orthogonal unit-variance components after z-scoring; ridge
  penalty is then applied equally to all 500 components)
- Ridge alpha grid: `10^1` to `10^8`, 16 values log-spaced
- Alpha selected on held-out validation story (mean voxelwise Pearson r)
- Note: with PCA + z-scoring, X^T X ≈ n·I so all alphas give the same
  Pearson r (scale-invariant). Alpha selection picks the first (smallest).
  This is expected and not a bug.
- Final fit on train+validation stories
- Evaluate on held-out test stories
- Target: voxelwise prediction

### Atlas + ROI
- Schaefer atlas: `1000` parcels
- Atlas resampled to native mask space when needed (`nearest` interpolation)
- Native mask: `mask_thick.nii.gz` at `pycortex-db/<SUBJ>/transforms/<SUBJ>_auto/`
- UTS01: 81126 nonzero voxels
- ROI groups: default_mode, frontoparietal, salience, dorsal_attention, limbic, language
- Mask downloaded via: `aws s3 sync s3://openneuro.org/ds003020/derivative/pycortex-db/$SUBJ/transforms ...`

### Financial Pipeline (downstream)
Reddit comments -> encoding model -> ROI neural features -> Granger causality filter -> Chronos-2 forecasting

---

## Key Implementation Decisions (and Why)

1. **FIR delays only, no HRF convolution**
- HRF + FIR double-application was a critical bug. Removed HRF convolution.
- FIR with ridge implicitly learns optimal hemodynamic weighting per voxel.

2. **TR = 2.0045s hardcoded fallback**
- ds003020 hf5 files have no TR attribute. Fallback in `_read_hf5_tr` returns 2.0045.
- Using 1.5s caused 33% temporal misalignment and destroyed encoding performance.

3. **Feature cache is subject-independent**
- LLM features depend only on story text (TextGrid), not on which subject listened.
- Cache key: `(story_id, context_tokens, layer_indices, model_name)`.
- Cross-subject fallback in `load_cached_word_features` scans the cache dir for any
  subject's file matching the story, so UTS02-UTS08 reuse UTS01's extracted features.
- This cuts per-subject training from ~55 min to ~3 min after UTS01 is cached.

4. **Story-level split (not row-level split)**
- Prevents temporal autocorrelation leakage across adjacent fMRI timepoints.

5. **Validation story separate from test stories**
- Alpha selection uses validation-only signal, avoids optimistic bias.

6. **Per-run per-voxel z-scoring before concatenation**
- Removes run-specific baseline/scanner drift.
- Applied independently per run, including test runs.

7. **Per-PC z-scoring before ridge**
- PCA components have very different eigenvalue-based variances.
- Without z-scoring, high-variance PCs dominate; low-variance PCs (which may carry
  brain-relevant signal) get suppressed by the L2 penalty.
- After z-scoring, all 500 components contribute equally.

8. **Local Lanczos implementation**
- `ridge_utils` may be unavailable; local `np.sinc` Lanczos is implemented.
- Warning about unavailable ridge_utils is expected and valid.

9. **Mask discovery by voxel-count match**
- For flat `.hf5` data, mask is selected by matching nonzero voxel count.

10. **`use_cache=False` in all forward passes**
- No useful cross-pass KV reuse in independent sliding windows.

---

## Current Status

- UTS01 full training complete on Lambda Labs H100 instance.
- All 8 subjects (UTS01–UTS08) training in progress via `run_all_subjects.sh`.
- `bootstrap.sh` exists in repo root for Lambda setup and full training bootstrap.
- Feature cache (84 stories, ~UTS01) fully built at `feature_cache/`.

---

## Observed Performance (UTS01, Lambda H100)

The correct metric for comparing to literature is **mean_top5pct_corr** (mean
correlation of the top 5% of voxels by held-out test-story performance).
`mean_corr` (all voxels) is diluted by subcortical/non-language regions and is
not the right gate metric.

| Metric | Value | Notes |
|--------|-------|-------|
| `mean_corr` (all voxels) | ~0.026 | Expected — most voxels don't respond to language |
| `mean_positive_corr` | ~0.060 | Mean over voxels with r > 0 |
| `mean_top5pct_corr` | ~0.13–0.15 | Comparable to Antonello et al. 2023 |
| Max voxel r | ~0.31 | Language/auditory cortex peak voxels |
| `n_voxels_after_corr_mask` | ~17K | Voxels with r >= 0.05 |

Validation gate (in `bootstrap.sh` / `run_all_subjects.sh`) checks:
`mean_top5pct_corr >= 0.08`

Literature target: ~0.15–0.25 (Antonello et al. 2023, Qwen2-class models on ds003020).

---

## TextGrid Handling

Four fallback levels in `textgrid_utils.py`:
1. praatio direct parse
2. praatio with interval boundary repair (fixes floating-point overlaps)
3. Regex extraction of words tier only (bypasses cross-tier validation — fixes `wheretheressmoke.TextGrid` phones-tier overlap)
4. Raise error

Also handles:
- Praat chronological format (detected by "chronological" in first 80 bytes)
- BOM (`utf-8-sig` encoding throughout)

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
3. Lanczos warning (ridge_utils unavailable) is expected and valid — ignore it.

---

## Running Training

```bash
# Full training on Lambda (single subject)
PYTHONPATH=src python -m neurofin.train \
  --data-root ds003020 \
  --output-dir artifacts_UTS01 \
  --subjects UTS01 \
  --context-tokens 256 \
  --n-test-stories 4 \
  --feature-cache-dir feature_cache \
  --batch-size 16

# All 8 subjects (use run_all_subjects.sh on Lambda)
bash /home/ubuntu/neurofin/run_all_subjects.sh
```

---

## Notes for Fresh Sessions

- Use `PYTHONPATH=src` when running from source tree if editable install is unavailable.
- Dataset layout:
  - `ds003020/derivative/preprocessed_data/<SUBJECT>/*.hf5`
  - `ds003020/derivative/TextGrids/*.TextGrid`
  - `ds003020/derivative/pycortex-db/<SUBJECT>/transforms/` (needed for mask)
- For cloud runs, download directly from OpenNeuro S3; do not upload local datasets.
- Always use `/home/ubuntu/neurofin/.venv/bin/python3` (not system python3) when
  inspecting pkl files — numpy version mismatch will cause `ModuleNotFoundError: numpy._core`.
- Lambda instance: `ubuntu@192-222-54-157` (may change between sessions).
- Feature cache at `~/neurofin/feature_cache/` — 84 pkl files for UTS01 stories.
  UTS02-UTS08 reuse these via cross-subject fallback; no re-extraction needed.
