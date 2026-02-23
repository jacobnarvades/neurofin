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
- Model: `meta-llama/Llama-3.1-8B` (32 transformer layers) — **current production model**
- Layers: `24..31` (inclusive), interpreted as transformer block indices (embedding output excluded)
- Hidden size: `4096`
- Sliding context window: `512` tokens
- Extraction batch size: `16` (H100 with float16, no 4-bit quant needed for 8B)
- `use_cache=False` on model forward passes
- bfloat16 compute; `bnb_4bit_compute_dtype` set if `--use-4bit` is passed

Previous model tested: `Qwen/Qwen3-14B-Base` layers 20-28, ctx 256 — comparable
performance to Llama-3.1-8B but slightly weaker on UTS01-03.

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
Reddit comments -> encoding model -> ROI neural features -> ISC filter -> Granger causality filter -> Chronos-2 forecasting

ISC filter: compute inter-subject correlation of predicted ROI activations across all 8 subjects.
Keep only ROIs where mean pairwise ISC > threshold (~0.15). This ensures only text-driven,
consensus neural signals enter the financial model. UTS01-03 will dominate; UTS04-08 add
noise-averaging weight.

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

- **All 8 subjects fully trained** with Llama-3.1-8B (layers 24-31, ctx 512) on Lambda H100.
- Trained model packages downloaded to local: `~/Desktop/neurofin_artifacts/`.
- Lambda instance terminated — no active compute costs.
- `run_all_subjects_llama8b.sh` in repo root for re-running on Lambda if needed.
- **Next step: implement inference pseudo-time pipeline** (`infer.py`) before Reddit text
  can be run through the encoding models for the financial pipeline.

---

## Observed Performance (Llama-3.1-8B, layers 24-31, ctx 512, Lambda H100)

The correct metric for comparing to literature is **mean_top5pct_corr** (mean
correlation of the top 5% of voxels by held-out test-story performance).
`mean_corr` (all voxels) is diluted by subcortical/non-language regions and is
not the right gate metric.

| Subject | n_runs | top5pct | mean_corr | n_voxels (corr mask) | alpha |
|---------|--------|---------|-----------|----------------------|-------|
| UTS01 | 84 | **0.1193** | 0.0251 | 16,358 | 3,981,072 |
| UTS02 | 84 | **0.1044** | 0.0241 | 19,125 | 1,359,356 |
| UTS03 | 84 | **0.1211** | 0.0257 | 20,523 | 3,981,072 |
| UTS04 | 26 | 0.0833 | 0.0092 | 8,377 | 158,489 |
| UTS05 | 27 | 0.0690 | 0.0039 | 6,693 | 464,159 |
| UTS06 | 27 | 0.0672 | 0.0051 | 5,549 | 10 ⚠️ |
| UTS07 | 27 | 0.0666 | 0.0055 | 5,617 | 158,489 |
| UTS08 | 27 | 0.0602 | 0.0049 | 3,515 | 29 ⚠️ |

**Key observations:**
- **Bimodal split**: UTS01-03 (84 stories, ~25K train TRs) are strong; UTS04-08 (~27 stories, ~7.5K TRs) are data-limited.
- UTS06 and UTS08 alphas hit the grid floor (10, 29) — scale-invariance of Pearson r means all alphas gave the same validation score; these fits are technically valid but lightly regularized.
- UTS08 has only 3,515 voxels passing the correlation mask — weakest subject overall.
- **For the financial pipeline: use UTS01-03 as primary signal.** UTS04-08 contribute weaker, noisier features.
- ISC (inter-subject correlation) filtering across subjects will naturally weight UTS01-03 more heavily.

Validation gate: `mean_top5pct_corr >= 0.08` (passed by UTS01-04)

Literature target: ~0.15–0.25 (Antonello et al. 2023, larger models on ds003020).
Current gap (~0.12 vs 0.15+) likely due to model scale (8B vs 65B+) — acceptable for pipeline use.

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
# Full training on Lambda — single subject (Llama-3.1-8B, production config)
PYTHONPATH=src python -m neurofin.train \
  --data-root ds003020 \
  --output-dir artifacts_llama8b_L24-31_ctx512_UTS01 \
  --model-name meta-llama/Llama-3.1-8B \
  --subjects UTS01 \
  --layer-indices 24,25,26,27,28,29,30,31 \
  --context-tokens 512 \
  --n-test-stories 4 \
  --feature-cache-dir feature_cache_llama8b_L24-31_ctx512 \
  --batch-size 16

# All 8 subjects
bash /home/ubuntu/neurofin/run_all_subjects_llama8b.sh
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
- Lambda instance IP changes each session — check dashboard on spin-up.
- Trained model packages (pkl files) saved locally at `~/Desktop/neurofin_artifacts/`.
- Feature cache on Lambda: `~/neurofin/feature_cache_llama8b_L24-31_ctx512/` — lost when instance is terminated; will regenerate from HuggingFace weights on next run (~60 min for UTS01, ~10 min each for UTS02-08 cache hits).
- `BitsAndBytesConfig` uses `bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"` — required for correct 4-bit performance. Without bfloat16 compute, inference is 4-10x slower and uses ~75GB vs ~50GB on H100.
