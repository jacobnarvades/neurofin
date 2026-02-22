# Neurofin: Neural-Inspired Encoding Model Training

This repository implements the weight-matrix training stage described in `instructions.txt`:
- Uses HuthLab Narratives (`ds003020`) for supervision.
- Extracts LLM hidden states (default: Qwen3-14B-Base middle layers 20-28).
- Aligns features to fMRI TR with interpolation + FIR delays.
- Trains a voxelwise encoding model with PCA + ridge.
- Saves a reusable inference package for Reddit-comment inference.

## 1. Windows setup (64GB RAM machine)

```powershell
.\scripts\setup_windows.ps1
```

If execution policy blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

## 2. Download Narratives data

```powershell
.\scripts\download_narratives.ps1 -TargetDir ds003020
```

Expected structure:
- `ds003020/stimuli/*.TextGrid`
- `ds003020/derivatives/preprocessed/sub-*/func/*_bold.nii.gz`

## 3. Train encoding model

Prototype run (small):

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m neurofin.train --data-root ds003020 --output-dir artifacts --max-runs 2
```

Full run:

```powershell
python -m neurofin.train --data-root ds003020 --output-dir artifacts --pca-components 500
```

Notes:
- Default uses full compute (no quantization).
- Use `--use-4bit` if you want lower VRAM usage.
- If model access/download is blocked, swap `--model-name` to another accessible checkpoint for smoke tests.

## 4. Single comment inference

```powershell
python -m neurofin.infer --package artifacts/encoding_model_inference_package.pkl --comment "I think this stock could mean-revert after earnings."
```

## Outputs

Training writes:
- `artifacts/encoding_model_inference_package.pkl`
- `artifacts/training_metrics.json`

The package includes:
- learned weights
- fitted PCA
- voxel masks and correlation map
- layer indices, delays, TR metadata
- ROI definitions

## Important implementation notes

- The code attempts to use HuthLab utilities when present (e.g. interpolation/ridge helpers), and falls back to robust local implementations if unavailable.
- NAcc/subcortical-specific ROI handling is not fully resolved by Schaefer cortical atlas alone; current pipeline preserves cortical ROI aggregation and keeps extension points for subcortical masks.
