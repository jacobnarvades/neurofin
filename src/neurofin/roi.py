from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import resample_to_img


@dataclass
class RoiPackage:
    parcel_image: nib.Nifti1Image
    parcel_labels: list[str]
    roi_definitions: dict[str, str | list[str]]


ROI_NETWORKS: dict[str, str | list[str]] = {
    "default_mode": "Default",
    "frontoparietal": "Cont",
    "salience": "SalVentAttn",
    "dorsal_attention": "DorsAttn",
    "limbic": "Limbic",
    "language": ["Temp", "PFC"],
}


_CONTEXT_HF5_PATH: Path | None = None
_CONTEXT_DERIV_ROOT: Path | None = None
_PARCEL_MAP_CACHE: dict[tuple[str, int, tuple[int, int, int]], np.ndarray] = {}
_RESAMPLED_ATLAS_CACHE: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}


def set_roi_mapping_context(hf5_path: Path | None, derivatives_root: Path | None) -> None:
    global _CONTEXT_HF5_PATH, _CONTEXT_DERIV_ROOT
    _CONTEXT_HF5_PATH = hf5_path
    _CONTEXT_DERIV_ROOT = derivatives_root


def fetch_schaefer_1000() -> RoiPackage:
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000)
    img = nib.load(atlas.maps)
    mask_path = _find_mask_file_for_context()
    if mask_path is not None:
        mask_img = nib.load(str(mask_path))
        if tuple(mask_img.shape) != tuple(img.shape):
            img = resample_to_img(
                img,
                mask_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
    labels = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in atlas.labels]
    return RoiPackage(parcel_image=img, parcel_labels=labels, roi_definitions=ROI_NETWORKS)


def aggregate_voxels_to_parcels(
    voxel_values: np.ndarray,
    voxel_mask_3d: np.ndarray,
    schaefer_data: np.ndarray,
) -> np.ndarray:
    """
    voxel_values: (n_selected_voxels,)
    voxel_mask_3d: bool array in the same 3D space as Schaefer atlas
    schaefer_data: int parcel id image
    """
    values = np.asarray(voxel_values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("voxel_values must be 1D or 2D")

    selected_flat = np.flatnonzero(voxel_mask_3d.reshape(-1))
    if selected_flat.shape[0] != values.shape[1]:
        raise ValueError("voxel_values and voxel_mask size mismatch")

    n_timepoints = values.shape[0]
    out = np.zeros((n_timepoints, 1000), dtype=np.float32)

    # Case 1: data already in full 3D atlas space.
    if voxel_mask_3d.shape == schaefer_data.shape:
        atlas_flat = schaefer_data.reshape(-1).astype(np.int32)
        selected_parcels = atlas_flat[selected_flat]
    else:
        # Case 2: flat .hf5 voxel space; map flat indices -> atlas parcel ids.
        parcel_ids_for_flat = _get_or_build_flat_parcel_ids(
            n_flat_total=voxel_mask_3d.size,
            schaefer_data=schaefer_data,
        )
        selected_parcels = parcel_ids_for_flat[selected_flat]

    for parcel_id in range(1, 1001):
        idx = np.flatnonzero(selected_parcels == parcel_id)
        if idx.size:
            out[:, parcel_id - 1] = values[:, idx].mean(axis=1)

    return out


def aggregate_parcels_to_rois(parcel_values: np.ndarray, labels: list[str]) -> dict[str, float]:
    results: dict[str, float] = {}
    for roi_name, rule in ROI_NETWORKS.items():
        idx = _select_parcels(labels, rule)
        if not idx:
            results[roi_name] = 0.0
            continue
        idx_arr = np.asarray(idx, dtype=np.int64)
        # Be robust to accidental 1-based parcel IDs.
        if idx_arr.size and idx_arr.max() >= parcel_values.shape[0] and idx_arr.min() >= 1:
            idx_arr = idx_arr - 1
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < parcel_values.shape[0])]
        if idx_arr.size == 0:
            results[roi_name] = 0.0
            continue
        results[roi_name] = float(np.mean(parcel_values[idx_arr]))
    return results


def _select_parcels(labels: list[str], rule: str | list[str]) -> list[int]:
    out: list[int] = []
    if isinstance(rule, str):
        rule_lower = rule.lower()
        for i, label in enumerate(labels):
            if rule_lower in label.lower():
                out.append(i)
    else:
        rule_parts = [r.lower() for r in rule]
        for i, label in enumerate(labels):
            low = label.lower()
            if any(part in low for part in rule_parts):
                out.append(i)
    return out


def _get_or_build_flat_parcel_ids(n_flat_total: int, schaefer_data: np.ndarray) -> np.ndarray:
    mask_candidates = _find_mask_files_for_context()
    if not mask_candidates:
        raise RuntimeError(
            "No brain mask found for flat .hf5 data. Expected mask.nii.gz/brain_mask.nii.gz/*_mask.nii.gz "
            "in hf5 dir, subject derivatives folder, or derivatives root."
        )
    mask_path = _pick_mask_by_voxel_count(mask_candidates, n_flat_total)
    if mask_path is None:
        counts = []
        for p in mask_candidates[:10]:
            try:
                c = int((np.asarray(nib.load(str(p)).get_fdata()) > 0).sum())
                counts.append(f"{p.name}:{c}")
            except Exception:
                counts.append(f"{p.name}:<unreadable>")
        raise RuntimeError(
            f"No mask with voxel count {n_flat_total} found for flat .hf5 data. "
            f"Top candidates: {', '.join(counts)}"
        )

    key = (str(mask_path), n_flat_total, tuple(schaefer_data.shape))
    cached = _PARCEL_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.get_fdata()) > 0
    mask_idx = np.flatnonzero(mask_data.reshape(-1))
    if mask_idx.size != n_flat_total:
        raise RuntimeError(
            f"Mask voxel count ({mask_idx.size}) does not match flat voxel count ({n_flat_total}) for {mask_path}."
        )

    atlas_flat = _atlas_flat_for_mask_space(mask_path, mask_img, schaefer_data)

    parcel_ids = atlas_flat[mask_idx]
    _PARCEL_MAP_CACHE[key] = parcel_ids
    return parcel_ids


def _find_mask_files_for_context() -> list[Path]:
    patterns = (
        "mask_thick.nii.gz",
        "mask_cortical.nii.gz",
        "cortical_mask.nii.gz",
        "mask.nii.gz",
        "brain_mask.nii.gz",
        "*_mask.nii.gz",
    )
    hf5 = _CONTEXT_HF5_PATH
    deriv_root = _CONTEXT_DERIV_ROOT
    hits: list[Path] = []
    seen: set[str] = set()

    def add(paths: list[Path]) -> None:
        for p in paths:
            k = str(p)
            if k not in seen:
                seen.add(k)
                hits.append(p)

    # 1) same directory as .hf5
    if hf5 is not None:
        same_dir = hf5.parent
        add(_all_matches(same_dir, patterns, recursive=False))

    # 2) subject derivatives folder
    if hf5 is not None and deriv_root is not None:
        subject = hf5.parent.name
        subject_dirs = [
            deriv_root / subject,
            deriv_root / "pycortex-db" / subject,
            deriv_root / "freesurfer_subjdir" / subject,
        ]
        for d in subject_dirs:
            add(_all_matches(d, patterns, recursive=True))

    # 3) derivatives root
    if deriv_root is not None:
        add(_all_matches(deriv_root, patterns, recursive=True))

    return hits


def _find_mask_file_for_context() -> Path | None:
    matches = _find_mask_files_for_context()
    return matches[0] if matches else None


def _first_match(base: Path, patterns: tuple[str, ...], recursive: bool) -> Path | None:
    if not base.exists():
        return None
    for pat in patterns:
        it = base.rglob(pat) if recursive else base.glob(pat)
        for p in it:
            if p.is_file():
                return p
    return None


def _all_matches(base: Path, patterns: tuple[str, ...], recursive: bool) -> list[Path]:
    out: list[Path] = []
    if not base.exists():
        return out
    for pat in patterns:
        it = base.rglob(pat) if recursive else base.glob(pat)
        for p in it:
            if p.is_file():
                out.append(p)
    return out


def _pick_mask_by_voxel_count(mask_paths: list[Path], n_flat_total: int) -> Path | None:
    for p in mask_paths:
        try:
            mask = np.asarray(nib.load(str(p)).get_fdata()) > 0
            if int(mask.sum()) == n_flat_total:
                return p
        except Exception:
            continue
    return None


def _atlas_flat_for_mask_space(
    mask_path: Path,
    mask_img: nib.Nifti1Image,
    schaefer_data: np.ndarray,
) -> np.ndarray:
    cache_key = (str(mask_path), tuple(mask_img.shape))
    cached = _RESAMPLED_ATLAS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if tuple(mask_img.shape) == tuple(schaefer_data.shape):
        atlas_flat = schaefer_data.reshape(-1).astype(np.int32)
        _RESAMPLED_ATLAS_CACHE[cache_key] = atlas_flat
        return atlas_flat

    # Resample Schaefer atlas from MNI space to native mask space (e.g., 256^3).
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=1000)
    schaefer_img = nib.load(schaefer.maps)
    resampled = resample_to_img(
        schaefer_img,
        mask_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_flat = np.asarray(resampled.get_fdata(), dtype=np.int32).reshape(-1)
    _RESAMPLED_ATLAS_CACHE[cache_key] = atlas_flat
    return atlas_flat
