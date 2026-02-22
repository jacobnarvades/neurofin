from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


@dataclass
class StoryRun:
    subject: str
    task: str
    bold_path: Path
    textgrid_path: Path
    tr: float
    n_trs: int


def discover_story_runs(derivatives_root: Path, stimuli_root: Path) -> list[StoryRun]:
    runs: list[StoryRun] = []
    for bold_path in derivatives_root.rglob("*_bold.nii.gz"):
        name = bold_path.name
        if "_task-" not in name:
            continue
        subject = _extract_between(name, "sub-", "_")
        task = _extract_between(name, "_task-", "_")
        textgrid = _find_textgrid(stimuli_root, task)
        if textgrid is None:
            continue
        img = nib.load(str(bold_path))
        zooms = img.header.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else 1.5
        n_trs = int(img.shape[3]) if len(img.shape) > 3 else 0
        runs.append(
            StoryRun(
                subject=subject,
                task=task,
                bold_path=bold_path,
                textgrid_path=textgrid,
                tr=tr,
                n_trs=n_trs,
            )
        )

    # HuthLab preprocessed format: derivative/preprocessed_data/UTSxx/<story>.hf5
    for hf5_path in derivatives_root.rglob("preprocessed_data/*/*.hf5"):
        subject = hf5_path.parent.name
        task = hf5_path.stem
        textgrid = _find_textgrid(stimuli_root, task)
        if textgrid is None:
            continue
        with h5py.File(hf5_path, "r") as f:
            if "data" not in f:
                continue
            n_trs = int(f["data"].shape[0])
        runs.append(
            StoryRun(
                subject=subject,
                task=task,
                bold_path=hf5_path,
                textgrid_path=textgrid,
                tr=1.5,
                n_trs=n_trs,
            )
        )
    return runs


def load_bold_matrix(bold_path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Returns:
      y: (n_trs, n_voxels)
      spatial_shape: (x, y, z)
    """
    if bold_path.suffix.lower() == ".hf5":
        with h5py.File(bold_path, "r") as f:
            if "data" not in f:
                raise ValueError(f"Expected dataset 'data' in {bold_path}")
            y = np.asarray(f["data"], dtype=np.float32)
        if y.ndim != 2:
            raise ValueError(f"Expected 2D hf5 data (time, voxels), got shape {y.shape}")
        # No volumetric shape in this preprocessed format; keep a reversible flat shape placeholder.
        spatial_shape = (1, y.shape[1], 1)
        return y, spatial_shape

    img = nib.load(str(bold_path))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D bold image, got shape {arr.shape}")
    spatial_shape = (arr.shape[0], arr.shape[1], arr.shape[2])
    n_trs = arr.shape[3]
    y = arr.reshape(-1, n_trs).T
    return y, spatial_shape


def _find_textgrid(stimuli_root: Path, task: str) -> Path | None:
    # Use substring matching because file naming varies between releases.
    candidates = list(stimuli_root.rglob(f"*{task}*.TextGrid"))
    if not candidates:
        candidates = list(stimuli_root.rglob(f"*{task}*.textgrid"))
    if not candidates:
        candidates = list(stimuli_root.rglob("*.TextGrid"))
        if not candidates:
            candidates = list(stimuli_root.rglob("*.textgrid"))
        task_lower = task.lower()
        filtered = [c for c in candidates if task_lower in c.name.lower()]
        candidates = filtered if filtered else candidates
    return candidates[0] if candidates else None


def _extract_between(text: str, left: str, right: str) -> str:
    if left not in text:
        return "unknown"
    tail = text.split(left, 1)[1]
    return tail.split(right, 1)[0] if right in tail else tail
