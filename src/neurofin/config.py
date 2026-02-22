from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    project_root: Path = Path(".")
    data_root: Path = Path("ds003020")
    output_dir: Path = Path("artifacts")

    model_name: str = "Qwen/Qwen3-14B-Base"
    layer_indices: list[int] = field(default_factory=lambda: list(range(20, 29)))
    hidden_dim: int = 5120
    use_4bit: bool = False
    device: str = "cuda"

    delays: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    target_tr: float = 1.5
    sentence_rate_hz: float = 0.5
    interpolation_method: str = "lanczos"
    hrf_model: str = "spm"
    lanczos_window: int = 3

    pca_components: int = 500
    ridge_alphas_log10_min: int = 1
    ridge_alphas_log10_max: int = 8
    ridge_alpha_count: int = 16

    voxel_corr_threshold: float = 0.05
    random_state: int = 42

    # Optimized for 64GB RAM host.
    story_batch_size: int = 1
    max_tokens: int = 4096
    context_tokens: int = 1024

    @property
    def derivatives_root(self) -> Path:
        plural = self.data_root / "derivatives"
        singular = self.data_root / "derivative"
        plural_has_data = any(plural.rglob("*_bold.nii.gz")) or any(plural.rglob("preprocessed_data/*/*.hf5"))
        singular_has_data = any(singular.rglob("*_bold.nii.gz")) or any(singular.rglob("preprocessed_data/*/*.hf5"))
        if plural_has_data:
            return plural
        if singular_has_data:
            return singular
        if plural.exists():
            return plural
        if singular.exists():
            return singular
        return plural

    @property
    def stimuli_root(self) -> Path:
        primary = self.data_root / "stimuli"
        derivative_textgrids = self.data_root / "derivative" / "TextGrids"
        primary_has_textgrids = any(primary.rglob("*.TextGrid")) or any(primary.rglob("*.textgrid"))
        derivative_has_textgrids = any(derivative_textgrids.rglob("*.TextGrid")) or any(derivative_textgrids.rglob("*.textgrid"))
        if primary_has_textgrids:
            return primary
        if derivative_has_textgrids:
            return derivative_textgrids
        if primary.exists():
            return primary
        if derivative_textgrids.exists():
            return derivative_textgrids
        return primary
