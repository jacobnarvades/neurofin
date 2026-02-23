from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


@dataclass
class PCAScaler:
    """PCA + per-component z-score normalizer fitted on training data."""

    pca: PCA
    pc_mean: np.ndarray   # (n_components,)
    pc_std: np.ndarray    # (n_components,)  — zero-variance PCs clamped to 1

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        z = self.pca.fit_transform(x)
        self.pc_mean = z.mean(axis=0)
        self.pc_std = z.std(axis=0)
        self.pc_std[self.pc_std < 1e-9] = 1.0
        return (z - self.pc_mean) / self.pc_std

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.pca.transform(x)
        return (z - self.pc_mean) / self.pc_std


@dataclass
class EncodingModel:
    scaler: PCAScaler
    weights: np.ndarray
    alpha: float | None = None

    # Keep pca as an alias for backward compat with saved packages that
    # reference it directly.
    @property
    def pca(self) -> PCA:
        return self.scaler.pca

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        return x_scaled @ self.weights


def fit_encoding_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    pca_components: int,
    alpha: float,
) -> EncodingModel:
    scaler = PCAScaler(
        pca=PCA(n_components=pca_components, svd_solver="randomized", random_state=42),
        pc_mean=np.zeros(pca_components, dtype=np.float32),
        pc_std=np.ones(pca_components, dtype=np.float32),
    )
    x_train_scaled = scaler.fit_transform(x_train)
    ridge = Ridge(alpha=float(alpha), fit_intercept=False)
    ridge.fit(x_train_scaled, y_train)
    # sklearn stores coef as (n_targets, n_features)
    weights = ridge.coef_.T.astype(np.float32)
    return EncodingModel(scaler=scaler, weights=weights, alpha=float(alpha))


def select_alpha_with_validation_story(
    x_train_core: np.ndarray,
    y_train_core: np.ndarray,
    x_val_story: np.ndarray,
    y_val_story: np.ndarray,
    pca_components: int,
    alpha_grid: np.ndarray,
) -> tuple[float, dict[float, float]]:
    """
    Select alpha by fitting on train-core and scoring on held-out validation story.
    Scores are mean voxelwise Pearson r on the validation story.
    """
    scaler = PCAScaler(
        pca=PCA(n_components=pca_components, svd_solver="randomized", random_state=42),
        pc_mean=np.zeros(pca_components, dtype=np.float32),
        pc_std=np.ones(pca_components, dtype=np.float32),
    )
    x_train_core_scaled = scaler.fit_transform(x_train_core)
    x_val_story_scaled = scaler.transform(x_val_story)

    scores: dict[float, float] = {}
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf

    for alpha in alpha_grid:
        ridge = Ridge(alpha=float(alpha), fit_intercept=False)
        ridge.fit(x_train_core_scaled, y_train_core)
        y_val_pred = ridge.predict(x_val_story_scaled)
        val_corr = voxelwise_correlation(y_val_pred, y_val_story)
        score = float(np.mean(val_corr))
        scores[float(alpha)] = score
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha, scores


def voxelwise_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true shape mismatch")
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    denom = np.sqrt((y_pred ** 2).sum(axis=0) * (y_true ** 2).sum(axis=0))
    denom[denom == 0] = np.nan
    corr = (y_pred * y_true).sum(axis=0) / denom
    return np.nan_to_num(corr, nan=0.0)
