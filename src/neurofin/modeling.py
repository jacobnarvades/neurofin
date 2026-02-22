from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


@dataclass
class EncodingModel:
    pca: PCA
    weights: np.ndarray
    alpha: float | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_pca = self.pca.transform(x)
        return x_pca @ self.weights


def fit_encoding_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    pca_components: int,
    alpha: float,
) -> EncodingModel:
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    ridge = Ridge(alpha=float(alpha), fit_intercept=False)
    ridge.fit(x_train_pca, y_train)
    # sklearn stores coef as (n_targets, n_features)
    weights = ridge.coef_.T.astype(np.float32)
    return EncodingModel(pca=pca, weights=weights, alpha=float(alpha))


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
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
    x_train_core_pca = pca.fit_transform(x_train_core)
    x_val_story_pca = pca.transform(x_val_story)

    scores: dict[float, float] = {}
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf

    for alpha in alpha_grid:
        ridge = Ridge(alpha=float(alpha), fit_intercept=False)
        ridge.fit(x_train_core_pca, y_train_core)
        y_val_pred = ridge.predict(x_val_story_pca)
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
