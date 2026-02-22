from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from nilearn.glm.first_level.hemodynamic_models import spm_hrf


def align_features_to_tr(
    word_features: np.ndarray,
    word_times: np.ndarray,
    tr: float,
    n_trs: int,
    delays: list[int],
    interpolation_method: str = "lanczos",
    lanczos_window: int = 3,
) -> np.ndarray:
    """
    Convert (n_words, n_layers, hidden_dim) to delayed TR-space design matrix:
    output shape (n_trs, n_delays * n_layers * hidden_dim)
    """
    if word_features.shape[0] == 0:
        raise ValueError("No word features provided.")
    if word_features.shape[0] != word_times.shape[0]:
        raise ValueError("word_features and word_times length mismatch.")

    tr_times = np.arange(n_trs, dtype=np.float64) * tr
    interpolated = _lanczos_or_interp(
        word_times,
        word_features,
        tr_times,
        interpolation_method=interpolation_method,
        lanczos_window=lanczos_window,
    )
    hrf_convolved = _convolve_with_spm_hrf(interpolated, tr)
    delayed = _apply_fir_delays(hrf_convolved, delays)
    return delayed


def _lanczos_or_interp(
    src_times: np.ndarray,
    src_features: np.ndarray,
    target_times: np.ndarray,
    interpolation_method: str,
    lanczos_window: int,
) -> np.ndarray:
    """
    Use Lanczos interpolation by default. If optional HuthLab helper is unavailable,
    fall back to local Lanczos implementation (not linear).
    """
    method = interpolation_method.lower()
    if method == "lanczos":
        try:
            # Optional dependency from encoding-model-scaling-laws.
            from ridge_utils.interpdata import lanczosinterp2D  # type: ignore

            # Flatten feature dims for interpolation utility.
            two_d = src_features.reshape(src_features.shape[0], -1)
            out = lanczosinterp2D(two_d, src_times, target_times, window=lanczos_window)
            return out.reshape(target_times.shape[0], src_features.shape[1], src_features.shape[2]).astype(np.float32)
        except ImportError as exc:
            warnings.warn(
                f"ridge_utils Lanczos unavailable ({exc!r}); using local Lanczos implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            two_d = src_features.reshape(src_features.shape[0], -1)
            out = _lanczos_interp_2d_local(two_d, src_times, target_times, lanczos_window)
            return out.reshape(target_times.shape[0], src_features.shape[1], src_features.shape[2]).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Lanczos interpolation failed: {exc!r}") from exc
    if method == "linear" or method == "lanczos":
        flat = src_features.reshape(src_features.shape[0], -1)
        fn = interp1d(
            src_times,
            flat,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        out = fn(target_times)
        return out.reshape(target_times.shape[0], src_features.shape[1], src_features.shape[2]).astype(np.float32)
    raise ValueError(f"Unsupported interpolation_method: {interpolation_method}")


def _lanczos_interp_2d_local(
    src_values: np.ndarray,
    src_times: np.ndarray,
    target_times: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Local Lanczos interpolation for (n_samples, n_features) arrays.
    """
    if window <= 0:
        raise ValueError("lanczos_window must be > 0")
    if src_values.ndim != 2:
        raise ValueError("src_values must be 2D")
    if src_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("src_times and target_times must be 1D")
    if src_values.shape[0] != src_times.shape[0]:
        raise ValueError("src_values and src_times length mismatch")

    # Ensure monotonic order.
    order = np.argsort(src_times)
    t_src = src_times[order].astype(np.float64)
    y_src = src_values[order].astype(np.float64)

    if t_src.shape[0] < 2:
        return np.repeat(y_src[:1], target_times.shape[0], axis=0).astype(np.float32)

    dt = float(np.median(np.diff(t_src)))
    if dt <= 0:
        raise ValueError("Non-positive source time spacing for Lanczos interpolation.")

    out = np.zeros((target_times.shape[0], y_src.shape[1]), dtype=np.float64)
    for i, t in enumerate(target_times.astype(np.float64)):
        u = (t - t_src) / dt
        in_win = np.abs(u) < window
        if not np.any(in_win):
            nearest = int(np.argmin(np.abs(t_src - t)))
            out[i] = y_src[nearest]
            continue
        u_w = u[in_win]
        # np.sinc(x) = sin(pi x)/(pi x)
        w = np.sinc(u_w) * np.sinc(u_w / float(window))
        w_sum = float(np.sum(w))
        if np.isclose(w_sum, 0.0):
            nearest = int(np.argmin(np.abs(t_src - t)))
            out[i] = y_src[nearest]
            continue
        w = w / w_sum
        out[i] = w @ y_src[in_win]
    return out.astype(np.float32)


def _apply_fir_delays(features_tr: np.ndarray, delays: list[int]) -> np.ndarray:
    n_trs, n_layers, hidden = features_tr.shape
    delayed = []
    for d in delays:
        shifted = np.zeros_like(features_tr)
        if d < n_trs:
            shifted[d:] = features_tr[:-d]
        delayed.append(shifted.reshape(n_trs, n_layers * hidden))
    return np.concatenate(delayed, axis=1).astype(np.float32)


def _convolve_with_spm_hrf(features_tr: np.ndarray, tr: float) -> np.ndarray:
    """
    Apply canonical double-gamma SPM HRF to each feature channel.
    """
    n_trs = features_tr.shape[0]
    hrf = spm_hrf(tr).astype(np.float32)
    flat = features_tr.reshape(n_trs, -1).astype(np.float32)
    out = np.zeros_like(flat, dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = fftconvolve(flat[:, i], hrf, mode="full")[:n_trs]
    return out.reshape(features_tr.shape)
