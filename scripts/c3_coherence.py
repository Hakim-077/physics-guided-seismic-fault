"""
C3 / eigenstructure coherence (Gersztenkorn & Marfurt, 1999).

Classical attribute, not learned -- covariance over a sliding window,
trace by trace. Adapted here for a single 2D line (no crossline), so
the window is (time_window x trace_window) instead of the usual 3D
inline/crossline/time box.

Per sample (i,j):
  C = X^T X over a local window
  coherence = lambda_max(C) / trace(C)
  fault attribute = 1 - coherence  (faults -> low coherence)
"""
import numpy as np


def c3_coherence(section: np.ndarray, time_window: int = 9, trace_window: int = 5) -> np.ndarray:
    """section: (samples, traces). Returns coherence in [0,1], same shape."""
    ns, nt = section.shape
    th = time_window // 2
    tw = trace_window // 2

    padded = np.pad(section, ((th, th), (tw, tw)), mode='reflect')
    coherence = np.ones((ns, nt), dtype=np.float32)

    for i in range(ns):
        for j in range(nt):
            window = padded[i:i + time_window, j:j + trace_window]
            X = window - window.mean(axis=0, keepdims=True)
            C = X.T @ X
            trace_C = np.trace(C)
            if trace_C < 1e-8:
                coherence[i, j] = 1.0
                continue
            eigvals = np.linalg.eigvalsh(C)
            coherence[i, j] = eigvals[-1] / trace_C

    return coherence


def c3_coherence_fast(section: np.ndarray, time_window: int = 9, trace_window: int = 5) -> np.ndarray:
    """Vectorized version using stride tricks instead of the double loop."""
    from numpy.lib.stride_tricks import sliding_window_view

    ns, nt = section.shape
    th = time_window // 2
    tw = trace_window // 2
    padded = np.pad(section, ((th, th), (tw, tw)), mode='reflect').astype(np.float64)

    windows = sliding_window_view(padded, (time_window, trace_window))
    Xmean = windows.mean(axis=2, keepdims=True)
    X = windows - Xmean

    C = np.einsum('ijkl,ijkm->ijlm', X, X)
    trace_C = np.einsum('ijkk->ij', C)

    eigvals = np.linalg.eigvalsh(C)
    lambda_max = eigvals[..., -1]

    coherence = np.where(trace_C > 1e-8, lambda_max / np.maximum(trace_C, 1e-8), 1.0)
    return coherence.astype(np.float32)
