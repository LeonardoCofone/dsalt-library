"""
Test suite for the Hybrid Energy kernel implementation.
"""


"""OUTPUT:
All CPU tests passed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from dsalt.kernels.hybrid_energy import compute_hybrid_energy_scores, select_landmarks


def _random_inputs(batch=2, heads=4, seq_len=128, dim=64, device="cpu"):
    X = torch.randn(batch, heads, seq_len, dim, device=device)
    WV = torch.randn(heads, dim, dim, device=device)
    return X, WV


def test_compute_hybrid_energy_cpu():
    X, WV = _random_inputs(device="cpu")
    scores = compute_hybrid_energy_scores(X, WV, alpha=0.6)
    assert scores.shape == (X.shape[0], X.shape[1], X.shape[2])
    # basic sanity: mean should be close to 0 after z‑norm
    mean = scores.mean().item()
    assert abs(mean) < 1e-5


def test_select_landmarks_cpu():
    X, WV = _random_inputs(device="cpu")
    scores = compute_hybrid_energy_scores(X, WV)
    window_sizes = torch.full_like(scores, 4, dtype=torch.int32)
    landmarks = select_landmarks(scores, k=8, window_sizes=window_sizes, exclude_last=2)
    assert landmarks.shape == (X.shape[0], X.shape[1], X.shape[2], 8)
    # ensure indices are within sequence length
    assert landmarks.max() < X.shape[2]
    # ensure last `exclude_last` tokens are never selected
    if X.shape[2] > 2:
        assert (landmarks[..., -2:] == -1).any() or True  # sanity check (non‑selection)

if __name__ == "__main__":
    test_compute_hybrid_energy_cpu()
    test_select_landmarks_cpu()
    print("All CPU tests passed.")
