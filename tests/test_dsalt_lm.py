"""
OUTPUT:
✓ test_dsalt_lm_forward passed
✓ test_dsalt_lm_loss_and_windows passed
DSALT LM wrapper tests completed successfully.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from dsalt.model import DSALTLMHeadModel


def test_dsalt_lm_forward():
    model = DSALTLMHeadModel(
        vocab_size=64,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_min=16,
        n_max=32,
        k_lmk=4,
    )
    input_ids = torch.randint(0, 64, (1, 8))
    out = model(input_ids)
    assert "logits" in out
    assert out["logits"].shape == (1, 8, 64)


def test_dsalt_lm_loss_and_windows():
    model = DSALTLMHeadModel(
        vocab_size=32,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_min=16,
        n_max=32,
        k_lmk=4,
    )
    input_ids = torch.randint(0, 32, (1, 10))
    labels = torch.randint(0, 32, (1, 10))
    out = model(input_ids, labels=labels, return_window=True)
    assert "loss" in out
    assert out["loss"].item() >= 0
    assert "windows" in out
    assert out["windows"] is not None
    assert isinstance(out["windows"], list)
    assert len(out["windows"]) == 1
    assert out["windows"][0].shape == (1, 10)


if __name__ == "__main__":
    test_dsalt_lm_forward()
    print("✓ test_dsalt_lm_forward passed")
    test_dsalt_lm_loss_and_windows()
    print("✓ test_dsalt_lm_loss_and_windows passed")
    print("DSALT LM wrapper tests completed successfully.")
