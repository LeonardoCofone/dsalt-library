"""
Test suite for the DSALT sparse‑attention Triton kernels.                                                                                                                       
"""                      

"""OUTPUT:
Sparse‑attention kernel tests completed.
"""
                                                                                                                                                                                
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from dsalt.kernels import sparse_attn

def _dummy_inputs(
    batch: int = 1,
    heads: int = 2,
    seq_len: int = 32,
    dim: int = 64,
    device: str = "cpu",
):
    """Genera tensori di input fittizi per Q, K, V, finestre e landmark."""
    Q = torch.randn(batch, heads, seq_len, dim, device=device)
    K = torch.randn(batch, heads, seq_len, dim, device=device)
    V = torch.randn(batch, heads, seq_len, dim, device=device)
    # finestra locale: tutti i token hanno una finestra di 4 posizioni
    win = torch.full((batch, heads, seq_len), 4, dtype=torch.int32, device=device)
    # landmark: scegliamo i primi 2 token come landmark per tutti
    lmk = torch.arange(2, device=device).repeat(batch, heads, 1)
    return Q, K, V, win, lmk


def _run_attention(Q, K, V, win, lmk, device: str = "cpu"):
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA is required for Triton path"
        Q = Q.cuda(); K = K.cuda(); V = V.cuda(); win = win.cuda(); lmk = lmk.cuda()
    out = sparse_attn.dsalt_attention(Q, K, V, win, lmk)
    return out.cpu() if device == "cuda" else out


def test_forward_cpu():
    Q, K, V, win, lmk = _dummy_inputs(device="cpu")
    cpu_out = _run_attention(Q, K, V, win, lmk, device="cpu")

    assert cpu_out.shape == Q.shape
    assert not torch.isnan(cpu_out).any()

    if sparse_attn._TRITON_AVAILABLE and torch.cuda.is_available():
        triton_out = _run_attention(Q, K, V, win, lmk, device="cuda")
        assert torch.allclose(cpu_out, triton_out, atol=1e-3, rtol=1e-3)
        assert not torch.isnan(triton_out).any()


def test_forward_triton_available():
    """Esegue il kernel solo se Triton è installato."""
    if not (sparse_attn._TRITON_AVAILABLE and torch.cuda.is_available()):
        return  # test saltato

    Q, K, V, win, lmk = _dummy_inputs(device="cuda")
    out = sparse_attn.dsalt_attention(Q, K, V, win, lmk)

    assert out.shape == Q.shape
    assert not torch.isnan(out).any()


def test_backward_consistency():
    Q, K, V, win, lmk = _dummy_inputs(device="cpu")
    Q.requires_grad_()
    K.requires_grad_()
    V.requires_grad_()

    cpu_out = sparse_attn.dsalt_attention(Q, K, V, win, lmk)
    cpu_loss = cpu_out.sum()
    cpu_loss.backward()

    assert Q.grad is not None and K.grad is not None and V.grad is not None
    cpu_grads = [Q.grad.clone(), K.grad.clone(), V.grad.clone()]

    if sparse_attn._TRITON_AVAILABLE and torch.cuda.is_available():
        Qc = Q.detach().clone().cuda().requires_grad_()
        Kc = K.detach().clone().cuda().requires_grad_()
        Vc = V.detach().clone().cuda().requires_grad_()
        out_cuda = sparse_attn.dsalt_attention(Qc, Kc, Vc, win.cuda(), lmk.cuda())
        cuda_loss = out_cuda.sum()
        cuda_loss.backward()

        cuda_grads = [Qc.grad.cpu(), Kc.grad.cpu(), Vc.grad.cpu()]
        for name, cpu_grad, cuda_grad in zip(["Q", "K", "V"], cpu_grads, cuda_grads):
            assert torch.allclose(cpu_grad, cuda_grad, atol=1e-3, rtol=1e-3), \
                f"Gradient mismatch for {name}"


if __name__ == "__main__":
    test_forward_cpu()
    test_forward_triton_available()
    test_backward_consistency()
    print("Sparse-attention kernel tests completed.")
