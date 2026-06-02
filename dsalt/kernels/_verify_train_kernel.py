"""GPU correctness check: Triton training kernel vs dense SDPA reference.

Run this ON A GPU (Kaggle T4) before trusting the kernel:

    python -m dsalt.kernels._verify_train_kernel

It builds one DSALTAttention, runs the SAME inputs through (a) the differentiable
Triton sparse kernel and (b) the dense SDPA fallback, and checks that forward
outputs and ALL gradients (q,k,v, win_gate, alpha) match within fp16 tolerance.
"""

import torch


def _run(attn, x, cu, force_triton):
    attn.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    import dsalt.modules.dsalt_attention as M
    saved = M._TRITON_TRAIN_OK
    M._TRITON_TRAIN_OK = force_triton
    try:
        out, _ = attn(x2, cu_seqlens=cu, max_seqlen=int((cu[1] - cu[0]).item()))
        loss = out.float().pow(2).mean()
        loss.backward()
    finally:
        M._TRITON_TRAIN_OK = saved
    g = {
        "out":   out.detach().float(),
        "x":     x2.grad.detach().float().clone(),
        "win":   attn.win_gate.weight.grad.detach().float().clone(),
        "alpha": attn.alpha_w.grad.detach().float().clone(),
    }
    return g


def main():
    assert torch.cuda.is_available(), "Serve una GPU."
    dev = "cuda"
    torch.manual_seed(0)
    from dsalt.modules.dsalt_attention import DSALTAttention, _TRITON_TRAIN_OK
    print("Triton train disponibile:", _TRITON_TRAIN_OK)

    attn = DSALTAttention(d_model=64, n_heads=4, n_min=8, n_max=48, k_lmk=8,
                          max_seq_len=256, layer_idx=0).to(dev)
    attn.train()
    
    # UNA sequenza (N=1): così lo z-score §4.3 per-sequenza del fallback coincide
    # con quello globale del kernel-path, e il confronto isola la matematica
    # dell'attenzione (finestra soft + landmark soft) senza differenze di scoring.
    L = 96
    x = torch.randn(L, 64, device=dev) * 0.5
    cu = torch.tensor([0, L], dtype=torch.int32, device=dev)

    g_tri = _run(attn, x, cu, force_triton=True)
    g_ref = _run(attn, x, cu, force_triton=False)

    def rel(a, b):
        return (a - b).abs().max().item(), (a - b).norm().item() / (b.norm().item() + 1e-9)

    for key in ("out", "x", "win", "alpha"):
        amax, rnorm = rel(g_tri[key], g_ref[key])
        print(f"{key:6s} | max|Δ|={amax:.3e}  relΔ={rnorm:.3e}")

    # tolleranze fp16 generose ma significative
    ok = True
    for key, tol in (("out", 2e-2), ("x", 5e-2), ("win", 8e-2), ("alpha", 8e-2)):
        _, rnorm = rel(g_tri[key], g_ref[key])
        if rnorm > tol:
            ok = False
            print(f"  ✗ {key} oltre tolleranza ({rnorm:.3e} > {tol})")
    print("RISULTATO:", "OK ✓ kernel coerente col riferimento" if ok else "✗ DISCREPANZA")


if __name__ == "__main__":
    main()
