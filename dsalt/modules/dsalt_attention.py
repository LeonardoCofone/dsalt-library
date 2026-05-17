import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    build_rope_cache,
    apply_rotary_emb,
)
from ..kernels.sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
)

try:
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
    print(f"--- [dsalt_attention] Triton disponibile: _TRITON_OK=True")
except Exception as e:
    _TRITON_OK = False
    print(f"--- [dsalt_attention] Triton NON disponibile: {e}")


@torch.no_grad()
def _hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
    dh:    int,
) -> torch.Tensor:
    t0 = time.perf_counter()
    T, d = x.shape
    n_heads = alpha.shape[0]
    print(f"--- [dsalt_attention] _hybrid_scores START | T={T} d={d} n_heads={n_heads} dh={dh} | device={x.device}")

    x_norm = x.norm(dim=-1)
    mu_x   = x_norm.mean()
    std_x  = x_norm.std().clamp(min=1e-6)
    z_x    = (x_norm - mu_x) / std_x
    print(f"--- [dsalt_attention] z_x calcolato | mu_x={mu_x.item():.4f} std_x={std_x.item():.4f}")

    W_V_h = W_V.view(n_heads, dh, -1)
    print(f"--- [dsalt_attention] W_V_h={tuple(W_V_h.shape)} - lancio einsum td,hde->the")
    t1  = time.perf_counter()
    xwv = torch.einsum("td,hde->the", x, W_V_h).norm(dim=-1)
    print(f"--- [dsalt_attention] einsum DONE | xwv={tuple(xwv.shape)} | t_einsum={time.perf_counter()-t1:.4f}s")

    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v

    scores = (alpha.unsqueeze(0) * z_v + (1.0 - alpha.unsqueeze(0)) * z_x.unsqueeze(1)).mean(dim=1)
    print(f"--- [dsalt_attention] _hybrid_scores DONE | scores={tuple(scores.shape)} mean={scores.mean().item():.4f} | t={time.perf_counter()-t0:.4f}s")
    return scores


@torch.no_grad()
def _build_mask_batched(
    x:           torch.Tensor,
    window_proj: nn.Linear,
    v_weight:    torch.Tensor,
    alpha:       torch.Tensor,
    n_min:       int,
    n_max:       int,
    k_lmk:      int,
    T:           int,
    device:      torch.device,
) -> torch.Tensor:
    t0 = time.perf_counter()
    print(f"--- [dsalt_attention] _build_mask_batched START | T={T} n_min={n_min} n_max={n_max} k_lmk={k_lmk} | device={device}")

    w_sizes     = compute_window_sizes(x, window_proj, n_min, n_max)
    window_mask = build_local_window_mask(T, w_sizes, device)

    scores    = _hybrid_scores(x, v_weight, alpha, v_weight.shape[0] // alpha.shape[0])
    in_window = window_mask.any(dim=0)
    n_in_win  = in_window.sum().item()
    print(f"--- [dsalt_attention] _build_mask_batched | tokens in window={n_in_win}/{T}")

    scores_filtered = scores.masked_fill(in_window, float("-inf"))
    k_act = min(k_lmk, int((scores_filtered > float("-inf")).sum()))
    print(f"--- [dsalt_attention] _build_mask_batched | k_act={k_act} (richiesto k={k_lmk})")

    if k_act > 0:
        _, idx = torch.topk(scores_filtered, k_act, sorted=False)
        window_mask[:, idx] = True
        print(f"--- [dsalt_attention] _build_mask_batched | landmark idx aggiunti alla mask | idx[:5]={idx[:5].tolist()}")
    else:
        print(f"--- [dsalt_attention] WARNING: nessun landmark aggiunto (k_act=0)")

    nonzero_frac = window_mask.float().mean().item()
    print(f"--- [dsalt_attention] _build_mask_batched DONE | nonzero_frac={nonzero_frac:.4f} | t={time.perf_counter()-t0:.4f}s")
    return window_mask


@torch.no_grad()
def _build_mask_packed(
    x:           torch.Tensor,
    w_sizes:     torch.Tensor,
    window_proj: nn.Linear,
    v_weight:    torch.Tensor,
    alpha:       torch.Tensor,
    cu_seqlens:  torch.Tensor,
    total_len:   int,
    k_lmk:      int,
    dh:          int,
    device:      torch.device,
) -> torch.Tensor:
    t0 = time.perf_counter()
    num_seqs = cu_seqlens.shape[0] - 1
    print(f"--- [dsalt_attention] _build_mask_packed START | total_len={total_len} num_seqs={num_seqs} k_lmk={k_lmk} | device={device}")

    lens    = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts  = cu_seqlens[:-1].to(device)
    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total_len, device=device) - starts[seq_ids]
    max_len = int(lens.max())
    print(f"--- [dsalt_attention] _build_mask_packed | max_len={max_len} | lens={lens.tolist()}")

    t1   = time.perf_counter()
    mask = build_local_window_mask_packed(cu_seqlens, w_sizes, total_len, device)
    print(f"--- [dsalt_attention] window mask packed costruita | t={time.perf_counter()-t1:.4f}s")

    scores = _hybrid_scores(x, v_weight, alpha, dh)
    print(f"--- [dsalt_attention] hybrid scores calcolati | t={time.perf_counter()-t0:.4f}s")

    in_win   = mask.any(dim=0)
    s_masked = scores.masked_fill(in_win, float("-inf"))
    print(f"--- [dsalt_attention] tokens in window={in_win.sum().item()}/{total_len}")

    score_pad = torch.full((num_seqs, max_len), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = s_masked
    print(f"--- [dsalt_attention] score_pad costruito | shape={tuple(score_pad.shape)}")

    k_eff     = min(k_lmk, max_len)
    vals, top = torch.topk(score_pad, k_eff, dim=1, sorted=False)
    valid     = vals > float("-inf")
    print(f"--- [dsalt_attention] topk DONE | k_eff={k_eff} | valid_per_seq={valid.sum(dim=1).tolist()}")

    abs_lmk = (starts.unsqueeze(1) + top).clamp(max=total_len - 1)

    for b in range(num_seqs):
        if not valid[b].any():
            print(f"--- [dsalt_attention] seq {b}: nessun landmark valido, skip")
            continue
        s   = int(cu_seqlens[b])
        e   = int(cu_seqlens[b + 1])
        idx = abs_lmk[b][valid[b]]
        print(f"--- [dsalt_attention] seq {b}: landmark idx aggiunti | s={s} e={e} n_lmk={len(idx)}")
        mask[s:e, idx] = True

    nz = mask.float().mean().item()
    print(f"--- [dsalt_attention] _build_mask_packed DONE | nonzero_frac={nz:.6f} | t_total={time.perf_counter()-t0:.4f}s")
    return mask


class DSALTAttention(nn.Module):
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_min:       int,
        n_max:       int,
        k_lmk:       int,
        max_seq_len: int,
        dropout:     float = 0.0,
        yarn_scale:  float = 1.0,
        layer_idx:   int   = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} non divisibile per n_heads={n_heads}"

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.n_min       = n_min
        self.n_max       = n_max
        self.k_lmk       = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout     = dropout
        self.yarn_scale  = yarn_scale
        self.layer_idx   = layer_idx

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)

        self.alpha_w = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        self._last_P: torch.Tensor | None = None

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), scale=yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        print(f"--- [DSALTAttention] init | layer={layer_idx} d_model={d_model} n_heads={n_heads} head_dim={self.head_dim} n_min={n_min} n_max={n_max} k_lmk={k_lmk} max_seq={max_seq_len}")

    def _alpha(self) -> torch.Tensor:
        a = torch.sigmoid(self.alpha_w)
        print(f"--- [DSALTAttention] layer={self.layer_idx} alpha values: min={a.min().item():.4f} max={a.max().item():.4f} mean={a.mean().item():.4f}")
        return a

    @torch.no_grad()
    def _rope(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter()
        if n <= self.rope_cos.shape[0]:
            cos = self.rope_cos[:n].to(device)
            sin = self.rope_sin[:n].to(device)
        else:
            print(f"--- [DSALTAttention] layer={self.layer_idx} rope cache miss: n={n} > max={self.rope_cos.shape[0]}, ricalcolo")
            cos, sin = build_rope_cache(n, self.head_dim, device, scale=self.yarn_scale)
        print(f"--- [DSALTAttention] layer={self.layer_idx} _rope | n={n} | t={time.perf_counter()-t0:.4f}s")
        return cos, sin

    def forward(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None          = None,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [DSALTAttention] layer={self.layer_idx} forward | x={tuple(x.shape)} | packed={cu_seqlens is not None} | training={self.training}")

        if cu_seqlens is not None:
            out = self._packed(x, cu_seqlens, x.shape[0], x.device)
        else:
            B, T, _ = x.shape
            out = self._batched(x, B, T, x.device)

        print(f"--- [DSALTAttention] layer={self.layer_idx} forward DONE | out={tuple(out.shape)} | t={time.perf_counter()-t0:.4f}s")
        return out

    def _batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [DSALTAttention] layer={self.layer_idx} _batched | B={B} T={T} device={device}")

        t1 = time.perf_counter()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        print(f"--- [DSALTAttention] layer={self.layer_idx} QKV proj | q={tuple(q.shape)} | t={time.perf_counter()-t1:.4f}s")

        cos, sin = self._rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_1b = x[0] if B > 1 else x.squeeze(0)
        print(f"--- [DSALTAttention] layer={self.layer_idx} uso x[0] per mask (B={B}) — x_1b={tuple(x_1b.shape)}")

        t2 = time.perf_counter()
        attn_mask = _build_mask_batched(
            x_1b, self.window_proj, self.v_proj.weight.detach(),
            self._alpha().detach(), self.n_min, self.n_max, self.k_lmk, T, device,
        )
        print(f"--- [DSALTAttention] layer={self.layer_idx} mask costruita | t={time.perf_counter()-t2:.4f}s")

        t3 = time.perf_counter()
        out = sparse_attention_forward(q, k, v, attn_mask, self.dropout, self.training)
        print(f"--- [DSALTAttention] layer={self.layer_idx} sparse_attention DONE | t={time.perf_counter()-t3:.4f}s")

        if not self.training:
            scale    = math.sqrt(self.head_dim)
            sc       = torch.matmul(q[0], k[0].transpose(-2, -1)) / scale
            additive = sc.new_zeros(sc.shape).masked_fill_(~attn_mask.unsqueeze(0).expand_as(sc), float("-inf"))
            self._last_P = torch.softmax(sc + additive, dim=-1).detach()
            print(f"--- [DSALTAttention] layer={self.layer_idx} _last_P salvato | shape={tuple(self._last_P.shape)}")
        else:
            self._last_P = None

        result = self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model))
        print(f"--- [DSALTAttention] layer={self.layer_idx} _batched DONE | result={tuple(result.shape)} | t_total={time.perf_counter()-t0:.4f}s")
        return result

    def _packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [DSALTAttention] layer={self.layer_idx} _packed | total_len={total_len} | n_seqs={cu_seqlens.shape[0]-1} | device={device}")

        t1 = time.perf_counter()
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)
        print(f"--- [DSALTAttention] layer={self.layer_idx} QKV proj packed | q={tuple(q.shape)} | t={time.perf_counter()-t1:.4f}s")

        cos, sin = self._rope(total_len, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        w_sizes = compute_window_sizes(x, self.window_proj, self.n_min, self.n_max).detach()
        print(f"--- [DSALTAttention] layer={self.layer_idx} w_sizes computed | mean={w_sizes.mean().item():.1f}")

        if _TRITON_OK and self.training and device.type == "cuda":
            print(f"--- [DSALTAttention] layer={self.layer_idx} usando kernel Triton")
            t2 = time.perf_counter()
            out = dsalt_triton_attention(
                q, k, v, x,
                self.v_proj.weight.detach(),
                self._alpha().detach(),
                w_sizes,
                cu_seqlens,
                self.k_lmk,
            )
            print(f"--- [DSALTAttention] layer={self.layer_idx} Triton DONE | t={time.perf_counter()-t2:.4f}s")
            self._last_P = None
            result = self.out_proj(out.contiguous().view(total_len, self.d_model))
            print(f"--- [DSALTAttention] layer={self.layer_idx} _packed (Triton) DONE | t_total={time.perf_counter()-t0:.4f}s")
            return result

        print(f"--- [DSALTAttention] layer={self.layer_idx} fallback a sparse_attention_forward_packed (no Triton)")
        t2 = time.perf_counter()
        attn_mask = _build_mask_packed(
            x, w_sizes, self.window_proj, self.v_proj.weight.detach(),
            self._alpha().detach(), cu_seqlens, total_len,
            self.k_lmk, self.head_dim, device,
        )
        print(f"--- [DSALTAttention] layer={self.layer_idx} mask packed costruita | t={time.perf_counter()-t2:.4f}s")

        t3 = time.perf_counter()
        out = sparse_attention_forward_packed(q, k, v, attn_mask, self.dropout, self.training)
        print(f"--- [DSALTAttention] layer={self.layer_idx} sparse_attention_packed DONE | t={time.perf_counter()-t3:.4f}s")

        if not self.training:
            scale = math.sqrt(self.head_dim)
            with torch.no_grad():
                s  = int(cu_seqlens[0]); e = int(cu_seqlens[1])
                sc = torch.matmul(q[s:e].transpose(0, 1), k[s:e].transpose(0, 1).transpose(-2, -1)) / scale
                ad = sc.new_zeros(sc.shape).masked_fill_(~attn_mask[s:e, s:e].unsqueeze(0).expand_as(sc), float("-inf"))
                self._last_P = torch.softmax(sc + ad, dim=-1).detach()
                print(f"--- [DSALTAttention] layer={self.layer_idx} _last_P salvato (packed) | shape={tuple(self._last_P.shape)}")
        else:
            self._last_P = None

        result = self.out_proj(out.contiguous().view(total_len, self.d_model))
        print(f"--- [DSALTAttention] layer={self.layer_idx} _packed DONE | result={tuple(result.shape)} | t_total={time.perf_counter()-t0:.4f}s")
        return result