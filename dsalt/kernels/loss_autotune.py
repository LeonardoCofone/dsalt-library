"""Per-device autotuning of the language-model loss.

The LM loss can be computed two ways:

* ``chunked`` — pure PyTorch: materialises ``[chunk, vocab]`` logits in fp32 and
  runs a dense ``cross_entropy`` per chunk. Memory-frugal, no Triton, works
  everywhere — but the fp32 logits materialisation + dense softmax cost real time
  on large vocabularies.
* ``liger``  — fused Triton kernel: never materialises the full logits, fuses
  ``lm_head @ x`` with the cross-entropy. Wins on newer GPUs (A100/H100/B200) with
  fast tensor cores and ample shared memory; *loses* on older cards (e.g. T4 sm_75)
  where the fp32 grad writes + ``@ weight.t()`` GEMM dominate.

Which one wins is a property of the **GPU**, not something to hard-code. So, in the
same spirit as the kernel block-size autotune (:mod:`dsalt.kernels.autotune`), when
``loss_fn="auto"`` we *measure* both strategies — and, for ``chunked``, a sweep of
``chunk_size`` candidates derived from the real token count — once per
``(device, vocab)`` and cache the winner for the whole run.

Nothing here is device-specific: the candidates are generated from the runtime
token count and the choice falls out of the measurement. The day this runs on an
A100/B200 the tuner will pick ``liger`` (or a larger chunk) on its own.
"""

import os

import torch

# Winner per (device_name, vocab_size).
# Value: dict { "loss_fn": "chunked"|"liger", "chunk_size": int|None }.
_LOSS_TUNED: dict = {}


def _key(device: torch.device, vocab: int) -> tuple:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return (name, int(vocab))


def _is_main_process(device: torch.device | None = None) -> bool:
    """True only on the main rank (mirrors :func:`autotune._is_main_process`).

    Each rank tunes its own GPU, but the debug table prints once. We avoid
    importing from ``autotune`` to keep this module's import graph minimal;
    the logic is identical (DDP rank → env rank → CUDA ordinal fallback).
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val) == 0
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return int(idx) == 0


def _chunk_candidates(n_tok: int, vocab: int) -> list[int]:
    """Candidate ``chunk_size`` values for the chunked loss, derived from runtime.

    Fractions of the real token count (full / 2 / 4 / 8 / 16), clamped to a sane
    floor and to ``n_tok``. NOT a fixed constant: a big-VRAM GPU keeps the large
    chunk (fewer kernel launches), a small one falls back to smaller chunks. The
    measurement decides — this only enumerates plausible sizes.
    """
    cands: list[int] = []
    for div in (1, 2, 4, 8, 16):
        c = max(256, (n_tok + div - 1) // div)
        c = min(c, n_tok)
        if c not in cands:
            cands.append(c)
    return cands


def _bench(loss_call, x_leaf, w_leaf, warmup: int = 2, iters: int = 5) -> float:
    """CUDA fwd+bwd time (ms) of a loss strategy on isolated leaves; ``inf`` on fail.

    Both strategies are timed forward **and** backward on detached leaves that
    require grad: this is the fair comparison (Liger computes its gradients inside
    the forward, gated on ``requires_grad``, so a no-grad bench would under-time it;
    chunked's backward is a dense softmax-grad that must count too). The leaves are
    local clones, so nothing touches the live forward graph or its memory.
    """
    try:
        for _ in range(warmup):
            x_leaf.grad = None
            w_leaf.grad = None
            loss_call().backward()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x_leaf.grad = None
            w_leaf.grad = None
            loss_call().backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("inf")
    except Exception:
        return float("inf")


def autotune_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab: int,
    chunked_fn,
    liger_fn,
    liger_ok: bool,
    default_chunk: int,
) -> dict:
    """Measure loss strategies once per ``(device, vocab)`` and cache the winner.

    Args mirror what :class:`DSALTLMHeadModel` already has on hand: ``x`` flat
    ``[n_tok, d_model]`` hidden states, the lm_head ``weight``, flat ``labels``,
    and the two loss callables. Returns ``{"loss_fn", "chunk_size"}``.

    Forward-only timing (no backward) is enough to rank the strategies: the
    backward cost tracks the forward cost for both, and we only need the *order*.
    """
    device = x.device
    key = _key(device, vocab)
    if key in _LOSS_TUNED:
        return _LOSS_TUNED[key]

    # CPU / no-Triton: chunked is the only option (Liger needs Triton).
    if device.type != "cuda" or not liger_ok or liger_fn is None:
        choice = {"loss_fn": "chunked", "chunk_size": default_chunk}
        _LOSS_TUNED[key] = choice
        return choice

    n_tok = x.shape[0]
    results: list[tuple[str, int | None, float]] = []

    # Isolated leaves (clones requiring grad) so the bench's fwd+bwd never touches
    # the live forward graph or its tensors' grads.
    x_leaf = x.detach().clone().requires_grad_(True)
    w_leaf = weight.detach().clone().requires_grad_(True)
    lbl    = labels.detach()

    # chunked: sweep chunk_size candidates derived from the runtime token count.
    for cs in _chunk_candidates(n_tok, vocab):
        ms = _bench(lambda cs=cs: chunked_fn(x_leaf, w_leaf, lbl, cs), x_leaf, w_leaf)
        results.append(("chunked", cs, ms))

    # liger: single fused config (its own internal block sizing is shape-driven).
    ms = _bench(lambda: liger_fn(x_leaf, w_leaf, lbl), x_leaf, w_leaf)
    results.append(("liger", None, ms))

    x_leaf.grad = None
    w_leaf.grad = None

    valid = [r for r in results if r[2] != float("inf")]
    if not valid:
        # Everything OOM'd (shouldn't happen): safe fallback.
        choice = {"loss_fn": "chunked", "chunk_size": default_chunk}
    else:
        best = min(valid, key=lambda r: r[2])
        choice = {"loss_fn": best[0], "chunk_size": best[1]}

    if _is_main_process(device):
        _print_table(device, vocab, results, choice)

    _LOSS_TUNED[key] = choice
    return choice


def _print_table(device, vocab, results, choice) -> None:
    name = torch.cuda.get_device_name(device)
    line = "─" * 64
    print(f"\n{line}")
    print(f"  DSALT loss autotune  |  vocab={vocab}  |  {name}")
    print(line)
    print(f"   {'loss_fn':<10}{'chunk':>10}{'ms/call':>12}")
    print("  " + "-" * 50)
    best_ms = min((m for *_, m in results if m != float("inf")), default=None)
    for fn, cs, ms in sorted(results, key=lambda r: (r[2] if r[2] != float("inf") else 1e18)):
        cs_s = "-" if cs is None else str(cs)
        if ms == float("inf"):
            ms_s = "OOM/fail"
        else:
            ms_s = f"{ms:.4f}"
            if ms == best_ms:
                ms_s += "  ←best"
        print(f"   {fn:<10}{cs_s:>10}{ms_s:>12}")
    print("  " + "-" * 50)
    cs_s = "-" if choice["chunk_size"] is None else str(choice["chunk_size"])
    print(f"  choice: loss_fn={choice['loss_fn']} chunk_size={cs_s}")
    print(line)
