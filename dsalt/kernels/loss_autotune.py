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

    Fractions of the real token count (1/16 … full), clamped to a sane floor and
    to ``n_tok``. NOT a fixed constant: a big-VRAM GPU keeps the large chunk
    (fewer kernel launches), a small one falls back to smaller chunks. The
    measurement decides — this only enumerates plausible sizes.

    Returned in **ascending** order so the caller can sweep small→large and stop
    at the first OOM: a larger chunk allocates strictly more (its ``[chunk,
    vocab]`` fp32 logits grow), so once one OOMs every larger one would too.
    """
    cands: list[int] = []
    for div in (16, 8, 4, 2, 1):
        c = max(256, (n_tok + div - 1) // div)
        c = min(c, n_tok)
        if c not in cands:
            cands.append(c)
    return cands


def _bench(loss_call, warmup: int = 2, iters: int = 5) -> float:
    """CUDA forward-only time (ms) of a loss strategy; ``inf`` on OOM/fail.

    Timed **forward-only** under ``torch.no_grad()`` on detached *views* of the
    live tensors (no clones, no extra grad buffers): this is the whole point of
    the design choice — the autotune fires inside the first forward, while the
    training graph is still alive and VRAM is already tight, so allocating clones
    of the 50k×512 embedding + their grads is exactly what blew up the T4. A
    forward-only measure on views adds ~zero memory.

    Forward-only is enough to *rank* the strategies: chunked's cost is dominated
    by the fp32 logits materialisation + dense softmax (a forward effect), and
    Liger's whole advantage is *not* materialising those logits — visible in the
    forward already. The backward tracks the forward for both, so the order is
    preserved on every GPU (chunked wins on T4, liger wins on A100+).

    ``empty_cache()`` is called on OOM so a failed (too-large) candidate does not
    leave fragmentation behind for the next candidate or for the real training.
    """
    try:
        with torch.no_grad():
            for _ in range(warmup):
                loss_call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                loss_call()
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
    main = _is_main_process(device)

    # Detached VIEWS of the live tensors — NOT clones. The autotune runs inside
    # the first forward, with the training graph alive and VRAM already tight;
    # cloning the 50k×512 embedding + grads is what OOM'd the T4. Forward-only
    # under no_grad on these views never writes a grad into the real parameter
    # and allocates ~nothing beyond each candidate's own logits.
    x_d = x.detach()
    w_d = weight.detach()
    lbl = labels.detach()

    # Progress log (only on the main rank), in the same spirit as the kernel
    # block-size autotune: one line per benchmarked candidate so the user sees
    # the tuner working instead of a silent stall on the first forward.
    chunk_cands = _chunk_candidates(n_tok, vocab)
    n_cands = len(chunk_cands) + 1  # + liger
    if main:
        name = torch.cuda.get_device_name(device)
        print(f"\n  DSALT loss autotune | vocab={vocab} | n_tok={n_tok} | {name}")
        print(f"  measuring {n_cands} candidate(s) (forward-only, no clones)…")

    # chunked: sweep chunk_size candidates small→large; stop at the first OOM
    # (a larger chunk allocates strictly more, so it would OOM too).
    for i, cs in enumerate(chunk_cands, start=1):
        ms = _bench(lambda cs=cs: chunked_fn(x_d, w_d, lbl, cs))
        results.append(("chunked", cs, ms))
        if main:
            ms_s = "OOM/fail" if ms == float("inf") else f"{ms:8.4f} ms"
            print(f"    [{i}/{n_cands}] chunked  chunk={cs:<8} {ms_s}")
        if ms == float("inf"):
            # Skip the remaining (larger) chunked candidates — all would OOM.
            for cs_skip in chunk_cands[i:]:
                results.append(("chunked", cs_skip, float("inf")))
            break

    # liger: single fused config (its own internal block sizing is shape-driven).
    ms = _bench(lambda: liger_fn(x_d, w_d, lbl))
    results.append(("liger", None, ms))
    if main:
        ms_s = "OOM/fail" if ms == float("inf") else f"{ms:8.4f} ms"
        print(f"    [{n_cands}/{n_cands}] liger    chunk={'-':<8} {ms_s}")

    # Release any scratch the candidates left so the real training step starts
    # from a clean allocator (no fragmentation carried over from the sweep).
    torch.cuda.empty_cache()

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
    major, minor = torch.cuda.get_device_capability(device)
    print("\n" + "─" * 68)
    print(f"  DSALT loss autotune  |  vocab={vocab}  |  {name} (sm_{major}{minor})")
    print("─" * 68)
    print(f"  {'loss_fn':>10} {'chunk':>10} {'ms/step':>12}")
    print("  " + "-" * 64)
    best_ms = min((m for *_, m in results if m != float("inf")), default=None)

    def _sort_key(r):
        ms = r[2]
        return (ms == float("inf"), ms if ms != float("inf") else 0.0)

    for fn, cs, ms in sorted(results, key=_sort_key):
        cs_s = "-" if cs is None else str(cs)
        if ms == float("inf"):
            ms_s = f"{'OOM/fail':>12}"
            mark = ""
        else:
            ms_s = f"{ms:12.4f}"
            mark = "  ←best" if ms == best_ms else ""
        print(f"  {fn:>10} {cs_s:>10} {ms_s}{mark}")

    print("  " + "-" * 64)
    cs_s = "-" if choice["chunk_size"] is None else str(choice["chunk_size"])
    best_str = f" ({best_ms:.4f} ms/step)" if best_ms is not None else ""
    print(f"  choice: loss_fn={choice['loss_fn']} chunk_size={cs_s}{best_str}")
    print("─" * 68 + "\n")
