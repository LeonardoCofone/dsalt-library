import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X_ptr + row * stride_row
    Y_ptr = Y_ptr + row * stride_row

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_ptr + cols, y.to(tl.float32), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    X_ptr, W_ptr, DY_ptr, DX_ptr, DW_partial_ptr,
    stride_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X_ptr + row * stride_row
    DY_ptr = DY_ptr + row * stride_row
    DX_ptr = DX_ptr + row * stride_row
    DW_partial_ptr = DW_partial_ptr + row * N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    _dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        _dot += dy * w * x
    dot = tl.sum(_dot)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = x * rstd
        dx = rstd * (dy * w - normed * dot / N)
        tl.store(DX_ptr + cols, dx.to(tl.float32), mask=mask)
        tl.store(DW_partial_ptr + cols, (dy * normed).to(tl.float32), mask=mask)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        M, N = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK_SIZE = min(256, triton.next_power_of_2(N))
        _rmsnorm_fwd_kernel[(M,)](
            x2d, weight, y,
            x2d.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x2d, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight = ctx.saved_tensors
        M, N = x2d.shape
        dy2d = dy.reshape(M, N)
        dx = torch.empty_like(x2d)
        dw_partial = torch.empty((M, N), dtype=torch.float32, device=x2d.device)
        _rmsnorm_bwd_kernel[(M,)](
            x2d, weight, dy2d, dx, dw_partial,
            x2d.stride(0),
            N, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        dw = dw_partial.sum(0)
        return dx.reshape(dy.shape), dw, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x.float(), self.weight, self.eps)