import torch
from torch.profiler import profile, ProfilerActivity
import time

from dsalt import DSALTConfig, DSALTLMHeadModel
from data_utils import make_loaders
from utils_dsalt import CFG  # noqa: F401

VOCAB_SIZE = 50257
device = "cuda:0"

train_loader, _, cfg = make_loaders(limit=200)
model_cfg = DSALTConfig(
    vocab_size=VOCAB_SIZE,
    d_model=cfg["d_model"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    n_min=cfg["N_min"],
    n_max=cfg["N_max"],
    k_lmk=cfg["k_lm"],
    d_ff=cfg["d_ff"],
    max_seq_len=cfg["seq_len"],
    dropout=cfg["dropout"],
    loss_fn=cfg["loss_type"],
    lm_head_chunk_size=cfg["head_chunk_size"],
)
model = DSALTLMHeadModel.from_config(model_cfg).to(device)
model.train()

ids, labels, cu_seqlens, max_seqlen = next(iter(train_loader))
ids = ids.to(device)
labels = labels.to(device)
cu_seqlens = cu_seqlens.to(device)
max_seqlen = int(max_seqlen)


def step():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, labels=labels)["loss"]
    loss.backward()
    model.zero_grad(set_to_none=True)


for _ in range(5):
    step()
torch.cuda.synchronize()

N = 20
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
print(f"\n>>> {1.0/dt:.3f} it/s  ({dt*1000:.1f} ms/step)  "
      f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
             record_shapes=False) as prof:
    step()
    torch.cuda.synchronize()

ka = prof.key_averages()
table_str = ka.table(sort_by="self_cuda_time_total", row_limit=150)

print("=" * 100)
print("TOP 25 per CUDA self-time")
print("=" * 100)
print("\n".join(table_str.split("\n")[:30]))

buckets = {
    "triton (attention/RoPE)": 0.0,
    "gemm/mm (proiezioni+ffn)": 0.0,
    "cast/copy (autocast fp16)": 0.0,
    "softmax/reduce": 0.0,
    "elementwise": 0.0,
    "rms_norm": 0.0,
    "altro": 0.0,
}

def parse_time_to_ms(t_str):
    if t_str.endswith("us"):
        return float(t_str[:-2]) / 1000.0
    elif t_str.endswith("ms"):
        return float(t_str[:-2])
    elif t_str.endswith("s"):
        return float(t_str[:-1]) * 1000.0
    return 0.0

total_tracked_ms = 0.0

for line in table_str.split("\n"):
    tokens = line.split()
    if len(tokens) < 11:
        continue
    
    if not (tokens[-1].isdigit() and "%" in tokens[-4] and "%" in tokens[-10]):
        continue
    
    self_cuda_str = tokens[-5]
    ms = parse_time_to_ms(self_cuda_str)
    if ms <= 0:
        continue
        
    name = " ".join(tokens[:-10])
    name_lower = name.lower()
    
    if "cutlass" in name_lower or "turing_fp16" in name_lower or "cunn_softmax" in name_lower:
        continue
    if "dsalttrainfunction" in name_lower:
        continue
        
    total_tracked_ms += ms
    
    if "triton" in name_lower or "_kernel" in name_lower:
        buckets["triton (attention/RoPE)"] += ms
    elif "::mm" in name_lower or "linear" in name_lower or "matmul" in name_lower:
        buckets["gemm/mm (proiezioni+ffn)"] += ms
    elif "copy" in name_lower or "to_copy" in name_lower or "to" in name_lower:
        buckets["cast/copy (autocast fp16)"] += ms
    elif "softmax" in name_lower or "reduce" in name_lower:
        buckets["softmax/reduce"] += ms
    elif "elementwise" in name_lower or "add" in name_lower or "mul" in name_lower or "fill" in name_lower:
        buckets["elementwise"] += ms
    elif "rms_norm" in name_lower or "norm" in name_lower:
        buckets["rms_norm"] += ms
    else:
        buckets["altro"] += ms

print("\n" + "=" * 60)
print(f"RIEPILOGO CUDA PER CATEGORIA (Totale Effettivo: {total_tracked_ms:.2f} ms)")
print("=" * 60)
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    pct = (100 * v / total_tracked_ms) if total_tracked_ms > 0 else 0
    print(f"  {k:30s} {v:8.2f} ms  {pct:5.1f}%")
print("=" * 60)