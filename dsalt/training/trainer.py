import math
import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from dsalt.training.gpu_auto import GPUAutoConfig

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DSALTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader: Optional[Any]               = None,
        lr: float                               = 3e-4,
        weight_decay: float                     = 0.1,
        max_grad_norm: float                    = 1.0,
        warmup_steps: int                       = 500,
        total_steps: int                        = 10_000,
        grad_accum: int                         = 1,
        log_every: int                          = 50,
        val_every: int                          = 500,
        save_every: int                         = 1000,
        save_dir: str                           = "checkpoints",
        dtype: torch.dtype                      = torch.bfloat16,
        window_reg_coef: float                  = 0.0,
        compute_metrics_fn: Optional[Callable]  = None,
        resume_from: Optional[str]              = None,
        ddp: bool                               = False,
        fsdp: bool                              = False,
        fsdp_cpu_offload: bool                  = False,
        gradient_checkpointing: bool            = False,
    ):
        assert not (ddp and fsdp), "Choose DDP or FSDP, not both."

        self.gpu_config = GPUAutoConfig(verbose=True)

        in_torchrun = self.gpu_config.in_torchrun
        if in_torchrun and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        if in_torchrun and not ddp and not fsdp:
            ddp = True

        self.ddp  = ddp
        self.fsdp = fsdp
        self.device = self.gpu_config.device

        if gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        model = model.to(self.device)

        if ddp:
            assert dist.is_initialized(), "dist not initialized. Use torchrun."
            model = DDP(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        elif fsdp:
            assert _FSDP_AVAILABLE, "FSDP not available in this version of PyTorch."
            assert dist.is_initialized(), "dist not initialized. Use torchrun."
            mp_policy = (
                MixedPrecision(
                    param_dtype=dtype,
                    reduce_dtype=torch.float32,
                    buffer_dtype=dtype,
                )
                if dtype in (torch.float16, torch.bfloat16)
                else None
            )
            cpu_off = CPUOffload(offload_params=True) if fsdp_cpu_offload else None
            from dsalt.modules.dsalt_transformer import DSALTBlock
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={DSALTBlock},
            )
            model = FSDP(
                model,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mp_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=cpu_off,
                device_id=self.device,
            )

        self.model = model
        self._model_base = (
            model.module
            if isinstance(model, (DDP, nn.DataParallel))
            else model
        )

        decay, no_decay, dsalt_params = [], [], []
        param_source = self._model_base if not fsdp else model
        for name, p in param_source.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ("window_pred", "alpha_w")):
                dsalt_params.append(p)
            elif any(kw in name for kw in ("norm", "bias", "emb", "tok_emb", "pos_emb")):
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = AdamW(
            [
                {"params": decay,        "lr": lr,       "weight_decay": weight_decay},
                {"params": no_decay,     "lr": lr,       "weight_decay": 0.0},
                {"params": dsalt_params, "lr": lr * 2.0, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        self.dtype   = dtype
        self.use_amp = dtype in (torch.float16, torch.bfloat16) and self.device.type == "cuda"
        self.scaler  = (
            torch.amp.GradScaler("cuda")
            if dtype == torch.float16 and self.use_amp
            else None
        )

        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.max_grad_norm      = max_grad_norm
        self.total_steps        = total_steps
        self.grad_accum         = grad_accum
        self.log_every          = log_every
        self.val_every          = val_every
        self.save_every         = save_every
        self.save_dir           = Path(save_dir)
        self.window_reg_coef    = window_reg_coef
        self.compute_metrics_fn = compute_metrics_fn
        self.global_step        = 0
        self.best_val_ppl       = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [], "val_ppl": [], "val_steps": [],
            "step_time": [], "gpu_mem_gb": [], "lr": [],
            "sigma2": [], "eff_rank": [], "res_norm": [], "attn_entropy": [],
            "noise_norm": [], "head_spec_std": [], "attn_sink": [],
            "token_dist": [], "sigma2_per_layer": [], "entropy_per_layer": [],
            "noise_per_layer": [], "eff_rank_per_layer": [], "res_per_layer": [],
            "token_dist_per_layer": [], "alpha_per_head": [], "oow_mass_per_layer": [],
        }
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if resume_from:
            self._load_checkpoint(resume_from)

    @staticmethod
    def _enable_gradient_checkpointing(model: nn.Module) -> None:
        try:
            from dsalt.modules.dsalt_attention import DSALTAttention
            for module in model.modules():
                if isinstance(module, DSALTAttention):
                    module.gradient_checkpointing = True
        except ImportError:
            pass

    def _is_main(self) -> bool:
        if self.ddp or self.fsdp:
            return dist.get_rank() == 0
        return True

    def print_training_setup(self):
        if not self._is_main():
            return
        print(f"\n{'='*70}")
        print(f"Training Setup Summary")
        print(f"{'='*70}")
        print(f"Model:               {self.model.__class__.__name__}")
        print(f"Device:              {self.device}")
        print(f"Dtype:               {self.dtype}")
        print(f"Strategy:            {self.gpu_config.recommended_strategy()}")
        print(f"DDP:                 {self.ddp}")
        print(f"FSDP:                {self.fsdp}")
        print(f"Total Steps:         {self.total_steps}")
        print(f"Grad Accum:          {self.grad_accum}")
        print(f"{'='*70}\n")

    def _forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            logits, cont_windows = self.model(x, return_window=self.window_reg_coef > 0)
            B, N, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.view(B * N, V),
                y.contiguous().view(B * N),
                ignore_index=-100,
            )
            win_reg = torch.tensor(0.0, device=self.device)
            if self.window_reg_coef > 0 and cont_windows is not None:
                if isinstance(cont_windows, torch.Tensor):
                    cont_windows = [cont_windows]
                for cw in cont_windows:
                    win_reg = win_reg - cw.float().var(dim=-1).mean()
                win_reg = win_reg / max(len(cont_windows), 1)
        return ce + self.window_reg_coef * win_reg, ce.detach(), win_reg.detach()

    def train(self) -> Dict[str, list]:
        self.model.train()
        data_iter  = iter(self.train_loader)
        loss_accum = 0.0
        n_steps_accumulated = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while self.global_step < self.total_steps:
            for acc_step in range(self.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                ctx = (
                    self.model.no_sync()
                    if (self.ddp and acc_step < self.grad_accum - 1)
                    else _nullctx()
                )
                with ctx:
                    loss, ce, win_reg = self._forward(x, y)
                    loss = loss / self.grad_accum
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                loss_accum += ce.item()

            n_steps_accumulated += 1

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % self.log_every == 0 and self._is_main():
                elapsed  = time.time() - t0
                avg_loss = loss_accum / n_steps_accumulated
                ppl      = math.exp(min(avg_loss, 1000))
                lr_now   = self.scheduler.get_last_lr()[0]
                mem_gb   = (
                    torch.cuda.memory_allocated(self.device) / 1e9
                    if self.device.type == "cuda" else 0.0
                )
                log_str = (
                    f"step {self.global_step:>6d}/{self.total_steps} │ "
                    f"loss {avg_loss:.4f} │ ppl {ppl:.2f} │ "
                    f"lr {lr_now:.2e} │ win_reg {win_reg.item():.4f} │ "
                    f"mem {mem_gb:.2f}GB │ {self.log_every / elapsed:.2f}it/s"
                )
                if self.compute_metrics_fn is not None:
                    m = self.compute_metrics_fn(self._model_base, x[:1])
                    log_str += (
                        f"\n        σ₂ {m['sigma2']:.4f} │ rank {m['eff_rank']:.1f} │ "
                        f"res {m['res_norm']:.4f} │ H {m['attn_entropy']:.4f} │ "
                        f"noise {m['noise_norm']:.4f} │ sink {m['attn_sink']:.4f} │ "
                        f"head_std {m['head_spec_std']:.4f}"
                    )
                    for k in [
                        "sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
                        "head_spec_std", "attn_sink", "token_dist", "sigma2_per_layer",
                        "entropy_per_layer", "noise_per_layer", "eff_rank_per_layer",
                        "res_per_layer", "token_dist_per_layer", "alpha_per_head",
                        "oow_mass_per_layer",
                    ]:
                        if k in m:
                            self.history[k].append(m[k])

                print(f"{'─'*50}\n{log_str}")
                self.history["train_loss"].append(avg_loss)
                self.history["step_time"].append(elapsed / self.log_every)
                self.history["gpu_mem_gb"].append(mem_gb)
                self.history["lr"].append(lr_now)
                loss_accum = 0.0
                n_steps_accumulated = 0
                t0 = time.time()

            if self.val_loader and self.global_step % self.val_every == 0 and self._is_main():
                val_ppl = self._validate()
                self.history["val_ppl"].append(val_ppl)
                self.history["val_steps"].append(self.global_step)
                print(f"  ╰─ val_ppl {val_ppl:.2f} at step {self.global_step}")
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best.pt")
                self.model.train()

            if self.global_step % self.save_every == 0 and self._is_main():
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")

        if self._is_main():
            self._save_checkpoint("final.pt")
            print(f"Done. Best val ppl: {self.best_val_ppl:.2f}")

        return self.history

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, ce, _ = self._forward(x, y)
            n_toks        = (y != -100).sum().item()
            total_loss   += ce.item() * n_toks
            total_tokens += n_toks

        if self.ddp or self.fsdp:
            tl_ = torch.tensor(total_loss,   device=self.device)
            tt_ = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(tl_, op=dist.ReduceOp.SUM)
            dist.all_reduce(tt_, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tl_.item(), tt_.item()

        return math.exp(min(total_loss / max(total_tokens, 1), 20))

    def _get_state_dict(self):
        if self.fsdp and _FSDP_AVAILABLE:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                return self.model.state_dict()
        return self._model_base.state_dict()

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "step":            self.global_step,
                "model_state":     self._get_state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_ppl":    self.best_val_ppl,
                "history":         self.history,
            },
            path,
        )
        print(f"  ╰─ checkpoint → {path}")

    def _load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._model_base.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        self.history      = ckpt.get("history", self.history)
        print(f"Ripreso da {path} al passo {self.global_step}")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass