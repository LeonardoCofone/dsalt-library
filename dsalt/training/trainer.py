import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .gpu_auto import (
    get_device,
    get_gpu_memory_stats,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    barrier,
)
from .logging_config import get_logger, StepTimer


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module
    if hasattr(model, "_orig_mod"):
        inner = model._orig_mod
        return inner.module if isinstance(inner, DDP) else inner
    return model


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> dict:
    m = _unwrap_model(model)
    m.eval()

    device    = ids.device
    total_len = ids.shape[0]

    x = m.embed_tokens(ids)
    layer_hiddens = [x.clone()]
    for layer in m.layers:
        x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        layer_hiddens.append(x.clone())

    last_attn = m.layers[-1].attn
    sigma2    = float("nan")
    eff_rank  = float("nan")
    sigma2_per_layer        = []
    eff_rank_per_layer_attn = []

    if hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
        P     = last_attn._last_P.float()
        P_avg = P.mean(dim=0).detach().cpu()
        sv    = torch.linalg.svdvals(P_avg)
        sigma2   = sv[1].item() if sv.shape[0] > 1 else 0.0
        sv_norm  = sv / (sv.sum() + 1e-9)
        eff_rank = torch.exp(-(sv_norm * (sv_norm + 1e-9).log()).sum()).item()

    for layer in m.layers:
        attn = layer.attn
        if hasattr(attn, "_last_P") and attn._last_P is not None:
            sv_l  = torch.linalg.svdvals(attn._last_P.float().mean(dim=0).detach().cpu())
            sigma2_per_layer.append(sv_l[1].item() if sv_l.shape[0] > 1 else 0.0)
            sv_ln = sv_l / (sv_l.sum() + 1e-9)
            eff_rank_per_layer_attn.append(torch.exp(-(sv_ln * (sv_ln + 1e-9).log()).sum()).item())
        else:
            sigma2_per_layer.append(float("nan"))
            eff_rank_per_layer_attn.append(float("nan"))

    eff_rank_per_layer = []
    for h_l in layer_hiddens[1:]:
        sv_h  = torch.linalg.svdvals(h_l.float().detach().cpu())
        sv_hn = sv_h / (sv_h.sum() + 1e-9)
        eff_rank_per_layer.append(torch.exp(-(sv_hn * (sv_hn + 1e-9).log()).sum()).item())

    h_final  = layer_hiddens[-1]
    h_mean   = h_final.mean(dim=0, keepdim=True)
    res_norm = ((h_final - h_mean).norm() / (layer_hiddens[0].norm() + 1e-9)).item()

    res_per_layer = []
    h0_norm       = layer_hiddens[0].norm().item()
    for h_l in layer_hiddens[1:]:
        xm = h_l.mean(dim=0, keepdim=True)
        res_per_layer.append(((h_l - xm).norm() / (h0_norm + 1e-9)).item())

    attn_entropy      = float("nan")
    entropy_per_layer = []
    if hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
        P_safe       = last_attn._last_P.float().clamp(min=1e-9)
        attn_entropy = -(P_safe * P_safe.log()).sum(dim=-1).mean().item()

    for layer in m.layers:
        attn = layer.attn
        if hasattr(attn, "_last_P") and attn._last_P is not None:
            P_l = attn._last_P.float().clamp(min=1e-9)
            entropy_per_layer.append(-(P_l * P_l.log()).sum(dim=-1).mean().item())
        else:
            entropy_per_layer.append(float("nan"))

    min_dist             = 64
    n_pairs              = min(64, total_len - min_dist)
    token_dist           = float("nan")
    token_dist_per_layer = []
    if n_pairs > 0:
        pairs_i    = torch.randint(min_dist, total_len, (n_pairs,), device=device)
        pairs_j    = pairs_i - min_dist
        hi         = layer_hiddens[-1][pairs_i]
        hj         = layer_hiddens[-1][pairs_j]
        token_dist = (hi - hj).norm(dim=-1).mean().item()
        for h_l in layer_hiddens[1:]:
            hi_l = h_l[pairs_i]
            hj_l = h_l[pairs_j]
            token_dist_per_layer.append((hi_l - hj_l).norm(dim=-1).mean().item())

    noise_norm      = float("nan")
    noise_per_layer = []
    seq0_len        = (cu_seqlens[1] - cu_seqlens[0]).item()
    if seq0_len > 128:
        inject_pos = int(seq0_len // 4)
        ids_pert   = ids.clone()
        ids_pert[inject_pos] = torch.randint(0, m.vocab_size, (1,), device=device).item()

        x_pert             = m.embed_tokens(ids_pert)
        layer_hiddens_pert = [x_pert.clone()]
        for layer in m.layers:
            x_pert = layer(x_pert, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            layer_hiddens_pert.append(x_pert.clone())

        far_start = inject_pos + 64
        far_end   = int(cu_seqlens[1].item())
        if far_end > far_start:
            noise_norm = (
                layer_hiddens_pert[-1][far_start:far_end] - layer_hiddens[-1][far_start:far_end]
            ).norm(dim=-1).mean().item()
            for hl_orig, hl_pert in zip(layer_hiddens[1:], layer_hiddens_pert[1:]):
                noise_per_layer.append(
                    (hl_pert[far_start:far_end] - hl_orig[far_start:far_end]).norm(dim=-1).mean().item()
                )

    head_spec_std = float("nan")
    attn_sink     = float("nan")
    if hasattr(last_attn, "_last_P") and last_attn._last_P is not None:
        P             = last_attn._last_P.float().clamp(min=1e-9)
        entropy_heads = -(P * P.log()).sum(dim=-1).mean(dim=-1)
        head_spec_std = entropy_heads.std(correction=0).item() if entropy_heads.numel() > 0 else float("nan")
        attn_sink     = last_attn._last_P.float()[:, :, 0].mean().item()

    alpha_per_head = []
    for layer in m.layers:
        attn = layer.attn
        if hasattr(attn, "alpha_w"):
            alpha_per_head.append(torch.sigmoid(attn.alpha_w).detach().cpu().tolist())

    oow_mass_per_layer = []
    for layer in m.layers:
        attn = layer.attn
        if hasattr(attn, "_last_P") and attn._last_P is not None and hasattr(attn, "n_min"):
            P_l       = attn._last_P.float()
            T_l       = P_l.shape[-1]
            positions = torch.arange(T_l, device=P_l.device).float()
            dist      = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            in_window = (dist < attn.n_min).unsqueeze(0).unsqueeze(0)
            oow       = P_l.masked_fill(in_window, 0.0).sum(dim=-1).mean().item()
            oow_mass_per_layer.append(oow)
        else:
            oow_mass_per_layer.append(float("nan"))

    m.train()

    return {
        "sigma2":                sigma2,
        "eff_rank":              eff_rank,
        "res_norm":              res_norm,
        "attn_entropy":          attn_entropy,
        "noise_norm":            noise_norm,
        "token_dist":            token_dist,
        "head_spec_std":         head_spec_std,
        "attn_sink":             attn_sink,
        "sigma2_per_layer":      sigma2_per_layer,
        "entropy_per_layer":     entropy_per_layer,
        "noise_per_layer":       noise_per_layer,
        "eff_rank_per_layer":    eff_rank_per_layer,
        "res_per_layer":         res_per_layer,
        "token_dist_per_layer":  token_dist_per_layer,
        "alpha_per_head":        alpha_per_head,
        "oow_mass_per_layer":    oow_mass_per_layer,
    }


class DSALTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 0.5,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        grad_accum: int = 1,
        log_every: int = 100,
        val_every: int = 500,
        save_every: int = 1000,
        save_dir: str = "./checkpoints_dsalt",
        mixed_precision: str = "bf16",
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        ddp_backend: str = "nccl",
        seed: int = 42,
    ):
        self.rank       = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main    = is_main_process(rank)

        self.lr                     = lr
        self.weight_decay           = weight_decay
        self.max_grad_norm          = max_grad_norm
        self.warmup_steps           = warmup_steps
        self.total_steps            = total_steps
        self.grad_accum             = grad_accum
        self.log_every              = log_every
        self.val_every              = val_every
        self.save_every             = save_every
        self.save_dir               = Path(save_dir)
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model          = compile_model
        self.ddp_backend            = ddp_backend
        self.seed                   = seed

        torch.manual_seed(seed + rank)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("dsalt.trainer", log_dir=str(self.save_dir))
        self.device = get_device(local_rank)

        self._amp_dtype = self._resolve_amp_dtype(mixed_precision)
        self._use_amp   = self._amp_dtype is not None
        self._scaler    = (
            torch.amp.GradScaler("cuda")
            if self._use_amp and self._amp_dtype == torch.float16
            else None
        )

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        model = model.to(self.device)

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        self.model = model

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer)

        self.global_step       = 0
        self.best_val_ppl      = float("inf")
        self._timer            = StepTimer(window=50, device=self.device)
        self._tokens_per_batch = 0
        self._accum_loss_sum   = 0.0
        self._accum_loss_steps = 0

        self.history = {k: [] for k in [
            "train_loss", "train_ppl", "sigma2", "eff_rank", "res_norm", "attn_entropy",
            "noise_norm", "token_dist", "head_spec_std", "attn_sink",
            "sigma2_per_layer", "entropy_per_layer", "noise_per_layer",
            "eff_rank_per_layer", "res_per_layer", "token_dist_per_layer",
            "alpha_per_head", "oow_mass_per_layer",
            "val_ppl", "val_steps", "gpu_mem_gb", "it_s",
        ]}

    def _resolve_amp_dtype(self, mixed_precision: str) -> torch.dtype | None:
        if mixed_precision == "bf16" and self.device.type == "cuda":
            return torch.bfloat16
        if mixed_precision == "fp16" and self.device.type == "cuda":
            return torch.float16
        return None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        base = _unwrap_model(self.model)
        decay, nodecay, dsalt_params = [], [], []

        for name, p in base.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ["win_gate", "alpha_w"]):
                dsalt_params.append(p)
            elif p.ndim < 2 or any(k in name for k in ("norm", "bias", "embed")):
                nodecay.append(p)
            else:
                decay.append(p)

        return torch.optim.AdamW(
            [
                {"params": decay,        "weight_decay": self.weight_decay, "lr": self.lr},
                {"params": nodecay,      "weight_decay": 0.0,               "lr": self.lr},
                {"params": dsalt_params, "weight_decay": 0.0,               "lr": self.lr * 2.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.device.type == "cuda",
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _extract_batch(self, batch):
        ids, labels, cu_seqlens, max_seqlen = batch
        return (
            ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
            cu_seqlens.to(self.device, non_blocking=True),
            int(max_seqlen),
        )

    def _forward_step(self, batch) -> torch.Tensor:
        ids, labels, cu_seqlens, max_seqlen = self._extract_batch(batch)
        self._tokens_per_batch = ids.numel()
        self._last_ids         = ids
        self._last_cu_seqlens  = cu_seqlens
        self._last_max_seqlen  = max_seqlen

        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            out  = self.model(
                ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                labels=labels,
                gradient_checkpointing=self.gradient_checkpointing,
            )
            loss = out["loss"]
        return loss

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        for batch in self.val_loader:
            ids, labels, cu_seqlens, max_seqlen = self._extract_batch(batch)
            valid_tokens  = (labels != -100).sum().item()
            out           = self.model(
                ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                labels=labels,
                gradient_checkpointing=False,
            )
            total_loss   += out["loss"].item() * valid_tokens
            total_tokens += valid_tokens

        self.model.train()

        if self.world_size > 1:
            t = torch.tensor([total_loss, float(total_tokens)], device=self.device)
            torch.distributed.all_reduce(t)
            total_loss, total_tokens = t[0].item(), t[1].item()

        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20.0))

    def _save_checkpoint(self, tag: str) -> None:
        if not self.is_main:
            return
        ckpt = {
            "step":                 self.global_step,
            "model_state_dict":     _unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl":         self.best_val_ppl,
            "history":              self.history,
        }
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        self.logger.info(f"checkpoint salvato → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        _unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        self.history      = ckpt.get("history", self.history)
        if self.is_main:
            self.logger.info(f"checkpoint ripreso dallo step {self.global_step}")

    def _log_step(self, accum_loss: float) -> None:
        if not self.is_main:
            return

        stats  = self._timer.stop()
        it_s   = stats.get("it_s", 0.0)
        mem_gb = 0.0
        if self.device.type == "cuda":
            mem_gb = get_gpu_memory_stats(self.device).get("allocated_gb", 0.0)

        lr_now   = self.scheduler.get_last_lr()[0]
        train_ppl = math.exp(min(accum_loss, 20.0))

        metrics = compute_metrics(
            _unwrap_model(self.model),
            self._last_ids,
            self._last_cu_seqlens,
            self._last_max_seqlen,
        )

        def _fs(v) -> str:
            return f"{v:.6f}" if math.isfinite(v) else "nan"

        msg = (
            f"step={self.global_step} | "
            f"loss={accum_loss:.4f} | "
            f"ppl={train_ppl:.4f} | "
            f"lr={lr_now:.2e} | "
            f"σ²={_fs(metrics['sigma2'])} | "
            f"rank={_fs(metrics['eff_rank'])} | "
            f"res={metrics['res_norm']:.4f} | "
            f"H={_fs(metrics['attn_entropy'])} | "
            f"noise={_fs(metrics['noise_norm'])} | "
            f"sink={_fs(metrics['attn_sink'])} | "
            f"head_std={_fs(metrics['head_spec_std'])}"
        )

        self.logger.info(msg, extra={"it_s": it_s, "mem_gb": mem_gb})

        self.history["train_loss"].append(accum_loss)
        self.history["train_ppl"].append(train_ppl)
        self.history["it_s"].append(it_s)
        self.history["gpu_mem_gb"].append(mem_gb)
        for k in ["sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
                  "token_dist", "head_spec_std", "attn_sink",
                  "sigma2_per_layer", "entropy_per_layer", "noise_per_layer",
                  "eff_rank_per_layer", "res_per_layer", "token_dist_per_layer",
                  "alpha_per_head", "oow_mass_per_layer"]:
            self.history[k].append(metrics[k])

        self._timer.start()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        data_iter  = iter(self.train_loader)
        accum_loss = 0.0

        if self.is_main:
            n_params = sum(p.numel() for p in _unwrap_model(self.model).parameters() if p.requires_grad)
            mode     = (
                f"DDP×{self.world_size} (backend={self.ddp_backend})"
                if self.world_size > 1 else "1×GPU"
            )
            self.logger.info(
                f"training start | mode={mode} | steps={self.total_steps} | "
                f"grad_accum={self.grad_accum} | device={self.device} | "
                f"amp={self._amp_dtype} | gc={self.gradient_checkpointing} | "
                f"params={n_params:,}"
            )

        self._timer.start()

        while self.global_step < self.total_steps:
            for _ in range(self.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if isinstance(self.train_loader.sampler, DistributedSampler):
                        self.train_loader.sampler.set_epoch(self.global_step)
                    data_iter = iter(self.train_loader)
                    batch     = next(data_iter)

                loss        = self._forward_step(batch) / self.grad_accum
                accum_loss += loss.item()

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self._scaler is not None:
                if self.max_grad_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                self._log_step(accum_loss)
                accum_loss = 0.0

            if self.global_step % self.val_every == 0:
                barrier(self.rank, self.world_size)
                val_ppl = self._validate()
                if self.is_main:
                    self.history["val_ppl"].append(val_ppl)
                    self.history["val_steps"].append(self.global_step)
                    is_best = val_ppl < self.best_val_ppl
                    self.logger.info(
                        f"step={self.global_step} | val_ppl={val_ppl:.4f}"
                        + (" ← best" if is_best else "")
                    )
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best")

            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            if self.global_step >= self.total_steps:
                break

            self._timer.start()

        self._save_checkpoint("final")
        if self.is_main:
            self.logger.info(f"done | best_val_ppl={self.best_val_ppl:.4f}")

        cleanup_ddp()