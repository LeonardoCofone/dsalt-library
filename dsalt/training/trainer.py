import math
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dsalt.training.gpu_auto import resolve_device, print_gpu_info
from dsalt.training.logging_config import setup_logging

logger = logging.getLogger(__name__)

DSALT_PARAM_KEYWORDS = ("window_pred", "alpha_w")
NO_DECAY_KEYWORDS    = ("norm", "bias", "emb", "tok_emb", "pos_emb")


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
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
        train_loader: DataLoader,
        val_loader: Optional[DataLoader]        = None,
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
        gradient_checkpointing: bool            = False,
        device: str                             = "cpu",
        num_gpus: int                           = 1,
        log_file: Optional[str]                 = None,
    ):
        setup_logging(level=logging.INFO, log_file=log_file)
        logger.info("=" * 60)
        logger.info("DSALT Trainer init starting")
        logger.info("=" * 60)

        self.primary_device, self.gpu_ids = resolve_device(device, num_gpus)
        self.use_dp = len(self.gpu_ids) > 1
        logger.info(f"Device resolved  →  primary={self.primary_device}  gpu_ids={self.gpu_ids}  DataParallel={self.use_dp}")

        if self.primary_device.type == "cuda":
            logger.info(print_gpu_info())

        self._gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self._gradient_checkpointing_enabled = self._enable_gradient_checkpointing(model)
            logger.info(f"Gradient checkpointing  →  {self._gradient_checkpointing_enabled}")

        logger.info(f"Moving model to {self.primary_device}")
        model = model.to(self.primary_device)
        logger.info("Model on device  ✓")

        if self.use_dp:
            print(f"[DEBUG] Moving model to cpu for DataParallel", flush=True)
            model = model.cpu()
            print(f"[DEBUG] Model on cpu", flush=True)
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
            print(f"[DEBUG] Wrapped in DataParallel", flush=True)
            logger.info(f"Wrapped in DataParallel  →  GPUs {self.gpu_ids}")
            # DataParallel requires model on device_ids[0]
            model = model.to(self.primary_device)
            print(f"[DEBUG] Moved wrapped model to {self.primary_device}", flush=True)
        elif len(self.gpu_ids) > 0:
            logger.info(f"Single GPU mode  →  {self.primary_device}")
        else:
            logger.info("CPU mode")

        if not self.use_dp:
            logger.info(f"Moving model to {self.primary_device}")
            model = model.to(self.primary_device)
            logger.info("Model on device  ✓")

        self.model       = model
        self._model_base = model.module if self.use_dp else model
        print(f"[DEBUG] _model_base type: {type(self._model_base)}", flush=True)
        self.device      = self.primary_device

        print(f"[DEBUG] Before named_parameters", flush=True)
        decay, no_decay, dsalt_params = [], [], []
        for name, p in self._model_base.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in DSALT_PARAM_KEYWORDS):
                dsalt_params.append(p)
            elif any(kw in name for kw in NO_DECAY_KEYWORDS):
                no_decay.append(p)
            else:
                decay.append(p)
        print(f"[DEBUG] After param grouping", flush=True)

        total_params = sum(p.numel() for p in self._model_base.parameters())
        trainable    = sum(p.numel() for p in self._model_base.parameters() if p.requires_grad)
        logger.info(f"Parameters  →  total={total_params:,}  trainable={trainable:,}")
        logger.info(f"Param groups  →  decay={len(decay)}  no_decay={len(no_decay)}  dsalt={len(dsalt_params)}")

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
        logger.info(f"Optimizer  →  AdamW  lr={lr}  wd={weight_decay}  warmup={warmup_steps}  total={total_steps}")

        self.dtype   = dtype
        self.use_amp = self.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        self.scaler  = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and dtype == torch.float16
            else None
        )
        logger.info(f"AMP  →  use_amp={self.use_amp}  dtype={dtype}  scaler={'yes (fp16)' if self.scaler else 'no'}")

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
        self._zero_tensor       = torch.tensor(0.0, device=self.device)
        self.history: Dict[str, list] = {
            "train_loss": [], "val_ppl": [], "val_steps": [],
            "step_time": [], "gpu_mem_gb": [], "lr": [],
            "sigma2": [], "eff_rank": [], "res_norm": [], "attn_entropy": [],
            "noise_norm": [], "head_spec_std": [], "attn_sink": [],
        }
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints  →  {self.save_dir.resolve()}")

        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info("=" * 60)
        logger.info(f"Trainer ready  →  steps={total_steps}  log_every={log_every}  val_every={val_every}  save_every={save_every}  grad_accum={grad_accum}")
        logger.info("=" * 60)
        print("[DEBUG] __init__ completed", flush=True)

    @staticmethod
    def _enable_gradient_checkpointing(model: nn.Module) -> bool:
        try:
            from dsalt.modules.dsalt_attention import DSALTAttention
            count = 0
            for module in model.modules():
                if isinstance(module, DSALTAttention):
                    module.gradient_checkpointing = True
                    count += 1
            if count > 0:
                logger.debug(f"Gradient checkpointing enabled for {count} DSALTAttention modules")
                return True
            return False
        except ImportError:
            logger.warning("Could not import DSALTAttention for gradient checkpointing")
            return False

    def _forward(self, x: torch.Tensor, y: torch.Tensor):
        print(f"[TRAINER] _forward start x.shape={x.shape} y.shape={y.shape} device={x.device}", flush=True)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            print(f"[TRAINER] calling model", flush=True)
            logits, cont_windows = self.model(x, return_window=self.window_reg_coef > 0)
            print(f"[TRAINER] model done logits.shape={logits.shape}", flush=True)
            B, N, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.view(B * N, V),
                y.contiguous().view(B * N),
                ignore_index=-100,
            )
            win_reg = self._zero_tensor.clone()
            if self.window_reg_coef > 0 and cont_windows is not None:
                if isinstance(cont_windows, torch.Tensor):
                    cont_windows = [cont_windows]
                for cw in cont_windows:
                    win_reg = win_reg - cw.float().var(dim=-1).mean()
                win_reg = win_reg / max(len(cont_windows), 1)
        print(f"[TRAINER] _forward done ce={ce.item():.4f}", flush=True)
        return ce + self.window_reg_coef * win_reg, ce.detach(), win_reg.detach()

    def train(self) -> Dict[str, list]:
        self.model.train()
        logger.info(f"train() starting from step {self.global_step}")

        # Log dataloader info
        try:
            n_batches = len(self.train_loader)
            logger.info(f"Train loader  →  {n_batches} batches/epoch")
        except TypeError:
            logger.info("Train loader  →  iterable (no len)")

        data_iter           = iter(self.train_loader)
        loss_accum          = 0.0
        n_steps_accumulated = 0
        t0                  = time.time()
        t_train_start       = time.time()

        # Log first batch shape
        _first_batch_logged = False

        while self.global_step < self.total_steps:
            for acc_step in range(self.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                if not _first_batch_logged:
                    logger.info(f"First batch  →  x={tuple(x.shape)}  y={tuple(y.shape)}  x.dtype={x.dtype}")
                    logger.info("Running first forward pass (Triton JIT compile on first call, may take ~10s) ...")
                    _first_batch_logged = True

                loss, ce, win_reg = self._forward(x, y)

                if self.global_step == 0 and acc_step == 0:
                    logger.info(f"First forward done  →  ce={ce.item():.4f}")

                loss = loss / self.grad_accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.global_step == 0 and acc_step == 0:
                    logger.info("First backward done  ✓")

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

            if self.global_step == 1:
                logger.info(f"Step 1 complete  →  total elapsed {time.time() - t_train_start:.1f}s")

            if self.global_step % self.log_every == 0:
                elapsed    = time.time() - t0
                avg_loss   = loss_accum / n_steps_accumulated
                ppl        = math.exp(min(avg_loss, 1000))
                lr_now     = self.scheduler.get_last_lr()[0]
                it_per_sec = self.log_every / elapsed if elapsed > 0 else float("inf")
                mem_gb     = (
                    torch.cuda.memory_allocated(self.device) / 1e9
                    if self.device.type == "cuda" else 0.0
                )
                mem_reserved = (
                    torch.cuda.memory_reserved(self.device) / 1e9
                    if self.device.type == "cuda" else 0.0
                )

                log_str = (
                    f"step {self.global_step:>6d}/{self.total_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  ppl={ppl:.2f}"
                    f"  lr={lr_now:.2e}"
                    f"  win_reg={win_reg.item():.4f}"
                    f"  mem={mem_gb:.2f}/{mem_reserved:.2f}GB"
                    f"  {it_per_sec:.2f}it/s"
                )

                if self.compute_metrics_fn is not None and self.global_step % (self.log_every * 5) == 0:
                    m = self.compute_metrics_fn(self._model_base, x[:1])
                    log_str += (
                        f"\n         σ₂={m['sigma2']:.4f}"
                        f"  rank={m['eff_rank']:.1f}"
                        f"  res={m['res_norm']:.4f}"
                        f"  H={m['attn_entropy']:.4f}"
                        f"  noise={m['noise_norm']:.4f}"
                        f"  sink={m['attn_sink']:.4f}"
                        f"  head_std={m['head_spec_std']:.4f}"
                    )
                    for k in ["sigma2", "eff_rank", "res_norm", "attn_entropy",
                              "noise_norm", "head_spec_std", "attn_sink"]:
                        if k in m:
                            self.history[k].append(m[k])

                logger.info(log_str)
                self.history["train_loss"].append(avg_loss)
                self.history["step_time"].append(elapsed / self.log_every)
                self.history["gpu_mem_gb"].append(mem_gb)
                self.history["lr"].append(lr_now)
                loss_accum          = 0.0
                n_steps_accumulated = 0
                t0                  = time.time()

            if self.val_loader and self.global_step % self.val_every == 0:
                logger.info(f"Running validation at step {self.global_step} ...")
                val_ppl = self._validate()
                self.history["val_ppl"].append(val_ppl)
                self.history["val_steps"].append(self.global_step)
                improved = val_ppl < self.best_val_ppl
                logger.info(
                    f"Validation  →  ppl={val_ppl:.2f}"
                    f"  best={self.best_val_ppl:.2f}"
                    f"{'  [NEW BEST]' if improved else ''}"
                )
                if improved:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best.pt")
                self.model.train()

            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")

        total_time = time.time() - t_train_start
        logger.info("=" * 60)
        logger.info(f"Training complete  →  steps={self.global_step}  best_val_ppl={self.best_val_ppl:.2f}  total_time={total_time/60:.1f}min")
        logger.info("=" * 60)
        self._save_checkpoint("final.pt")
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
        return math.exp(min(total_loss / max(total_tokens, 1), 20))

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "step":            self.global_step,
                "model_state":     self._model_base.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_ppl":    self.best_val_ppl,
                "history":         self.history,
            },
            path,
        )
        logger.info(f"Checkpoint saved  →  {path}")

    def _load_checkpoint(self, path: str) -> None:
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self._model_base.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.global_step  = ckpt["step"]
            self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
            self.history      = ckpt.get("history", self.history)
            logger.info(f"Resumed from {path}  →  step={self.global_step}  best_val_ppl={self.best_val_ppl:.2f}")
        except (FileNotFoundError, IOError, RuntimeError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise