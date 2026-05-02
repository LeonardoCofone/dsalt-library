"""
dsalt/training/trainer.py
--------------------------
Minimal but complete training loop for DSALT language models.

Features:
  - Mixed precision (bf16 / fp16) via torch.autocast
  - Gradient clipping
  - Cosine LR schedule with linear warmup
  - Optional window entropy regularisation
  - Periodic validation + perplexity logging
  - Checkpoint save / resume
"""

import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ═════════════════════════════════════════════════════════════════════════════
# LR Schedule: cosine with linear warmup
# ═════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer:       torch.optim.Optimizer,
    warmup_steps:    int,
    total_steps:     int,
    min_lr_ratio:    float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup then cosine decay to min_lr_ratio * base_lr.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════════════════
# Trainer
# ═════════════════════════════════════════════════════════════════════════════

class DSALTTrainer:
    """
    Training harness for DSALTTransformer.

    Parameters
    ----------
    model          : DSALTTransformer instance
    train_loader   : DataLoader yielding (input_ids,) or (input_ids, labels)
    val_loader     : DataLoader for validation (optional)
    lr             : peak learning rate
    weight_decay   : AdamW weight decay
    max_grad_norm  : gradient clip norm
    warmup_steps   : linear warmup steps
    total_steps    : total training steps
    log_every      : log interval in steps
    val_every      : validation interval in steps
    save_every     : checkpoint interval in steps
    save_dir       : directory for checkpoints
    dtype          : torch.bfloat16 / torch.float16 / torch.float32
    window_reg_coef: coefficient for window entropy regularisation (0 = off)
    device         : torch.device (auto-detected if None)
    resume_from    : path to a checkpoint to resume from
    """

    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      Optional[DataLoader] = None,
        lr:              float = 3e-4,
        weight_decay:    float = 0.1,
        max_grad_norm:   float = 1.0,
        warmup_steps:    int   = 500,
        total_steps:     int   = 10_000,
        log_every:       int   = 50,
        val_every:       int   = 500,
        save_every:      int   = 1000,
        save_dir:        str   = "checkpoints",
        dtype:           torch.dtype = torch.bfloat16,
        window_reg_coef: float = 0.0,
        device:          Optional[torch.device] = None,
        resume_from:     Optional[str] = None,
    ):
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.max_grad_norm   = max_grad_norm
        self.total_steps     = total_steps
        self.log_every       = log_every
        self.val_every       = val_every
        self.save_every      = save_every
        self.save_dir        = Path(save_dir)
        self.dtype           = dtype
        self.window_reg_coef = window_reg_coef

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model  = self.model.to(device)

        # Optimizer (separate WD for embeddings / LayerNorm params)
        decay_params    = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in ("norm", "bias", "emb")):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": decay_params,    "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision scaler (only for fp16; bf16 doesn't need it)
        self.use_amp  = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        self.scaler   = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

        self.global_step = 0
        self.best_val_ppl = float("inf")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        if resume_from:
            self._load_checkpoint(resume_from)

    # ─────────────────────────────────────────────────────────────────────────

    def _forward_step(
        self,
        batch: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward + loss computation.
        Expects batch to be either:
          - a Tensor of shape [B, N+1]  (input_ids; last col is target)
          - a tuple (input_ids [B, N], labels [B, N])
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch[0], batch[1]
        elif isinstance(batch, (list, tuple)) and len(batch) == 1:
            # TensorDataset with single tensor → treat as [B, N+1] sequence
            batch = batch[0]
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:]
        else:
            # Shift: input = batch[:, :-1], label = batch[:, 1:]
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:]

        input_ids = input_ids.to(self.device)
        labels    = labels.to(self.device)

        # Forward pass (with optional window regularisation)
        return_windows = self.window_reg_coef > 0
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.use_amp,
        ):
            logits, cont_windows = self.model(
                input_ids, return_windows=return_windows
            )
            # logits: [B, N, V]

            # Cross-entropy loss
            B, N, V = logits.shape
            ce_loss = nn.functional.cross_entropy(
                logits.view(B * N, V),
                labels.contiguous().view(B * N),
                ignore_index=-100,
            )

            # Window entropy regularisation
            win_reg = torch.tensor(0.0, device=self.device)
            if return_windows and cont_windows:
                for cw in cont_windows:
                    # cw: [B, N]; penalise low variance (collapsed window)
                    win_reg = win_reg + (-cw.var(dim=-1).mean())
                win_reg = win_reg / len(cont_windows)

            total_loss = ce_loss + self.window_reg_coef * win_reg

        return {
            "loss":     total_loss,
            "ce_loss":  ce_loss.detach(),
            "win_reg":  win_reg.detach(),
        }

    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        self.model.train()
        data_iter   = iter(self.train_loader)
        t0          = time.time()
        running_loss = 0.0

        while self.global_step < self.total_steps:
            # Fetch next batch (loop around if dataset exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # ── Forward ──────────────────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)
            metrics = self._forward_step(batch)
            loss    = metrics["loss"]

            # ── Backward ─────────────────────────────────────────────────
            if self.dtype == torch.float16 and self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            running_loss += metrics["ce_loss"].item()

            # ── Logging ──────────────────────────────────────────────────
            if self.global_step % self.log_every == 0:
                elapsed  = time.time() - t0
                avg_loss = running_loss / self.log_every
                ppl      = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                lr_now   = self.scheduler.get_last_lr()[0]
                print(
                    f"[step {self.global_step:>6d}/{self.total_steps}] "
                    f"loss={avg_loss:.4f}  ppl={ppl:.2f}  "
                    f"lr={lr_now:.2e}  "
                    f"win_reg={metrics['win_reg'].item():.4f}  "
                    f"time={elapsed:.1f}s"
                )
                running_loss = 0.0
                t0 = time.time()

            # ── Validation ───────────────────────────────────────────────
            if self.val_loader and self.global_step % self.val_every == 0:
                val_ppl = self._validate()
                print(f"  └─ val_ppl={val_ppl:.2f}")
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best.pt")
                self.model.train()

            # ── Checkpointing ────────────────────────────────────────────
            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")

        # Final checkpoint
        self._save_checkpoint("final.pt")
        print(f"Training complete. Best val ppl: {self.best_val_ppl:.2f}")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss  = 0.0
        total_tokens = 0

        for batch in self.val_loader:
            metrics = self._forward_step(batch)
            # Count non-padding tokens
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                labels = batch[1].to(self.device)
            elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                labels = batch[0][:, 1:].to(self.device)
            else:
                labels = batch[:, 1:].to(self.device)
            n_toks = (labels != -100).sum().item()
            total_loss   += metrics["ce_loss"].item() * n_toks
            total_tokens += n_toks

        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20))

    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "step":           self.global_step,
                "model_state":    self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_ppl":   self.best_val_ppl,
            },
            path,
        )
        print(f"  └─ saved checkpoint → {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        print(f"Resumed from {path} (step {self.global_step})")