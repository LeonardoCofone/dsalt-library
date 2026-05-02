import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DSALTTrainer:
    def __init__(self, model, train_loader, val_loader=None, lr=3e-4,
                 weight_decay=0.1, max_grad_norm=1.0, warmup_steps=500,
                 total_steps=10_000, log_every=50, val_every=500,
                 save_every=1000, save_dir="checkpoints",
                 dtype=torch.bfloat16, window_reg_coef=0.0,
                 device=None, resume_from=None, ddp=False):
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
        self.ddp             = ddp

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model  = self.model.to(device)
        if self.ddp:
            self.model = DDP(self.model,
                             device_ids=[self.device] if self.device.type == "cuda" else None)

        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in ("norm", "bias", "emb")):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = AdamW(
            [{"params": decay_params,    "weight_decay": weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        # GradScaler aggiornato alla sintassi non deprecata
        self.scaler  = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

        self.global_step  = 0
        self.best_val_ppl = float("inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if resume_from:
            self._load_checkpoint(resume_from)

    def _is_main_process(self):
        return not self.ddp or dist.get_rank() == 0

    def _forward_step(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch[0], batch[1]
        else:
            batch     = batch[0] if isinstance(batch, (list, tuple)) else batch
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:]

        input_ids = input_ids.to(self.device)
        labels    = labels.to(self.device)
        return_windows = self.window_reg_coef > 0

        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.use_amp):
            logits, cont_windows = self.model(input_ids, return_windows=return_windows)
            B, N, V = logits.shape
            ce_loss = nn.functional.cross_entropy(
                logits.view(B * N, V), labels.contiguous().view(B * N),
                ignore_index=-100,
            )
            win_reg = torch.tensor(0.0, device=self.device)
            if return_windows and cont_windows:
                for cw in cont_windows:
                    win_reg = win_reg + (-cw.var(dim=-1).mean())
                win_reg = win_reg / len(cont_windows)
            total_loss = ce_loss + self.window_reg_coef * win_reg

        return {"loss": total_loss, "ce_loss": ce_loss.detach(), "win_reg": win_reg.detach()}

    def train(self):
        self.model.train()
        data_iter    = iter(self.train_loader)
        t0           = time.time()
        running_loss = 0.0

        while self.global_step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            self.optimizer.zero_grad(set_to_none=True)
            metrics = self._forward_step(batch)
            loss    = metrics["loss"]

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

            if self.global_step % self.log_every == 0 and self._is_main_process():
                elapsed  = time.time() - t0
                avg_loss = running_loss / self.log_every
                ppl      = math.exp(min(avg_loss, 20))
                lr_now   = self.scheduler.get_last_lr()[0]
                print(f"[step {self.global_step:>6d}/{self.total_steps}] "
                      f"loss={avg_loss:.4f}  ppl={ppl:.2f}  "
                      f"lr={lr_now:.2e}  "
                      f"win_reg={metrics['win_reg'].item():.4f}  "
                      f"time={elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            if self.val_loader and self.global_step % self.val_every == 0 and self._is_main_process():
                val_ppl = self._validate()
                print(f"  └─ val_ppl={val_ppl:.2f}")
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best.pt")
                self.model.train()

            if self.global_step % self.save_every == 0 and self._is_main_process():
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")

        if self._is_main_process():
            self._save_checkpoint("final.pt")
            print(f"Training complete. Best val ppl: {self.best_val_ppl:.2f}")

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.val_loader:
            metrics = self._forward_step(batch)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                labels = batch[1].to(self.device)
            else:
                b = batch[0] if isinstance(batch, (list, tuple)) else batch
                labels = b[:, 1:].to(self.device)
            n_toks = (labels != -100).sum().item()
            total_loss   += metrics["ce_loss"].item() * n_toks
            total_tokens += n_toks

        if self.ddp:
            tl = torch.tensor(total_loss,   device=self.device)
            tt = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tl.item(), tt.item()

        return math.exp(min(total_loss / max(total_tokens, 1), 20))

    def _save_checkpoint(self, filename):
        path = self.save_dir / filename
        model_state = self.model.module.state_dict() if self.ddp else self.model.state_dict()
        torch.save({"step": self.global_step, "model_state": model_state,
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_val_ppl": self.best_val_ppl}, path)
        print(f"  └─ saved checkpoint → {path}")

    def _load_checkpoint(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        m = self.model.module if self.ddp else self.model
        m.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        print(f"Resumed from {path} (step {self.global_step})")