import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
        model,
        train_loader,
        val_loader        = None,
        lr                = 3e-4,
        weight_decay      = 0.1,
        max_grad_norm     = 1.0,
        warmup_steps      = 500,
        total_steps       = 10_000,
        grad_accum        = 1,
        log_every         = 50,
        val_every         = 500,
        save_every        = 1000,
        save_dir          = "checkpoints",
        dtype             = torch.bfloat16,
        window_reg_coef   = 0.0,
        compute_metrics_fn= None,
        device            = None,
        resume_from       = None,
        ddp               = False,
    ):
        self.model              = model
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.max_grad_norm      = max_grad_norm
        self.total_steps        = total_steps
        self.grad_accum         = grad_accum
        self.log_every          = log_every
        self.val_every          = val_every
        self.save_every         = save_every
        self.save_dir           = Path(save_dir)
        self.dtype              = dtype
        self.window_reg_coef    = window_reg_coef
        self.compute_metrics_fn = compute_metrics_fn
        self.ddp                = ddp

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model  = self.model.to(device)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device] if self.device.type == "cuda" else None)

        decay, no_decay, dsalt = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ("window_pred", "alpha_w")):
                dsalt.append(p)
            elif any(kw in name for kw in ("norm", "bias", "emb")):
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = AdamW(
            [
                {"params": decay,    "lr": lr,        "weight_decay": weight_decay},
                {"params": no_decay, "lr": lr,        "weight_decay": 0.0},
                {"params": dsalt,    "lr": lr * 2.0,  "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95), eps=1e-8,
        )
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        self.use_amp   = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        self.scaler    = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

        self.global_step  = 0
        self.best_val_ppl = float("inf")
        self.history      = {
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

    def _is_main(self):
        return not self.ddp or dist.get_rank() == 0

    def _forward(self, x, y):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            logits, cont_windows = self.model(x, return_windows=self.window_reg_coef > 0)
            B, N, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.view(B * N, V), y.contiguous().view(B * N), ignore_index=-100,
            )
            win_reg = torch.tensor(0.0, device=self.device)
            if self.window_reg_coef > 0 and cont_windows:
                for cw in cont_windows:
                    win_reg = win_reg + (-cw.var(dim=-1).mean())
                win_reg = win_reg / len(cont_windows)
            return ce + self.window_reg_coef * win_reg, ce.detach(), win_reg.detach()

    def train(self):
        self.model.train()
        data_iter    = iter(self.train_loader)
        loss_accum   = 0.0
        t0           = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while self.global_step < self.total_steps:
            for acc_step in range(self.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)
                loss, ce, win_reg = self._forward(x, y)
                loss = loss / self.grad_accum
                if self.dtype == torch.float16 and self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += ce.item()

            if self.dtype == torch.float16 and self.use_amp:
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
                avg_loss = loss_accum / self.log_every
                ppl      = math.exp(min(avg_loss, 20))
                lr_now   = self.scheduler.get_last_lr()[0]
                mem_gb   = torch.cuda.memory_allocated(self.device) / 1e9
                log_str  = (
                    f"step {self.global_step:>6d}/{self.total_steps} │ "
                    f"loss {avg_loss:.4f} │ ppl {ppl:.2f} │ "
                    f"lr {lr_now:.2e} │ win_reg {win_reg.item():.4f} │ "
                    f"mem {mem_gb:.2f}GB │ {self.log_every/elapsed:.2f}it/s"
                )

                if self.compute_metrics_fn is not None:
                    m = self.compute_metrics_fn(self.model, x[:1])
                    log_str += (
                        f"\n        σ₂ {m['sigma2']:.4f} │ rank {m['eff_rank']:.1f} │ "
                        f"res {m['res_norm']:.4f} │ H {m['attn_entropy']:.4f} │ "
                        f"noise {m['noise_norm']:.4f} │ sink {m['attn_sink']:.4f} │ "
                        f"head_std {m['head_spec_std']:.4f} │ dist {m['token_dist']:.4f}"
                    )
                    for k in ["sigma2", "eff_rank", "res_norm", "attn_entropy", "noise_norm",
                              "head_spec_std", "attn_sink", "token_dist", "sigma2_per_layer",
                              "entropy_per_layer", "noise_per_layer", "eff_rank_per_layer",
                              "res_per_layer", "token_dist_per_layer", "alpha_per_head",
                              "oow_mass_per_layer"]:
                        self.history[k].append(m[k])

                print(log_str)
                self.history["train_loss"].append(avg_loss)
                self.history["step_time"].append(elapsed / self.log_every)
                self.history["gpu_mem_gb"].append(mem_gb)
                self.history["lr"].append(lr_now)
                loss_accum = 0.0
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
    def _validate(self):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, ce, _ = self._forward(x, y)
            n_toks       = y.numel()
            total_loss   += ce.item() * n_toks
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
        m    = self.model.module if self.ddp else self.model
        torch.save({
            "step":            self.global_step,
            "model_state":     m.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_ppl":    self.best_val_ppl,
            "history":         self.history,
        }, path)
        print(f"  ╰─ checkpoint → {path}")

    def _load_checkpoint(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        m    = self.model.module if self.ddp else self.model
        m.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step  = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        self.history      = ckpt.get("history", self.history)
        print(f"Resumed from {path} at step {self.global_step}")