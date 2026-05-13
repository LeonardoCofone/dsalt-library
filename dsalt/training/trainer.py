import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .gpu_auto import init_accelerator, get_gpu_memory_stats
from .logging_config import get_logger, StepTimer


def _effective_rank(matrix: torch.Tensor) -> float:
    with torch.no_grad():
        mat = matrix.float()
        if mat.ndim < 2:
            return 1.0
        try:
            sv = torch.linalg.svdvals(mat.view(mat.shape[0], -1))
            sv = sv / (sv.sum() + 1e-9)
            eff = torch.exp(-(sv * torch.log(sv + 1e-9)).sum())
            return eff.item()
        except Exception:
            return float("nan")


class DSALTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        window_reg_coef: float = 0.01,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        seed: int = 42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum = grad_accum
        self.log_every = log_every
        self.val_every = val_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.window_reg_coef = window_reg_coef
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model = compile_model

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("dsalt.trainer", log_dir=str(self.save_dir))

        self.accelerator = init_accelerator(
            mixed_precision=mixed_precision,
            grad_accum=grad_accum,
            log_dir=str(self.save_dir),
            seed=seed,
        )
        self.device = self.accelerator.device

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        self.model = model
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer)

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model,
            self.optimizer,
            train_loader,
            val_loader,
            self.scheduler,
        )

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            if self.accelerator.is_main_process:
                self.logger.info("torch.compile() enabled")

        self.use_triton = self.device.type == "cuda"
        self.global_step = 0
        self.best_val_ppl = float("inf")
        self._timer = StepTimer(window=50)
        self._tokens_per_batch: int = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, nodecay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(k in name for k in ("norm", "bias", "embed")):
                nodecay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.device.type == "cuda",
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _window_reg_loss(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        reg = torch.tensor(0.0, device=device)
        base = self.accelerator.unwrap_model(self.model)
        for layer in base.layers:
            reg = reg + layer.attn.window_proj.weight.pow(2).mean()
        return self.window_reg_coef * reg

    def _extract_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            ids, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            ids = batch["input_ids"]
            labels = batch.get("labels", ids)
        else:
            ids = batch
            labels = ids
        return ids.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def _forward_step(self, batch) -> torch.Tensor:
        ids, labels = self._extract_batch(batch)
        self._tokens_per_batch = ids.numel()

        with self.accelerator.autocast():
            out = self.model(
                ids,
                labels=labels,
                use_triton=self.use_triton,
                gradient_checkpointing=self.gradient_checkpointing,
            )
            loss = out["loss"] + self._window_reg_loss()
        return loss

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_loader:
            ids, labels = self._extract_batch(batch)
            out = self.model(
                ids,
                labels=labels,
                use_triton=self.use_triton,
                gradient_checkpointing=False,
            )
            n_tokens = (labels[:, 1:] != -100).sum().item()
            total_tokens += n_tokens
            total_loss += out["loss"].item() * n_tokens

        avg_loss = total_loss / max(1, total_tokens)
        self.model.train()
        return math.exp(min(avg_loss, 20.0))

    def _save_checkpoint(self, tag: str):
        base = self.accelerator.unwrap_model(self.model)
        ckpt = {
            "step": self.global_step,
            "model_state_dict": base.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
        }
        path = self.save_dir / f"checkpoint_{tag}.pt"
        self.accelerator.save(ckpt, path)
        if self.accelerator.is_main_process:
            self.logger.info(f"checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        base = self.accelerator.unwrap_model(self.model)
        base.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        if self.accelerator.is_main_process:
            self.logger.info(f"resumed from step {self.global_step}")

    def _log_step(self, accum_loss: float) -> None:
        if not self.accelerator.is_main_process:
            return

        it_s = self._timer.avg_it_s
        tok_s = int(it_s * self._tokens_per_batch * self.grad_accum)
        lr_now = self.scheduler.get_last_lr()[0]
        ppl = math.exp(min(accum_loss, 20.0))

        mem_str = ""
        eff_rank_str = ""

        if self.device.type == "cuda":
            stats = get_gpu_memory_stats(self.device)
            mem_str = f" | mem={stats.get('allocated_gb', 0):.1f}/{stats.get('total_gb', 0):.1f}GB({stats.get('utilization_pct', 0):.0f}%)"

        base = self.accelerator.unwrap_model(self.model)
        if hasattr(base, "layers") and len(base.layers) > 0:
            layer = base.layers[0]
            if hasattr(layer, "attn") and hasattr(layer.attn, "window_proj"):
                eff_r = _effective_rank(layer.attn.window_proj.weight)
                eff_rank_str = f" | eff_rank={eff_r:.1f}"

        self.logger.info(
            f"step={self.global_step:6d}/{self.total_steps}"
            f" | loss={accum_loss:.4f}"
            f" | ppl={ppl:.2f}"
            f" | lr={lr_now:.2e}"
            f" | {it_s:.1f}it/s"
            f" | {tok_s:,}tok/s"
            f"{mem_str}"
            f"{eff_rank_str}"
        )

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)
        accum_loss = 0.0

        if self.accelerator.is_main_process:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(
                f"training start | steps={self.total_steps} | "
                f"grad_accum={self.grad_accum} | "
                f"device={self.device} | "
                f"processes={self.accelerator.num_processes} | "
                f"triton={self.use_triton} | "
                f"gc={self.gradient_checkpointing} | "
                f"params={n_params:,}"
            )

        self._timer.start()

        while self.global_step < self.total_steps:
            with self.accelerator.accumulate(self.model):
                for _ in range(self.grad_accum):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    loss = self._forward_step(batch) / self.grad_accum
                    self.accelerator.backward(loss)
                    accum_loss += loss.item()

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            self._timer.stop()

            if self.global_step % self.log_every == 0:
                self._log_step(accum_loss)
                accum_loss = 0.0

            if self.global_step % self.val_every == 0:
                val_ppl = self._validate()
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"step={self.global_step:6d} | val_ppl={val_ppl:.4f}"
                        + (" ← best" if val_ppl < self.best_val_ppl else "")
                    )
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best")

            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            if self.global_step < self.total_steps:
                self._timer.start()
            else:
                break

        self._save_checkpoint("final")
        if self.accelerator.is_main_process:
            self.logger.info(f"done | best_val_ppl={self.best_val_ppl:.4f}")