import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .gpu_auto import init_accelerator, prepare_model_training
from .logging_config import get_logger


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
        device: str = "cuda",
        num_gpus: int = 1,
        dtype: torch.dtype = torch.float32,
        window_reg_coef: float = 0.01,
        gradient_checkpointing: bool = False,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.dtype = dtype
        self.window_reg_coef = window_reg_coef
        self.gradient_checkpointing = gradient_checkpointing

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("dsalt.trainer", log_dir=str(self.save_dir))

        self.accelerator = init_accelerator(
            mixed_precision="fp16",
            grad_accum=grad_accum,
        )

        self.device = self.accelerator.device

        self.model = model
        self.scheduler = self._build_scheduler()
        self.optimizer = self._build_optimizer()

        self.model, self.optimizer, self.train_loader, self.val_loader = prepare_model_training(
            self.accelerator,
            model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        )

        self.use_triton = self.device.type == "cuda"

        self.global_step = 0
        self.best_val_ppl = float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        nodecay_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in name or "bias" in name or "embed" in name:
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.lr, betas=(0.9, 0.95), eps=1e-8)

    def _build_scheduler(self):
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _window_regularization_loss(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        reg = torch.tensor(0.0, device=device)
        base_model = self.accelerator.unwrap_model(self.model)
        for layer in base_model.layers:
            w_proj = layer.attn.window_proj
            reg = reg + w_proj.weight.pow(2).mean()
        return self.window_reg_coef * reg

    def _forward_step(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            input_ids, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
        else:
            input_ids = batch
            labels = input_ids

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        with self.accelerator.autocast():
            out = self.model(
                input_ids,
                labels=labels,
                use_triton=self.use_triton,
                gradient_checkpointing=self.gradient_checkpointing,
            )
            loss = out["loss"]
            loss = loss + self._window_regularization_loss()

        return loss

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
            else:
                input_ids = batch.to(self.device)
                labels = input_ids

            out = self.model(
                input_ids,
                labels=labels,
                use_triton=self.use_triton,
                gradient_checkpointing=False,
            )
            n_tokens = (labels[:, 1:] != -100).sum().item()

            total_tokens += n_tokens
            total_loss += out["loss"].item() * n_tokens

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(avg_loss, 20.0))
        self.model.train()
        return ppl

    def _save_checkpoint(self, tag: str):
        base_model = self.accelerator.unwrap_model(self.model)
        ckpt = {
            "step": self.global_step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
        }
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        if self.accelerator.is_main_process:
            self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        base_model = self.accelerator.unwrap_model(self.model)
        base_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        if self.accelerator.is_main_process:
            self.logger.info(f"Resumed from step {self.global_step}")

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        data_iter = iter(self.train_loader)
        accum_loss = 0.0

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Starting training | total_steps={self.total_steps} | "
                f"grad_accum={self.grad_accum} | device={self.device} | "
                f"triton={self.use_triton} | gc={self.gradient_checkpointing}"
            )

        while self.global_step < self.total_steps:
            for micro_step in range(self.grad_accum):
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

            if self.global_step % self.log_every == 0:
                lr_now = self.scheduler.get_last_lr()[0]
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"step={self.global_step:6d} | "
                        f"loss={accum_loss:.4f} | "
                        f"ppl={math.exp(min(accum_loss, 20.0)):.2f} | "
                        f"lr={lr_now:.2e}"
                    )
                accum_loss = 0.0

            if self.global_step % self.val_every == 0:
                val_ppl = self._validate()
                if self.accelerator.is_main_process:
                    self.logger.info(f"step={self.global_step:6d} | val_ppl={val_ppl:.4f}")
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best")

            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            if self.global_step >= self.total_steps:
                break

        self._save_checkpoint("final")
        if self.accelerator.is_main_process:
           self.logger.info(f"Training complete | best_val_ppl={self.best_val_ppl:.4f}")