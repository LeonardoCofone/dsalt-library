import math
import time
import logging
import os
from pathlib import Path
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

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


def _ddp_worker(
    rank: int,
    world_size: int,
    trainer_kwargs: dict,
    model: nn.Module,
    return_dict,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = trainer_kwargs.pop("_master_port", "29500")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    train_dataset = trainer_kwargs.pop("_train_dataset")
    val_dataset   = trainer_kwargs.pop("_val_dataset", None)
    batch_size    = trainer_kwargs.pop("_batch_size")
    num_workers   = trainer_kwargs.pop("_num_workers", 0)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    _runner = _TrainerCore(
        model=model,
        model_base=model.module,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        is_ddp=True,
        **trainer_kwargs,
    )

    history = _runner.run()

    if rank == 0:
        return_dict["history"] = history

    dist.destroy_process_group()


class _TrainerCore:

    def __init__(
        self,
        model: nn.Module,
        model_base: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        rank: int,
        is_ddp: bool,
        lr: float,
        weight_decay: float,
        max_grad_norm: float,
        warmup_steps: int,
        total_steps: int,
        grad_accum: int,
        log_every: int,
        val_every: int,
        save_every: int,
        save_dir: Path,
        dtype: torch.dtype,
        window_reg_coef: float,
        compute_metrics_fn: Optional[Callable],
        resume_from: Optional[str],
        log_file: Optional[str],
    ):
        self.model        = model
        self.model_base   = model_base
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.rank         = rank
        self.is_ddp       = is_ddp
        self.is_main      = (rank == 0)

        if self.is_main:
            setup_logging(level=logging.INFO, log_file=log_file)

        decay, no_decay, dsalt_params = [], [], []
        for name, p in model_base.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in DSALT_PARAM_KEYWORDS):
                dsalt_params.append(p)
            elif any(kw in name for kw in NO_DECAY_KEYWORDS):
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
        self.scheduler     = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        self.dtype         = dtype
        self.use_amp       = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        self.scaler        = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and dtype == torch.float16
            else None
        )
        self.max_grad_norm      = max_grad_norm
        self.total_steps        = total_steps
        self.grad_accum         = grad_accum
        self.log_every          = log_every
        self.val_every          = val_every
        self.save_every         = save_every
        self.save_dir           = save_dir
        self.window_reg_coef    = window_reg_coef
        self.compute_metrics_fn = compute_metrics_fn
        self.global_step        = 0
        self.best_val_ppl       = float("inf")
        self._zero_tensor       = torch.tensor(0.0, device=device)
        self.history: Dict[str, list] = {
            "train_loss": [], "val_ppl": [], "val_steps": [],
            "step_time": [], "gpu_mem_gb": [], "lr": [],
            "sigma2": [], "eff_rank": [], "res_norm": [], "attn_entropy": [],
            "noise_norm": [], "head_spec_std": [], "attn_sink": [],
        }
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if resume_from:
            self._load_checkpoint(resume_from)

    def _forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            logits, cont_windows = self.model(x, return_window=self.window_reg_coef > 0)
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
        return ce + self.window_reg_coef * win_reg, ce.detach(), win_reg.detach()

    def run(self) -> Dict[str, list]:
        self.model.train()
        if self.is_main:
            logger.info("=" * 60)
            logger.info(f"Training starting from step {self.global_step}")
            logger.info("=" * 60)

        data_iter           = iter(self.train_loader)
        loss_accum          = 0.0
        n_steps_accumulated = 0
        t0                  = time.time()
        t_train_start       = time.time()
        _first_batch_logged = False

        while self.global_step < self.total_steps:
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(self.global_step)

            for acc_step in range(self.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                if self.is_main and not _first_batch_logged:
                    logger.info(f"First batch  →  x={tuple(x.shape)}  y={tuple(y.shape)}  x.dtype={x.dtype}")
                    logger.info("Running first forward pass (Triton JIT compile on first call, may take ~10s) ...")
                    _first_batch_logged = True

                loss, ce, win_reg = self._forward(x, y)

                if self.is_main and self.global_step == 0 and acc_step == 0:
                    logger.info(f"First forward done  →  ce={ce.item():.4f}")

                loss = loss / self.grad_accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.is_main and self.global_step == 0 and acc_step == 0:
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

            if self.is_main and self.global_step == 1:
                logger.info(f"Step 1 complete  →  total elapsed {time.time() - t_train_start:.1f}s")

            if self.is_main and self.global_step % self.log_every == 0:
                elapsed      = time.time() - t0
                avg_loss     = loss_accum / n_steps_accumulated
                ppl          = math.exp(min(avg_loss, 1000))
                lr_now       = self.scheduler.get_last_lr()[0]
                it_per_sec   = self.log_every / elapsed if elapsed > 0 else float("inf")
                mem_gb       = torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
                mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9  if self.device.type == "cuda" else 0.0

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
                    m = self.compute_metrics_fn(self.model_base, x[:1])
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
                if self.is_main:
                    logger.info(f"Running validation at step {self.global_step} ...")
                val_ppl = self._validate()
                if self.is_main:
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

            if self.is_main and self.global_step % self.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")

        if self.is_main:
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

        if self.is_ddp:
            stats = torch.tensor([total_loss, float(total_tokens)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = stats[0].item(), stats[1].item()

        return math.exp(min(total_loss / max(total_tokens, 1), 20))

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "step":            self.global_step,
                "model_state":     self.model_base.state_dict(),
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
            self.model_base.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.global_step  = ckpt["step"]
            self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
            self.history      = ckpt.get("history", self.history)
            logger.info(f"Resumed from {path}  →  step={self.global_step}  best_val_ppl={self.best_val_ppl:.2f}")
        except (FileNotFoundError, IOError, RuntimeError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


class DSALTTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader]       = None,
        lr: float                              = 3e-4,
        weight_decay: float                    = 0.1,
        max_grad_norm: float                   = 1.0,
        warmup_steps: int                      = 500,
        total_steps: int                       = 10_000,
        grad_accum: int                        = 1,
        log_every: int                         = 50,
        val_every: int                         = 500,
        save_every: int                        = 1000,
        save_dir: str                          = "checkpoints",
        dtype: torch.dtype                     = torch.bfloat16,
        window_reg_coef: float                 = 0.0,
        compute_metrics_fn: Optional[Callable] = None,
        resume_from: Optional[str]             = None,
        gradient_checkpointing: bool           = False,
        device: str                            = "cpu",
        num_gpus: int                          = 1,
        log_file: Optional[str]                = None,
        master_port: str                       = "29500",
    ):
        setup_logging(level=logging.INFO, log_file=log_file)
        logger.info("=" * 60)
        logger.info("DSALT Trainer init starting")
        logger.info("=" * 60)

        self.primary_device, self.gpu_ids = resolve_device(device, num_gpus)
        self.use_ddp = len(self.gpu_ids) > 1
        logger.info(f"Device resolved  →  primary={self.primary_device}  gpu_ids={self.gpu_ids}  DDP={self.use_ddp}")

        if self.primary_device.type == "cuda":
            logger.info(print_gpu_info())

        if gradient_checkpointing:
            self._enable_gradient_checkpointing(model)
            logger.info("Gradient checkpointing enabled")

        if self.use_ddp:
            logger.info(f"Multi-GPU mode  →  DDP on GPUs {self.gpu_ids}")
        elif len(self.gpu_ids) == 1:
            logger.info(f"Single GPU mode  →  {self.primary_device}")
        else:
            logger.info("CPU mode")

        self._model         = model
        self._train_loader  = train_loader
        self._val_loader    = val_loader
        self._core_kwargs   = dict(
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            grad_accum=grad_accum,
            log_every=log_every,
            val_every=val_every,
            save_every=save_every,
            save_dir=Path(save_dir),
            dtype=dtype,
            window_reg_coef=window_reg_coef,
            compute_metrics_fn=compute_metrics_fn,
            resume_from=resume_from,
            log_file=log_file,
        )
        self._master_port   = master_port
        self._save_dir      = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {}

        logger.info("=" * 60)
        logger.info("Trainer ready. Call trainer.train() to start training.")
        logger.info("=" * 60)

    @staticmethod
    def _enable_gradient_checkpointing(model: nn.Module) -> None:
        try:
            from dsalt.modules.dsalt_attention import DSALTAttention
            for module in model.modules():
                if isinstance(module, DSALTAttention):
                    module.gradient_checkpointing = True
        except ImportError:
            logger.warning("Could not import DSALTAttention for gradient checkpointing")

    def train(self) -> Dict[str, list]:
        if not self.use_ddp:
            device = self.primary_device
            model  = self._model.to(device)

            train_loader = self._train_loader
            val_loader   = self._val_loader

            core = _TrainerCore(
                model=model,
                model_base=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                rank=0,
                is_ddp=False,
                **self._core_kwargs,
            )
            self.history = core.run()
            return self.history

        world_size   = len(self.gpu_ids)
        train_dataset = self._train_loader.dataset
        val_dataset   = self._val_loader.dataset if self._val_loader is not None else None
        batch_size    = self._train_loader.batch_size
        num_workers   = self._train_loader.num_workers

        self._model.share_memory()

        manager     = mp.Manager()
        return_dict = manager.dict()

        worker_kwargs = dict(**self._core_kwargs)
        worker_kwargs["_train_dataset"] = train_dataset
        worker_kwargs["_val_dataset"]   = val_dataset
        worker_kwargs["_batch_size"]    = batch_size
        worker_kwargs["_num_workers"]   = num_workers
        worker_kwargs["_master_port"]   = self._master_port

        mp.spawn(
            _ddp_worker,
            args=(world_size, worker_kwargs, self._model, return_dict),
            nprocs=world_size,
            join=True,
        )

        self.history = return_dict.get("history", {})
        return self.history