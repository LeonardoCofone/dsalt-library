import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing as mp
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
    """
    Trainer per DSALT con supporto automatico a CPU, 1 GPU e multi-GPU (DDP).

    Args:
        model:                  Il modello nn.Module da addestrare.
        train_loader:           DataLoader per il training set.
        val_loader:             DataLoader per il validation set.
        rank:                   Rank del processo corrente (0 per single-GPU/CPU,
                                impostato da mp.spawn in DDP).
        world_size:             Numero totale di processi / GPU (1 = single device).
        lr:                     Learning rate iniziale per AdamW.
        weight_decay:           Weight decay applicato ai parametri >= 2D
                                (esclusi norm, bias, embedding).
        max_grad_norm:          Soglia per gradient clipping (0 = disabilitato).
        warmup_steps:           Step di warmup lineare del learning rate.
        total_steps:            Numero totale di step di ottimizzazione.
        grad_accum:             Numero di mini-batch da accumulare prima di ogni
                                aggiornamento dei pesi (simula batch size maggiori).
        log_every:              Frequenza (in step) del logging su console.
        val_every:              Frequenza (in step) della valutazione su val set.
        save_every:             Frequenza (in step) del salvataggio periodico.
        save_dir:               Directory dove salvare checkpoint e log.
        mixed_precision:        'bf16', 'fp16' o 'no'. bf16 è consigliato su
                                Ampere+; fp16 su GPU più vecchie; 'no' per CPU.
        window_reg_coef:        Coefficiente della regolarizzazione L2 sul proiettore
                                della finestra adattiva (window_proj). Valori tipici:
                                1e-3 – 1e-2.
        gradient_checkpointing: Se True, ricomputa le attivazioni durante il backward
                                per ridurre la memoria GPU (più lento ma < VRAM).
        compile_model:          Se True, usa torch.compile() (richiede PyTorch >= 2.0
                                e una GPU compatibile). Accelera il training dopo
                                il primo step di warmup del compilatore.
        ddp_backend:            Backend per DDP: 'nccl' (GPU, raccomandato),
                                'gloo' (CPU o fallback), 'mpi' se disponibile.
        seed:                   Seed per riproducibilità (applicato su tutti i processi).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        rank: int = 0,
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
        window_reg_coef: float = 0.01,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        ddp_backend: str = "nccl",
        seed: int = 42,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)
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
        self.ddp_backend = ddp_backend
        self.seed = seed

        torch.manual_seed(seed + rank)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("dsalt.trainer", log_dir=str(self.save_dir))

        self.device = get_device(rank)

        if self.world_size > 1:
            setup_ddp(rank, world_size, backend=ddp_backend)

        self._amp_dtype = self._resolve_amp_dtype(mixed_precision)
        self._use_amp = self._amp_dtype is not None
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if self._use_amp and self._amp_dtype == torch.float16
            else None
        )

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        model = model.to(self.device)

        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.model = model

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            if self.is_main:
                self.logger.info("torch.compile() abilitato")

        self.use_triton = self.device.type == "cuda"

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer)

        self.global_step = 0
        self.best_val_ppl = float("inf")
        self._timer = StepTimer(window=50)
        self._tokens_per_batch: int = 0

    def _resolve_amp_dtype(self, mixed_precision: str) -> torch.dtype | None:
        if mixed_precision == "bf16" and self.device.type == "cuda":
            return torch.bfloat16
        if mixed_precision == "fp16" and self.device.type == "cuda":
            return torch.float16
        return None

    def _unwrap(self) -> nn.Module:
        if isinstance(self.model, DDP):
            return self.model.module
        if hasattr(self.model, "_orig_mod"):
            inner = self.model._orig_mod
            return inner.module if isinstance(inner, DDP) else inner
        return self.model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        base = self._unwrap()
        decay, nodecay = [], []
        for name, p in base.named_parameters():
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
        reg = torch.tensor(0.0, device=self.device)
        for layer in self._unwrap().layers:
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
        return (
            ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
        )

    def _forward_step(self, batch) -> torch.Tensor:
        ids, labels = self._extract_batch(batch)
        self._tokens_per_batch = ids.numel()

        ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._use_amp
            else torch.no_grad.__class__.__new__(torch.no_grad)
        )

        if self._use_amp:
            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype):
                out = self.model(
                    ids,
                    labels=labels,
                    use_triton=self.use_triton,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
                loss = out["loss"] + self._window_reg_loss()
        else:
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

        if self.world_size > 1:
            t = torch.tensor([total_loss, total_tokens], device=self.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            total_loss, total_tokens = t[0].item(), t[1].item()

        avg_loss = total_loss / max(1, total_tokens)
        self.model.train()
        return math.exp(min(avg_loss, 20.0))

    def _save_checkpoint(self, tag: str):
        if not self.is_main:
            return
        base = self._unwrap()
        ckpt = {
            "step": self.global_step,
            "model_state_dict": base.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
        }
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        self.logger.info(f"checkpoint salvato → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap().load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        if self.is_main:
            self.logger.info(f"ripreso dallo step {self.global_step}")

    def _log_step(self, accum_loss: float) -> None:
        if not self.is_main:
            return

        it_s = self._timer.avg_it_s
        tok_s = int(it_s * self._tokens_per_batch * self.grad_accum)
        lr_now = self.scheduler.get_last_lr()[0]
        ppl = math.exp(min(accum_loss, 20.0))

        mem_str = ""
        eff_rank_str = ""

        if self.device.type == "cuda":
            stats = get_gpu_memory_stats(self.device)
            mem_str = (
                f" | mem={stats.get('allocated_gb', 0):.1f}/"
                f"{stats.get('total_gb', 0):.1f}GB"
                f"({stats.get('utilization_pct', 0):.0f}%)"
            )

        base = self._unwrap()
        if hasattr(base, "layers") and len(base.layers) > 0:
            layer = base.layers[0]
            if hasattr(layer, "attn") and hasattr(layer.attn, "window_proj"):
                eff_r = _effective_rank(layer.attn.window_proj.weight)
                eff_rank_str = f" | eff_rank={eff_r:.1f}"

        mode = "CPU" if self.device.type == "cpu" else (f"DDP×{self.world_size}" if self.world_size > 1 else "GPU")

        self.logger.info(
            f"[{mode}] step={self.global_step:6d}/{self.total_steps}"
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

        if self.is_main:
            n_params = sum(p.numel() for p in self._unwrap().parameters() if p.requires_grad)
            mode = (
                "CPU"
                if self.device.type == "cpu"
                else (f"DDP×{self.world_size} (backend={self.ddp_backend})" if self.world_size > 1 else "1×GPU")
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
                    batch = next(data_iter)

                loss = self._forward_step(batch) / self.grad_accum

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

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
            self._timer.stop()

            if self.global_step % self.log_every == 0:
                self._log_step(accum_loss)
                accum_loss = 0.0

            if self.global_step % self.val_every == 0:
                barrier(self.rank, self.world_size)
                val_ppl = self._validate()
                if self.is_main:
                    self.logger.info(
                        f"step={self.global_step:6d} | val_ppl={val_ppl:.4f}"
                        + (" ← best" if val_ppl < self.best_val_ppl else "")
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


def _ddp_worker(
    rank: int,
    world_size: int,
    model_fn,
    train_dataset,
    val_dataset,
    trainer_kwargs: dict,
):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    batch_size = trainer_kwargs.pop("batch_size", 8)
    num_workers = trainer_kwargs.pop("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = model_fn()

    trainer = DSALTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rank=rank,
        world_size=world_size,
        **trainer_kwargs,
    )
    trainer.train()


def launch_training(
    model_fn,
    train_dataset,
    val_dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    **trainer_kwargs,
):
    """
    Entry point unificato. Rileva automaticamente CPU / 1 GPU / multi-GPU
    e avvia il training con la strategia corretta.

    Args:
        model_fn:       Callable senza argomenti che restituisce il modello.
                        In DDP viene chiamata una volta per processo.
        train_dataset:  Dataset PyTorch per il training.
        val_dataset:    Dataset PyTorch per la validazione.
        batch_size:     Batch size per GPU/CPU.
        num_workers:    Worker per DataLoader.
        **trainer_kwargs: Tutti gli altri argomenti di DSALTTrainer.

    Esempi:
        launch_training(lambda: MyModel(), train_ds, val_ds,
                        lr=1e-4, total_steps=5000, mixed_precision="bf16")
    """
    world_size = torch.cuda.device_count()

    if world_size > 1:
        kwargs = dict(trainer_kwargs, batch_size=batch_size, num_workers=num_workers)
        mp.spawn(
            _ddp_worker,
            args=(world_size, model_fn, train_dataset, val_dataset, kwargs),
            nprocs=world_size,
            join=True,
        )
    else:
        device = torch.device("cuda:0") if world_size == 1 else torch.device("cpu")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        model = model_fn()
        trainer = DSALTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            rank=0,
            world_size=1,
            **trainer_kwargs,
        )
        trainer.train()