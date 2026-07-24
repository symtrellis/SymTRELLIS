import os
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Any, Generic, TypeVar, cast

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import Prefetcher
from symtrellis.geometry import t_abs2grid
from symtrellis.mapper import (
    NeighborGraphLatentMapper,
    NeighborGraphLatentMapperScale,
    Swin3DLatentMapper,
    neighbor_graph_latent_mapper_config,
    swin_3d_latent_mapper_config,
)
from trainer.config import TrainConfig
from trainer.data import TrainingData, create_data_loaders
from trainer.metrics import compute_feature_metrics

ConfigT = TypeVar("ConfigT", bound=TrainConfig)


class BaseTask(Generic[ConfigT]):
    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.amp_dtype = torch.bfloat16
        self.amp_enabled = False

    def setup(self, device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool) -> None:
        self.device = device
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

    def feature_mask(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(target.shape[0], dtype=torch.bool, device=target.device)

    def extra_loss_and_metrics(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return prediction.new_zeros(()), {}, {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return


class Trainer:
    def __init__(self, config: TrainConfig, task: BaseTask[Any], task_name: str) -> None:
        self.config = config
        self.task = task
        self.task_name = task_name

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.amp_dtype = torch.bfloat16
        self.process_group_initialized = False

        self.data: TrainingData
        self.model: torch.nn.Module
        self.ddp_model: DDP
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None
        self.scaler: GradScaler
        self.writer: SummaryWriter | None = None
        self.coord_shift_generator = torch.Generator()
        self.coord_shift_generator.manual_seed(self.config.coord_shift_seed)

        self.start_epoch = 0
        self.global_step = 0
        self.update_step = 0
        self.start_alloc = 0
        self.start_reserved = 0

    def setup(self) -> None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.coord_shift_generator.manual_seed(self.config.coord_shift_seed + self.rank)
        dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        self.process_group_initialized = True

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        self.data = create_data_loaders(self.config, self.rank, self.world_size)
        self.model = self.create_model()
        self.load_initial_checkpoint()
        self.model.to(self.device)
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
        )

        self.task.setup(self.device, self.amp_dtype, self.config.amp)

        self.optimizer = torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = GradScaler(enabled=self.config.amp)
        self.writer = SummaryWriter(self.config.log_dir) if self.rank == 0 and self.config.log_dir else None

        self.scheduler = None
        if self.config.lr_scheduler == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
            )

        self.load_resume_checkpoint()
        self.print_memory_start()

    def micro_batches_per_epoch(self) -> int:
        # steps_per_epoch counts optimizer updates; each update consumes accumulated micro-batches.
        return self.config.steps_per_epoch * self.config.accumulation_steps

    def create_model(self) -> torch.nn.Module:
        if self.config.model_backend == "neighbor_graph":
            model_config = neighbor_graph_latent_mapper_config(
                scale=cast(NeighborGraphLatentMapperScale, self.config.model_scale),
                latent_dim=self.config.latent_dim,
                lowrank_rank=self.config.lowrank_rank,
            )
            return NeighborGraphLatentMapper(model_config)

        if self.config.model_backend == "swin3d":
            model_config = swin_3d_latent_mapper_config(
                scale=self.config.model_scale,
                latent_dim=self.config.latent_dim,
                lowrank_rank=self.config.lowrank_rank,
            )
            model_config = replace(model_config, attn_backend=self.config.attention_backend)
            return Swin3DLatentMapper(model_config)

        raise ValueError(f"Unknown model backend: {self.config.model_backend}")

    def load_initial_checkpoint(self) -> None:
        if not self.config.initial_checkpoint_path:
            return

        checkpoint = torch.load(self.config.initial_checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

    def load_resume_checkpoint(self) -> None:
        if not self.config.resume_checkpoint_path:
            return

        checkpoint = torch.load(self.config.resume_checkpoint_path, map_location="cpu")
        self.ddp_model.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.update_step = int(
            checkpoint.get(
                "update_step",
                (self.global_step + self.config.accumulation_steps - 1) // self.config.accumulation_steps,
            )
        )
        if "task" in checkpoint:
            self.task.load_state_dict(checkpoint["task"])

    def run(self) -> None:
        if self.rank == 0:
            print("=" * 50)
            print("Evaluating before training")
            print("=" * 50)
        self.evaluate_split(self.data.val_loader, "val")
        self.evaluate_split(self.data.test_loader, "test")
        dist.barrier()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch: int) -> None:
        micro_batches_per_epoch = self.micro_batches_per_epoch()
        self.data.train_sampler.set_iter_count(epoch * micro_batches_per_epoch)
        self.ddp_model.train()
        prefetcher = Prefetcher(self.data.train_loader, self.device)

        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        accumulated_metric_sums: dict[str, torch.Tensor] = {}
        accumulated_metric_counts: dict[str, torch.Tensor] = {}

        for batch in prefetcher:
            if micro_step >= micro_batches_per_epoch:
                break

            should_sync = ((micro_step + 1) % self.config.accumulation_steps) == 0
            sync_context = nullcontext() if should_sync else self.ddp_model.no_sync()
            with sync_context:
                total_loss, metrics, metric_validity = self.compute_batch(batch)
                loss = total_loss / self.config.accumulation_steps
                self.scaler.scale(loss).backward()

            # Log metrics at the same cadence as optimizer updates: average within
            # the local valid accumulation window first, then reduce across DDP ranks.
            for name, value in metrics.items():
                validity = metric_validity[name].detach().float()
                if name in accumulated_metric_sums:
                    accumulated_metric_sums[name] += value.detach().float() * validity
                    accumulated_metric_counts[name] += validity
                else:
                    accumulated_metric_sums[name] = value.detach().float() * validity
                    accumulated_metric_counts[name] = validity

            self.global_step += 1
            micro_step += 1
            if not should_sync:
                continue

            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.update_step += 1

            metrics_log = self.reduce_metrics(accumulated_metric_sums, accumulated_metric_counts)
            accumulated_metric_sums = {}
            accumulated_metric_counts = {}

            if self.rank == 0 and self.update_step % self.config.log_interval == 0:
                metric_text = " ".join(f"{name} {value:.6f}" for name, value in metrics_log.items())
                print(f"epoch {epoch} global_step {self.global_step} update_step {self.update_step} {metric_text}")
                if self.writer is not None:
                    for name, value in metrics_log.items():
                        self.writer.add_scalar(f"train/{name}", value, self.update_step)
        prefetcher.close()

        dist.barrier()
        if self.rank == 0:
            print(f"\n[Epoch {epoch}] Evaluating splits...")
        self.evaluate_split(self.data.val_loader, "val")
        self.evaluate_split(self.data.test_loader, "test")
        dist.barrier()

        self.save_checkpoint(epoch)

    def compute_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.config.amp):
            # Batch contract: pair features plus a destination-to-source transform
            # produce a coefficient operator, which predicts destination features.
            # The dataset stores normalized-position translations; mapper modules consume grid-index translations.
            t_grid = t_abs2grid(
                t_abs=batch["t_dst2src"],
                O=batch["O_dst2src"],
                grid_size=self.config.grid_size,
            )
            coords_src = batch["coords_src"]
            coords_dst = batch["coords_dst"]
            use_coord_shift = self.ddp_model.training and self.config.train_coord_shift_range > 0

            if use_coord_shift:
                batch_size = batch["O_dst2src"].shape[0]
                shift_low = -self.config.train_coord_shift_range
                shift_high = self.config.train_coord_shift_range + 1
                shift_src = torch.randint(
                    shift_low,
                    shift_high,
                    (batch_size, 3),
                    generator=self.coord_shift_generator,
                    dtype=torch.int64,
                ).to(device=self.device)
                shift_dst = torch.randint(
                    shift_low,
                    shift_high,
                    (batch_size, 3),
                    generator=self.coord_shift_generator,
                    dtype=torch.int64,
                ).to(device=self.device)
                src_coord_shift = shift_src[coords_src[:, 0].long()].to(dtype=coords_src.dtype)
                dst_coord_shift = shift_dst[coords_dst[:, 0].long()].to(dtype=coords_dst.dtype)

                # Train-only coordinate-frame translation. Restore batch coords before losses/decoders.
                coords_src[:, 1:] += src_coord_shift
                coords_dst[:, 1:] += dst_coord_shift
                shift_src_float = shift_src.to(dtype=t_grid.dtype)
                shift_dst_float = shift_dst.to(dtype=t_grid.dtype)
                dst_shift_in_src = torch.bmm(batch["O_dst2src"], shift_dst_float[..., None])[..., 0]
                t_grid = t_grid + shift_src_float - dst_shift_in_src

            coeff = self.ddp_model(
                coords_src=coords_src,
                coords_dst=coords_dst,
                O_dst2src=batch["O_dst2src"],
                t_dst2src=t_grid,
                s_dst2src=batch["s_dst2src"],
            )
            if use_coord_shift:
                coords_src[:, 1:] -= src_coord_shift
                coords_dst[:, 1:] -= dst_coord_shift
            prediction = coeff.apply(batch["feats_src"].to(dtype=coeff.dtype))

        target = batch["feats_dst"]
        mask = self.task.feature_mask(batch, prediction, target)
        feature_metrics = compute_feature_metrics(prediction, target, mask)
        extra_loss, extra_metrics, extra_metric_validity = self.task.extra_loss_and_metrics(
            batch,
            prediction,
            target,
            mask,
        )
        total_loss = feature_metrics["feature_l2"] + extra_loss

        metrics = {
            "loss_total": total_loss,
            "loss_feature_l2": feature_metrics["feature_l2"],
            **feature_metrics,
            **extra_metrics,
        }
        metric_validity = {name: value.new_ones((), dtype=torch.float32) for name, value in metrics.items()}
        metric_validity.update(extra_metric_validity)
        return total_loss, metrics, metric_validity

    def evaluate_split(self, loader: torch.utils.data.DataLoader, split_name: str) -> dict[str, float]:
        self.ddp_model.eval()
        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, torch.Tensor] = {}
        prefetcher = Prefetcher(loader, self.device)

        with torch.no_grad():
            for batch in prefetcher:
                _, metrics, metric_validity = self.compute_batch(batch)
                for name, value in metrics.items():
                    validity = metric_validity[name].detach().float()
                    if name in metric_sums:
                        metric_sums[name] += value.detach().float() * validity
                        metric_counts[name] += validity
                    else:
                        metric_sums[name] = value.detach().float() * validity
                        metric_counts[name] = validity
        prefetcher.close()

        # Each metric carries its own valid-batch count before reduction across ranks.
        metrics_log = self.reduce_metrics(metric_sums, metric_counts)
        if self.rank == 0:
            metric_text = " ".join(f"{name} {value:.6f}" for name, value in metrics_log.items())
            print(f"[{split_name}] {metric_text}")
            if self.writer is not None:
                for name, value in metrics_log.items():
                    self.writer.add_scalar(f"{split_name}/{name}", value, self.update_step)
        return metrics_log

    def reduce_metrics(
        self,
        metric_sums: dict[str, torch.Tensor],
        metric_counts: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        metric_names = list(metric_sums.keys())
        stats = torch.cat(
            [
                torch.stack([metric_sums[name] for name in metric_names]),
                torch.stack([metric_counts[name] for name in metric_names]),
            ]
        ).to(
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        num_metrics = len(metric_names)
        return {name: (stats[i] / stats[num_metrics + i].clamp_min(1.0)).item() for i, name in enumerate(metric_names)}

    def save_checkpoint(self, epoch: int) -> None:
        if self.rank != 0 or not self.config.checkpoint_dir:
            return

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(
            {
                "config": asdict(self.config),
                "task_name": self.task_name,
                "epoch": epoch,
                "model": self.ddp_model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "update_step": self.update_step,
                "task": self.task.state_dict(),
            },
            checkpoint_path,
        )

    def print_memory_start(self) -> None:
        if self.device.type == "cuda":
            dist.barrier()
            torch.cuda.synchronize(self.device)
            self.start_alloc = torch.cuda.memory_allocated(self.device)
            self.start_reserved = torch.cuda.memory_reserved(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            message = f"[rank {self.rank}][local_rank {self.local_rank}][{self.device}] " f"mem start: alloc={self.start_alloc / 1024**2:.1f} MiB " f"reserved={self.start_reserved / 1024**2:.1f} MiB"
        else:
            message = f"[rank {self.rank}][local_rank {self.local_rank}][{self.device}] cuda memory stats skipped: CUDA unavailable"

        for print_rank in range(self.world_size):
            dist.barrier()
            if self.rank == print_rank:
                print(message, flush=True)
        dist.barrier()

    def print_memory_end(self) -> None:
        if self.device.type == "cuda":
            dist.barrier()
            torch.cuda.synchronize(self.device)
            end_alloc = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            message = (
                f"[rank {self.rank}][local_rank {self.local_rank}][{self.device}] "
                f"mem end: alloc={end_alloc / 1024**2:.1f} MiB "
                f"reserved={end_reserved / 1024**2:.1f} MiB "
                f"peak_alloc={peak_alloc / 1024**2:.1f} MiB "
                f"peak_reserved={peak_reserved / 1024**2:.1f} MiB "
                f"peak_alloc_minus_start_alloc={(peak_alloc - self.start_alloc) / 1024**2:.1f} MiB"
            )
        else:
            message = f"[rank {self.rank}][local_rank {self.local_rank}][{self.device}] cuda memory stats skipped: CUDA unavailable"

        for print_rank in range(self.world_size):
            dist.barrier()
            if self.rank == print_rank:
                print(message, flush=True)
        dist.barrier()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.process_group_initialized:
            self.print_memory_end()
            dist.destroy_process_group()
            self.process_group_initialized = False
