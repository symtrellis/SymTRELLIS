from dataclasses import dataclass

from torch.utils.data import DataLoader

from dataset import DirectFileLoadDataset, MultiScaleMixedPairSampler, build_catalog, pair_collate
from trainer.config import TrainConfig


@dataclass
class TrainingData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_sampler: MultiScaleMixedPairSampler


def create_data_loaders(config: TrainConfig, rank: int, world_size: int) -> TrainingData:
    # Keep the old script's archive-level train/eval split and pair sampler semantics intact.
    train_shape_paths = build_catalog(
        [config.train_data_dir],
        seed=config.train_catalog_seed,
    )
    val_shape_paths = build_catalog(
        [config.eval_data_dir],
        seed=config.val_seed,
        chunk=config.val_num_chunk,
        chunk_id=config.val_chunk_id,
    )
    test_shape_paths = build_catalog(
        [config.eval_data_dir],
        seed=config.test_seed,
        chunk=config.test_num_chunk,
        chunk_id=config.test_chunk_id,
    )

    dataset_train = DirectFileLoadDataset(
        shape_paths=train_shape_paths,
        grid_size=config.grid_size,
        num_scale=config.num_scale,
        num_rots=config.num_rots,
        num_perts=config.num_perts,
    )
    dataset_val = DirectFileLoadDataset(
        shape_paths=val_shape_paths,
        grid_size=config.grid_size,
        num_scale=config.num_scale,
        num_rots=config.num_rots,
        num_perts=config.num_perts,
    )
    dataset_test = DirectFileLoadDataset(
        shape_paths=test_shape_paths,
        grid_size=config.grid_size,
        num_scale=config.num_scale,
        num_rots=config.num_rots,
        num_perts=config.num_perts,
    )

    sampler_train = MultiScaleMixedPairSampler(
        num_shapes=dataset_train.num_shapes,
        num_scale=dataset_train.num_scale,
        num_rots=dataset_train.num_rots,
        num_perts=dataset_train.num_perts,
        batch_size=config.batch_size,
        num_batch_per_epoch=config.steps_per_epoch * config.accumulation_steps,
        rank=rank,
        world_size=world_size,
        seed=config.seed,
        same_rot_diff_pert_ratio=config.same_rot_diff_pert_ratio,
    )
    sampler_val = MultiScaleMixedPairSampler(
        num_shapes=dataset_val.num_shapes,
        num_scale=dataset_val.num_scale,
        num_rots=dataset_val.num_rots,
        num_perts=dataset_val.num_perts,
        batch_size=config.batch_size,
        num_batch_per_epoch=config.max_eval_batches,
        rank=rank,
        world_size=world_size,
        seed=config.val_seed,
        same_rot_diff_pert_ratio=config.same_rot_diff_pert_ratio,
    )
    sampler_test = MultiScaleMixedPairSampler(
        num_shapes=dataset_test.num_shapes,
        num_scale=dataset_test.num_scale,
        num_rots=dataset_test.num_rots,
        num_perts=dataset_test.num_perts,
        batch_size=config.batch_size,
        num_batch_per_epoch=config.max_eval_batches,
        rank=rank,
        world_size=world_size,
        seed=config.test_seed,
        same_rot_diff_pert_ratio=config.same_rot_diff_pert_ratio,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=pair_collate,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_sampler=sampler_val,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=pair_collate,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_sampler=sampler_test,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=pair_collate,
    )

    return TrainingData(
        train_loader=loader_train,
        val_loader=loader_val,
        test_loader=loader_test,
        train_sampler=sampler_train,
    )
