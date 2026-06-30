import argparse
from dataclasses import dataclass, fields
from typing import Literal, TypeVar, cast, get_args, get_origin

from symtrellis.mapper.config import Swin3DLatentMapperScale


@dataclass
class TrainConfig:
    train_data_dir: str = ""
    eval_data_dir: str = ""

    grid_size: int = 16
    latent_dim: int = 8
    lowrank_rank: int = 8
    num_scale: int = 1
    num_rots: int = 16
    num_perts: int = 4
    same_rot_diff_pert_ratio: float = 0.2

    val_seed: int = 42
    val_chunk_id: int = 0
    val_num_chunk: int = 2
    test_seed: int = 42
    test_chunk_id: int = 1
    test_num_chunk: int = 2

    model_backend: str = "swin3d"
    model_scale: Swin3DLatentMapperScale = "small"
    attention_backend: str = "xformers"
    initial_checkpoint_path: str = ""

    batch_size: int = 16
    steps_per_epoch: int = 250
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-2
    accumulation_steps: int = 16
    amp: bool = False
    max_grad_norm: float = 0.0

    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False

    max_eval_batches: int = 100

    log_interval: int = 50
    log_dir: str = ""
    checkpoint_dir: str = ""
    resume_checkpoint_path: str = ""

    seed: int = 114514
    train_catalog_seed: int = 123456
    train_coord_shift_range: int = 512 # aumentation parameter so that the mapper extrapolate to larger range of grid
    coord_shift_seed: int = 1919810


ConfigT = TypeVar("ConfigT", bound=TrainConfig)


def add_config_arguments(
    parser: argparse.ArgumentParser,
    config_type: type[ConfigT],
) -> None:
    config_defaults = config_type()
    for config_field in fields(config_type):
        name = config_field.name
        default = getattr(config_defaults, name)
        argument_name = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            action = "store_false" if default else "store_true"
            parser.add_argument(argument_name, action=action, default=default, help=f"default: {default}")
            continue

        field_type = config_field.type
        if get_origin(field_type) is Literal:
            parser.add_argument(
                argument_name,
                type=type(default),
                choices=get_args(field_type),
                default=default,
                help=f"default: {default}",
            )
        else:
            parser.add_argument(argument_name, type=type(default), default=default, help=f"default: {default}")


def parse_config(args: argparse.Namespace, config_type: type[ConfigT]) -> ConfigT:
    values = {config_field.name: getattr(args, config_field.name) for config_field in fields(config_type)}
    return cast(ConfigT, config_type(**values))
