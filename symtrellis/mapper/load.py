from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import NeighborGraphLatentMapperConfig, Swin3DLatentMapperConfig
from .model import NeighborGraphLatentMapper, Swin3DLatentMapper

PRETRAINED_MAPPER_PATHS = {
    "trellis2_sparse_structure_neighbor_graph_pretrain": "trellis2/sparse_structure/neighbor_graph/pretrain",
    "trellis2_sparse_structure_neighbor_graph_finetune": "trellis2/sparse_structure/neighbor_graph/finetune",
    "trellis2_shape_neighbor_graph_pretrain": "trellis2/shape/neighbor_graph/pretrain",
}


def from_pretrained(
    model_id_or_path: str | Path,
    repo_id: str = "quantaji/SymTRELLIS",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
) -> Swin3DLatentMapper | NeighborGraphLatentMapper:
    """Load a released SymTRELLIS mapper from a local directory or HF repo path."""

    local_path = Path(model_id_or_path)
    if local_path.is_dir():
        config_path = local_path / "config.json"
        weights_path = local_path / "model.safetensors"
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Loading SymTRELLIS mapper weights from Hugging Face requires " "`huggingface_hub`. Install it with `pip install huggingface_hub safetensors`.") from exc

        model_id = str(model_id_or_path).strip("/")
        model_id = PRETRAINED_MAPPER_PATHS.get(model_id, model_id)
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_id}/config.json",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_id}/model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Loading SymTRELLIS mapper weights requires `safetensors`. " "Install it with `pip install safetensors`.") from exc

    config = json.loads(Path(config_path).read_text())
    if config["model_class"] == "Swin3DLatentMapper":
        model_config = Swin3DLatentMapperConfig(**config["model_config"])
        model = Swin3DLatentMapper(model_config)
    elif config["model_class"] == "NeighborGraphLatentMapper":
        model_config = NeighborGraphLatentMapperConfig(**config["model_config"])
        model = NeighborGraphLatentMapper(model_config)
    else:
        raise ValueError(f"Unknown mapper class: {config['model_class']}")

    state_dict = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)
