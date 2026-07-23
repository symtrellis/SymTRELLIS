from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

import trellis2.models as trellis2_models
from trellis2.models.sparse_structure_vae import SparseStructureDecoder

from trainer.base import BaseTask
from trainer.config import TrainConfig

DEFAULT_DECODER_PRETRAINED_PATH = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"


@dataclass
class Config(TrainConfig):
    dst_input_norm_threshold: float = 1.5
    decoder_pretrained_path: str = DEFAULT_DECODER_PRETRAINED_PATH
    decoded_loss_weight: float = 0.002


class Task(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.decoder: SparseStructureDecoder | None = None

    def setup(self, device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool) -> None:
        super().setup(device, amp_dtype, amp_enabled)

        if self.config.decoded_loss_weight == 0.0:
            return

        decoder = cast(SparseStructureDecoder, trellis2_models.from_pretrained(self.config.decoder_pretrained_path))
        decoder.convert_to_fp32()
        decoder.float().eval().to(device)
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder = decoder

    def feature_mask(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.dst_input_norm_threshold >= 0:
            return target.float().norm(dim=1) >= self.config.dst_input_norm_threshold
        return torch.ones(target.shape[0], dtype=torch.bool, device=target.device)

    def extra_loss_and_metrics(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.config.decoded_loss_weight == 0.0:
            return prediction.new_zeros(()), {}, {}

        decoder = self.decoder
        assert decoder is not None

        batch_size = batch["O_dst2src"].shape[0]
        latent_dim = target.shape[1]
        grid_size = int(batch["grid_size"][0].item())

        dense_target = scatter_sparse_features(
            coords=batch["coords_dst"],
            feats=target,
            batch_size=batch_size,
            latent_dim=latent_dim,
            grid_size=grid_size,
        )
        dense_prediction = scatter_sparse_features_with_detached_mask(
            coords=batch["coords_dst"],
            feats=prediction,
            mask=mask,
            batch_size=batch_size,
            latent_dim=latent_dim,
            grid_size=grid_size,
        )

        # The target decoder pass is a fixed teacher; only predicted features receive gradients.
        with torch.no_grad():
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits_target = decoder(dense_target)

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            logits_prediction = decoder(dense_prediction)

        logits_target = logits_target.float()
        logits_prediction = logits_prediction.float()

        soft_target = logits_target.sigmoid()
        hard_target = (logits_target > 0).to(dtype=logits_prediction.dtype)

        decoded_l2 = F.mse_loss(logits_prediction, logits_target)
        decoder_soft_bce = F.binary_cross_entropy_with_logits(logits_prediction, soft_target)
        decoder_hard_bce = F.binary_cross_entropy_with_logits(logits_prediction, hard_target)

        occupied_target = logits_target > 0
        occupied_prediction = logits_prediction > 0
        intersection = (occupied_target & occupied_prediction).flatten(1).sum(dim=1).float()
        union = (occupied_target | occupied_prediction).flatten(1).sum(dim=1).float()
        decoder_iou = (intersection / union.clamp_min(1.0)).mean()

        extra_loss = self.config.decoded_loss_weight * decoded_l2
        return (
            extra_loss,
            {
                "loss_decoded_l2_weighted": extra_loss,
                "decoder_l2": decoded_l2,
                "decoder_soft_bce": decoder_soft_bce,
                "decoder_hard_bce": decoder_hard_bce,
                "decoder_iou": decoder_iou,
            },
            {},
        )


def scatter_sparse_features(
    coords: torch.Tensor,
    feats: torch.Tensor,
    batch_size: int,
    latent_dim: int,
    grid_size: int,
) -> torch.Tensor:
    dense = feats.new_zeros((batch_size, latent_dim, grid_size, grid_size, grid_size))
    batch_id = coords[:, 0].long()
    xyz = coords[:, 1:].long()
    dense[batch_id, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = feats
    return dense


def scatter_sparse_features_with_detached_mask(
    coords: torch.Tensor,
    feats: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    latent_dim: int,
    grid_size: int,
) -> torch.Tensor:
    dense = feats.new_zeros((batch_size, latent_dim, grid_size, grid_size, grid_size))

    # Kept rows receive decoder-loss gradients; dropped rows provide context only.
    coords_keep = coords[mask]
    feats_keep = feats[mask]
    batch_id_keep = coords_keep[:, 0].long()
    xyz_keep = coords_keep[:, 1:].long()
    dense[batch_id_keep, :, xyz_keep[:, 0], xyz_keep[:, 1], xyz_keep[:, 2]] = feats_keep

    coords_drop = coords[~mask]
    feats_drop = feats[~mask].detach()
    batch_id_drop = coords_drop[:, 0].long()
    xyz_drop = coords_drop[:, 1:].long()
    dense[batch_id_drop, :, xyz_drop[:, 0], xyz_drop[:, 1], xyz_drop[:, 2]] = feats_drop

    return dense
