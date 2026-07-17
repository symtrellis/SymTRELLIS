import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS2_ROOT = REPO_ROOT / "third_party" / "trellis2"
# TRELLIS-specific imports stay in the task layer so base training remains package-local.
if str(TRELLIS2_ROOT) not in sys.path:
    sys.path.append(str(TRELLIS2_ROOT))

import trellis2.models as trellis2_models
from trellis2.modules.sparse import SparseTensor

from trainer.base import BaseTask
from trainer.config import TrainConfig

if TYPE_CHECKING:
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

DEFAULT_DECODER_PRETRAINED_PATH = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"


def decode_shape_features(
    decoder: "FlexiDualGridVaeDecoder",
    latent: SparseTensor,
    guide_subdivisions: list[SparseTensor] | None = None,
) -> tuple[SparseTensor, list[SparseTensor]]:
    h = decoder.from_latent(latent)
    h = h.type(decoder.dtype)

    subdivision_logits: list[SparseTensor] = []
    for stage_index, blocks in enumerate(decoder.blocks):
        for block_index, block in enumerate(blocks):
            is_subdivision_block = stage_index < len(decoder.blocks) - 1 and block_index == len(blocks) - 1
            if not is_subdivision_block:
                h = block(h)
                continue

            if guide_subdivisions is None:
                h, sub_logits = block(h)
                subdivision_logits.append(sub_logits)
                continue

            sub_logits = block.to_subdiv(h)
            guide_subdivision = guide_subdivisions[len(subdivision_logits)]
            guide_mask = guide_subdivision.replace(guide_subdivision.feats > 0)

            residual = h
            h = h.replace(block.norm1(h.feats))
            h = h.replace(F.silu(h.feats))
            h = block.conv1(h)
            h = block.updown(h, guide_mask)
            residual = block.updown(residual, guide_mask)
            h = h.replace(block.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
            h = block.conv2(h)
            h = h + block.skip_connection(residual)
            subdivision_logits.append(sub_logits)

    h = h.type(latent.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    # The 7 output channels contain vertex offsets, intersected logits, and quad weights.
    return decoder.output_layer(h), subdivision_logits


@dataclass
class Config(TrainConfig):
    decoder_pretrained_path: str = DEFAULT_DECODER_PRETRAINED_PATH
    decoded_output_weight: float = 0.0
    decoded_subdivision_weight: float = 0.0
    decoder_resolution: int = 512
    max_decoded_output_points: int = 2_500_000


class Task(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.decoder: "FlexiDualGridVaeDecoder | None" = None

    def setup(self, device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool) -> None:
        super().setup(device, amp_dtype, amp_enabled)

        if self.config.decoded_output_weight == 0.0 and self.config.decoded_subdivision_weight == 0.0:
            return

        decoder = cast("FlexiDualGridVaeDecoder", trellis2_models.from_pretrained(self.config.decoder_pretrained_path))
        decoder.convert_to_fp32()
        # The decoder feature path casts activations to self.dtype.
        decoder.use_fp16 = False
        decoder.dtype = torch.float32
        decoder.set_resolution(self.config.decoder_resolution)
        decoder.float().eval().to(device)
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        for stage in decoder.blocks:
            for block in stage:
                block.use_checkpoint = True
        self.decoder = decoder

    def extra_loss_and_metrics(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.decoded_output_weight == 0.0 and self.config.decoded_subdivision_weight == 0.0:
            return prediction.new_zeros(()), {}

        decoder = cast("FlexiDualGridVaeDecoder", self.decoder)
        decoder_dtype = next(decoder.parameters()).dtype
        target_slat = SparseTensor(feats=target.to(dtype=decoder_dtype), coords=batch["coords_dst"])
        prediction_slat = SparseTensor(feats=prediction.to(dtype=decoder_dtype), coords=batch["coords_dst"])

        with torch.no_grad():
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                target_output, target_subdivisions = decode_shape_features(decoder, target_slat)

        decoded_output_points = target_output.feats.shape[0]
        decoded_output_points_metric = prediction.new_tensor(float(decoded_output_points))
        # Keep latent-space supervision while skipping a guided lattice that exceeds the decoder memory budget.
        if decoded_output_points > self.config.max_decoded_output_points:
            decoder_loss = prediction.new_zeros(())
            return decoder_loss, {
                "loss_decoded_output_l2_weighted": prediction.new_zeros(()),
                "loss_decoded_subdivision_l2_weighted": prediction.new_zeros(()),
                "decoder_loss": decoder_loss,
                "decoder_output_l2": prediction.new_zeros(()),
                "decoder_subdivision_l2": prediction.new_zeros(()),
                "decoder_supervision_keep_ratio": prediction.new_zeros(()),
                "decoder_target_output_points": decoded_output_points_metric,
            }

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            prediction_output, prediction_subdivisions = decode_shape_features(
                decoder,
                prediction_slat,
                target_subdivisions,
            )

        decoded_output_l2 = F.mse_loss(prediction_output.feats.float(), target_output.feats.float())
        decoded_subdivision_l2 = torch.stack([F.mse_loss(prediction_subdivision.feats.float(), target_subdivision.feats.float()) for prediction_subdivision, target_subdivision in zip(prediction_subdivisions, target_subdivisions)]).mean()

        output_loss = self.config.decoded_output_weight * decoded_output_l2
        subdivision_loss = self.config.decoded_subdivision_weight * decoded_subdivision_l2
        decoder_loss = output_loss + subdivision_loss
        return decoder_loss, {
            "loss_decoded_output_l2_weighted": output_loss,
            "loss_decoded_subdivision_l2_weighted": subdivision_loss,
            "decoder_loss": decoder_loss,
            "decoder_output_l2": decoded_output_l2,
            "decoder_subdivision_l2": decoded_subdivision_l2,
            "decoder_supervision_keep_ratio": prediction.new_ones(()),
            "decoder_target_output_points": decoded_output_points_metric,
        }
