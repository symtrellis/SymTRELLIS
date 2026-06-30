import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS2_ROOT = REPO_ROOT / "third_party" / "trellis2"
# TRELLIS-specific imports stay in the task layer so base training remains package-local.
if str(TRELLIS2_ROOT) not in sys.path:
    sys.path.append(str(TRELLIS2_ROOT))

import trellis2.models as trellis2_models
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from trellis2.modules.sparse import SparseTensor

from trainer.base import BaseTask
from trainer.config import TrainConfig

if TYPE_CHECKING:
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

DEFAULT_DECODER_PRETRAINED_PATH = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"


@dataclass
class Config(TrainConfig):
    decoder_pretrained_path: str = DEFAULT_DECODER_PRETRAINED_PATH
    decoded_loss_weight: float = 0.0
    decoder_resolution: int = 512


class Task(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.decoder: "FlexiDualGridVaeDecoder | None" = None

    def setup(self, device: torch.device, amp_dtype: torch.dtype, amp_enabled: bool) -> None:
        super().setup(device, amp_dtype, amp_enabled)

        if self.config.decoded_loss_weight == 0.0:
            return

        decoder = cast("FlexiDualGridVaeDecoder", trellis2_models.from_pretrained(self.config.decoder_pretrained_path))
        decoder.convert_to_fp32()
        # SparseUnetVaeDecoder.forward casts activations to self.dtype.
        decoder.use_fp16 = False
        decoder.dtype = torch.float32
        decoder.set_resolution(self.config.decoder_resolution)
        decoder.float().eval().to(device)
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder = decoder

    def extra_loss_and_metrics(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.decoded_loss_weight == 0.0:
            return prediction.new_zeros(()), {}

        decoder = self.decoder
        assert decoder is not None

        decoder_dtype = next(decoder.parameters()).dtype
        target_slat = SparseTensor(feats=target.to(dtype=decoder_dtype), coords=batch["coords_dst"])
        prediction_slat = SparseTensor(feats=prediction.to(dtype=decoder_dtype), coords=batch["coords_dst"])

        # Use the raw sparse decoder output, not the FDG mesh wrapper returned by public forward.
        with torch.no_grad():
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                decoded_target = cast(
                    SparseTensor,
                    cast(Any, SparseUnetVaeDecoder.forward)(decoder, target_slat),
                )

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            decoded_prediction = cast(
                SparseTensor,
                cast(Any, SparseUnetVaeDecoder.forward)(decoder, prediction_slat),
            )

        extent = self.config.decoder_resolution + 1
        pred_coords = decoded_prediction.coords.long()
        target_coords = decoded_target.coords.long()
        pred_keys = pred_coords[:, 0] * extent * extent * extent + pred_coords[:, 1] * extent * extent + pred_coords[:, 2] * extent + pred_coords[:, 3]
        target_keys = target_coords[:, 0] * extent * extent * extent + target_coords[:, 1] * extent * extent + target_coords[:, 2] * extent + target_coords[:, 3]

        target_order = torch.argsort(target_keys)
        target_keys_sorted = target_keys[target_order]
        pred_pos = torch.searchsorted(target_keys_sorted, pred_keys)
        valid_pos = pred_pos < target_keys_sorted.numel()
        pred_valid_index = torch.arange(pred_keys.shape[0], device=pred_keys.device)[valid_pos]
        pred_valid_pos = pred_pos[valid_pos]
        matched = target_keys_sorted[pred_valid_pos] == pred_keys[pred_valid_index]
        pred_index = pred_valid_index[matched]
        target_index = target_order[pred_valid_pos[matched]]

        if pred_index.numel() == 0:
            decoded_l2 = decoded_prediction.feats.sum() * 0.0
        else:
            decoded_l2 = F.mse_loss(
                decoded_prediction.feats[pred_index].float(),
                decoded_target.feats[target_index].float(),
            )

        pred_count = decoded_l2.new_tensor(float(pred_keys.shape[0]))
        target_count = decoded_l2.new_tensor(float(target_keys.shape[0]))
        intersection_count = decoded_l2.new_tensor(float(pred_index.numel()))
        union_count = pred_count + target_count - intersection_count
        extra_loss = self.config.decoded_loss_weight * decoded_l2
        return extra_loss, {
            "loss_decoded_l2_weighted": extra_loss,
            "decoder_l2": decoded_l2,
            "decoder_coord_iou": intersection_count / union_count.clamp_min(1.0),
            "decoder_coord_intersection_ratio_pred": intersection_count / pred_count.clamp_min(1.0),
            "decoder_coord_intersection_ratio_target": intersection_count / target_count.clamp_min(1.0),
            "decoder_coord_count_pred": pred_count,
            "decoder_coord_count_target": target_count,
            "decoder_coord_count_intersection": intersection_count,
        }
