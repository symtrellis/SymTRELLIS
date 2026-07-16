import sys
from pathlib import Path
from typing import cast

trellis2_root = Path(__file__).resolve().parents[3] / "third_party" / "trellis2"
if str(trellis2_root) not in sys.path:
    sys.path.insert(0, str(trellis2_root))

import trellis2.models as trellis2_model_registry
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.sparse_structure_vae import SparseStructureDecoder
from trellis2.models.structured_latent_flow import ElasticSLatFlowModel
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
from trellis2.pipelines.rembg import BiRefNet

from symtrellis.mapper import BaseSpatialTransformLatentMapper, NeighborGraphLatentMapper, Swin3DLatentMapper, from_pretrained

DEVICE = "cuda:0"

SS_FLOW_MODEL = "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16"
SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"

SHAPE_FLOW_512 = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
SHAPE_FLOW_1024 = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
SHAPE_DECODER = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

TEXTURE_FLOW_512 = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
TEXTURE_FLOW_1024 = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
TEXTURE_DECODER = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"

IMAGE_COND_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_MODEL = "briaai/RMBG-2.0"

SYMTRELLIS_REPO = "quantaji/SymTRELLIS"
SPARSE_STRUCTURE_MAPPER = "trellis2_sparse_structure_neighbor_graph_finetune"
SHAPE_MAPPER = "trellis2_shape_neighbor_graph_pretrain"


class TRELLIS2Runtime:
    def __init__(self):
        self._image_cond_model: DinoV3FeatureExtractor | None = None
        self._rembg_model: BiRefNet | None = None

        self._ss_flow_model: SparseStructureFlowModel | None = None
        self._ss_decoder: SparseStructureDecoder | None = None

        self._shape_flow_model_512: ElasticSLatFlowModel | None = None
        self._shape_flow_model_1024: ElasticSLatFlowModel | None = None
        self._shape_decoder: FlexiDualGridVaeDecoder | None = None

        self._texture_flow_model_512: ElasticSLatFlowModel | None = None
        self._texture_flow_model_1024: ElasticSLatFlowModel | None = None
        self._texture_decoder: SparseUnetVaeDecoder | None = None

        self._ss_mapper: BaseSpatialTransformLatentMapper | None = None
        self._shape_mapper: BaseSpatialTransformLatentMapper | None = None

    @property
    def image_cond_model(self) -> DinoV3FeatureExtractor:
        if self._image_cond_model is None:
            self._image_cond_model = DinoV3FeatureExtractor(IMAGE_COND_MODEL, image_size=512)
            self._image_cond_model.model.eval()

        return self._image_cond_model

    @property
    def rembg_model(self) -> BiRefNet:
        if self._rembg_model is None:
            self._rembg_model = BiRefNet(REMBG_MODEL)
            self._rembg_model.model.eval()

        return self._rembg_model

    @property
    def ss_flow_model(self) -> SparseStructureFlowModel:
        if self._ss_flow_model is None:
            self._ss_flow_model = cast(
                SparseStructureFlowModel,
                trellis2_model_registry.from_pretrained(SS_FLOW_MODEL).eval(),
            )

        return self._ss_flow_model

    @property
    def ss_decoder(self) -> SparseStructureDecoder:
        if self._ss_decoder is None:
            self._ss_decoder = cast(
                SparseStructureDecoder,
                trellis2_model_registry.from_pretrained(SS_DECODER).eval(),
            )

        return self._ss_decoder

    @property
    def shape_flow_model_512(self) -> ElasticSLatFlowModel:
        if self._shape_flow_model_512 is None:
            self._shape_flow_model_512 = cast(
                ElasticSLatFlowModel,
                trellis2_model_registry.from_pretrained(SHAPE_FLOW_512).eval(),
            )

        return self._shape_flow_model_512

    @property
    def shape_flow_model_1024(self) -> ElasticSLatFlowModel:
        if self._shape_flow_model_1024 is None:
            self._shape_flow_model_1024 = cast(
                ElasticSLatFlowModel,
                trellis2_model_registry.from_pretrained(SHAPE_FLOW_1024).eval(),
            )

        return self._shape_flow_model_1024

    @property
    def shape_decoder(self) -> FlexiDualGridVaeDecoder:
        if self._shape_decoder is None:
            self._shape_decoder = cast(
                FlexiDualGridVaeDecoder,
                trellis2_model_registry.from_pretrained(SHAPE_DECODER).eval(),
            )

        return self._shape_decoder

    @property
    def texture_flow_model_512(self) -> ElasticSLatFlowModel:
        if self._texture_flow_model_512 is None:
            self._texture_flow_model_512 = cast(
                ElasticSLatFlowModel,
                trellis2_model_registry.from_pretrained(TEXTURE_FLOW_512).eval(),
            )

        return self._texture_flow_model_512

    @property
    def texture_flow_model_1024(self) -> ElasticSLatFlowModel:
        if self._texture_flow_model_1024 is None:
            self._texture_flow_model_1024 = cast(
                ElasticSLatFlowModel,
                trellis2_model_registry.from_pretrained(TEXTURE_FLOW_1024).eval(),
            )

        return self._texture_flow_model_1024

    @property
    def texture_decoder(self) -> SparseUnetVaeDecoder:
        if self._texture_decoder is None:
            self._texture_decoder = cast(
                SparseUnetVaeDecoder,
                trellis2_model_registry.from_pretrained(TEXTURE_DECODER).eval(),
            )

        return self._texture_decoder

    @property
    def ss_mapper(self) -> BaseSpatialTransformLatentMapper:
        if self._ss_mapper is None:
            self._ss_mapper = from_pretrained(
                SPARSE_STRUCTURE_MAPPER,
                repo_id=SYMTRELLIS_REPO,
                device="cpu",
            ).eval()

        return self._ss_mapper

    @property
    def shape_mapper(self) -> BaseSpatialTransformLatentMapper:
        if self._shape_mapper is None:
            self._shape_mapper = from_pretrained(
                SHAPE_MAPPER,
                repo_id=SYMTRELLIS_REPO,
                device="cpu",
            ).eval()

        return self._shape_mapper
