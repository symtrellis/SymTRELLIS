from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh

from inference.trellis2 import (
    TRELLIS2FlowPredictor,
    TRELLIS2TextureLatentNoiseSampler,
    trelli2_mesh_to_glb,
    trellis2_shape_sparse_view_to_latent,
    trellis2_texture_latent_to_sparse_view,
)
from symtrellis.flow import ClassifierFreeGuidanceWrapper, EulerSolver

from ..loaders.trellis2 import DEVICE, TRELLIS2Runtime
from . import Operation, OperationContext, OperationInputs, OperationOutput, OperationResult, forward_glb_progress
from .trellis2_image_condition import IMAGE_CONDITION_512, IMAGE_CONDITION_1024, Trellis2ImageCondition
from .trellis2_symmetry_shape import Trellis2SymmetryShape
from .trellis2_vanilla_shape import SHAPE_LATENT, SHAPE_RAW_MESH, Trellis2VanillaShape

TEXTURE_LATENT = "texture_latent"
PBR_VOXEL = "pbr_voxel"
FULL_VISUALIZATION_MESH = "full_visualization_mesh"


class Trellis2Texture(Operation):
    operation_id = "trellis2.texture.generate"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (TEXTURE_LATENT, PBR_VOXEL, FULL_VISUALIZATION_MESH)

    def __init__(self, runtime: TRELLIS2Runtime):
        self.runtime = runtime

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        image_condition_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            Trellis2ImageCondition.operation_id,
        )
        if image_condition_record is None:
            raise ValueError("trellis2.texture.generate requires trellis2.image_condition")

        shape_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            Trellis2SymmetryShape.operation_id,
        )
        if shape_record is None:
            shape_record = coordinator.find_lineage_node_run(
                request.parent_run_keys,
                Trellis2VanillaShape.operation_id,
            )
        if shape_record is None:
            raise ValueError("trellis2.texture.generate requires a shape node run")

        condition_512 = image_condition_record["outputs"].get(IMAGE_CONDITION_512)
        if condition_512 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        condition_1024 = image_condition_record["outputs"].get(IMAGE_CONDITION_1024)
        if condition_1024 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_1024")

        shape_latent = shape_record["outputs"].get(SHAPE_LATENT)
        if shape_latent is None:
            raise ValueError(f"{shape_record['operation_id']} did not produce shape_latent")

        shape_raw_mesh = shape_record["outputs"].get(SHAPE_RAW_MESH)
        if shape_raw_mesh is None:
            raise ValueError(f"{shape_record['operation_id']} did not produce shape_raw_mesh")

        return OperationInputs(
            records={
                "image_condition": image_condition_record,
                "shape": shape_record,
            },
            paths={
                IMAGE_CONDITION_512: Path(condition_512["path"]),
                IMAGE_CONDITION_1024: Path(condition_1024["path"]),
                SHAPE_LATENT: Path(shape_latent["path"]),
                SHAPE_RAW_MESH: Path(shape_raw_mesh["path"]),
            },
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_roles": [IMAGE_CONDITION_512, IMAGE_CONDITION_1024],
            "shape_node_run_key": inputs.records["shape"]["node_run_key"],
            "shape_operation_id": inputs.records["shape"]["operation_id"],
            "shape_roles": [SHAPE_LATENT, SHAPE_RAW_MESH],
        }

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any],
    ) -> OperationResult:
        seed = int(params["seed"])
        steps = int(params["steps"])
        time_step_rescale = float(params["timeStepRescale"])

        cfg_strength = float(params["cfgStrength"])
        cfg_duration = (
            float(params["cfgDuration"][0]),
            float(params["cfgDuration"][1]),
        )
        cfg_rescale = float(params["cfgRescale"])

        shape_metadata = inputs.records["shape"]["outputs"][SHAPE_LATENT]["metadata"]
        shape_latent_grid_size = int(shape_metadata["shapeLatentGridSize"])
        o_voxel_grid_size = int(shape_metadata["oVoxelGridSize"])

        if shape_latent_grid_size == 32:
            condition = torch.load(inputs.paths[IMAGE_CONDITION_512], map_location="cpu")
            texture_flow_model = self.runtime.texture_flow_model_512
        else:
            condition = torch.load(inputs.paths[IMAGE_CONDITION_1024], map_location="cpu")
            texture_flow_model = self.runtime.texture_flow_model_1024

        shape_payload = torch.load(inputs.paths[SHAPE_LATENT], map_location="cpu")
        normalized_shape_latent = trellis2_shape_sparse_view_to_latent(
            sparse_view=shape_payload["feats"],
            coords=shape_payload["coords"],
            sp_class=SparseTensor,
        )

        texture_flow_model.to(DEVICE)
        condition = condition.to(DEVICE)
        normalized_shape_latent = normalized_shape_latent.to(DEVICE)
        neg_condition = torch.zeros_like(condition)

        noise_sampler = TRELLIS2TextureLatentNoiseSampler()
        normalized_texture_noise = noise_sampler.sample(
            sp_class=SparseTensor,
            coords=normalized_shape_latent.coords,
            feat_dim=texture_flow_model.in_channels - normalized_shape_latent.feats.shape[1],
            grid_size=shape_latent_grid_size,
            seed=seed,
            device=DEVICE,
        )

        flow_predictor = TRELLIS2FlowPredictor(model=texture_flow_model)
        cfg_predictor = ClassifierFreeGuidanceWrapper(
            predictor=flow_predictor,
            strength=cfg_strength,
            interval=cfg_duration,
            rescale=cfg_rescale,
        )

        flow_solver = EulerSolver()
        normalized_texture_latent = normalized_texture_noise

        progress(0.0, desc="texture_flow")
        for step in flow_solver.iter_steps(
            noise=normalized_texture_noise,
            predictor=cfg_predictor,
            steps=steps,
            predictor_args={
                "cond": condition,
                "neg_cond": neg_condition,
                "concat_cond": normalized_shape_latent,
            },
            sigma_min=1e-5,
            rescale_t=time_step_rescale,
        ):
            normalized_texture_latent = step.x_t
            progress(0.55 * float(step.t), desc="texture_flow")

        texture_latent = normalized_texture_latent.replace(
            trellis2_texture_latent_to_sparse_view(normalized_texture_latent),
        )

        texture_latent = texture_latent.cpu()
        texture_flow_model.cpu()
        del condition, neg_condition, normalized_shape_latent
        del noise_sampler, normalized_texture_noise, normalized_texture_latent
        del flow_predictor, cfg_predictor, flow_solver, step
        torch.cuda.empty_cache()

        raw_mesh_payload = torch.load(inputs.paths[SHAPE_RAW_MESH], map_location="cpu")

        shape_subs = [
            SparseTensor(
                feats=item["feats"].to(DEVICE),
                coords=item["coords"].to(DEVICE),
            )
            for item in raw_mesh_payload["shape_subs"]
        ]

        texture_decoder = self.runtime.texture_decoder
        texture_decoder.to(DEVICE)

        texture_latent = texture_latent.to(DEVICE)
        progress(0.55, desc="texture_decode")
        pbr_voxel = texture_decoder(texture_latent, guide_subs=shape_subs) * 0.5 + 0.5

        texture_latent = texture_latent.cpu()
        pbr_voxel = pbr_voxel.cpu()
        del shape_subs
        texture_decoder.cpu()
        torch.cuda.empty_cache()
        progress(0.65, desc="texture_decode")

        texture_latent_path = context.work_dir / "texture_latent.pt"
        pbr_voxel_path = context.work_dir / "pbr_voxel.pt"
        full_visualization_mesh_path = context.work_dir / "full_visualization_mesh.glb"

        shape_raw_mesh = Mesh(
            vertices=raw_mesh_payload["vertices"],
            faces=raw_mesh_payload["faces"],
        )

        full_visualization_mesh = trelli2_mesh_to_glb(
            shape_mesh=shape_raw_mesh,
            res=o_voxel_grid_size,
            device=torch.device(DEVICE),
            texture_size=2048,
            pbr_voxel=pbr_voxel,
            remesh=True,
            decimation_target=100_000,
            remesh_band=1.0,
            remesh_project=0.0,
            report=partial(forward_glb_progress, progress, 0.65, 0.35),
        )
        full_visualization_mesh.export(full_visualization_mesh_path)

        pbr_voxel = pbr_voxel.cpu()
        del shape_raw_mesh
        torch.cuda.empty_cache()
        texture_voxel_count = int(pbr_voxel.coords.shape[0])

        torch.save(
            {
                "coords": texture_latent.coords,
                "feats": texture_latent.feats,
            },
            texture_latent_path,
        )

        torch.save(
            {
                "coords": pbr_voxel.coords,
                "feats": pbr_voxel.feats,
            },
            pbr_voxel_path,
        )

        return OperationResult(
            outputs=[
                OperationOutput(
                    role=TEXTURE_LATENT,
                    path=texture_latent_path,
                    filename="texture_latent.pt",
                    metadata={
                        "shapeLatentGridSize": shape_latent_grid_size,
                        "oVoxelGridSize": o_voxel_grid_size,
                        "textureVoxelCount": texture_voxel_count,
                    },
                ),
                OperationOutput(
                    role=PBR_VOXEL,
                    path=pbr_voxel_path,
                    filename="pbr_voxel.pt",
                    metadata={
                        "shapeLatentGridSize": shape_latent_grid_size,
                        "oVoxelGridSize": o_voxel_grid_size,
                        "textureVoxelCount": texture_voxel_count,
                    },
                ),
                OperationOutput(
                    role=FULL_VISUALIZATION_MESH,
                    path=full_visualization_mesh_path,
                    filename="full_visualization_mesh.glb",
                    metadata={
                        "shapeLatentGridSize": shape_latent_grid_size,
                        "oVoxelGridSize": o_voxel_grid_size,
                        "textureVoxelCount": texture_voxel_count,
                    },
                ),
            ],
            metadata={
                "shapeLatentGridSize": shape_latent_grid_size,
                "oVoxelGridSize": o_voxel_grid_size,
                "textureVoxelCount": texture_voxel_count,
            },
        )
