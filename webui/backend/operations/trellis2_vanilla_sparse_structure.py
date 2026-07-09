from pathlib import Path
from typing import Any

import torch

from inference.trellis2 import (
    TRELLIS2FlowPredictor,
    TRELLIS2SparseStructureLatentNoiseSampler,
    trellis2_occ_to_visualization_mesh,
    trellis2_sparse_structure_logits_to_coords,
)
from symtrellis.flow import ClassifierFreeGuidanceWrapper, EulerSolver

from ..loaders.trellis2 import DEVICE, TRELLIS2Loader
from . import Emit, Operation, OperationContext, OperationInputs, OperationOutput, OperationResult

OCC_COORDINATES = "occ_coordinates"
OCC_VISUALIZATION_MESH = "occ_visualization_mesh"
SPARSE_STRUCTURE_LATENT = "sparse_structure_latent"


class Trellis2VanillaSparseStructure(Operation):
    operation_id = "trellis2.sparse_structure.vanilla"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SPARSE_STRUCTURE_LATENT, OCC_VISUALIZATION_MESH, OCC_COORDINATES)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        image_condition_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            "trellis2.image_condition",
        )
        if image_condition_record is None:
            raise ValueError("trellis2.sparse_structure.vanilla requires trellis2.image_condition")

        condition_output = image_condition_record["outputs"].get("image_condition_512")
        if condition_output is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        return OperationInputs(
            records={
                "image_condition": image_condition_record,
            },
            paths={
                "condition": Path(condition_output["path"]),
            },
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_role": "image_condition_512",
        }

    async def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        emit: Emit,
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

        condition = torch.load(inputs.paths["condition"], map_location="cpu")

        ss_flow_model = self.loader.ss_flow_model
        ss_flow_model.to(DEVICE)
        condition = condition.to(DEVICE)
        neg_condition = torch.zeros_like(condition)

        noise_sampler = TRELLIS2SparseStructureLatentNoiseSampler()
        noise = noise_sampler.sample(
            batch_size=1,
            grid_size=16,
            feat_dim=8,
            seed=seed,
            device=DEVICE,
        )

        flow_predictor = TRELLIS2FlowPredictor(model=ss_flow_model)
        cfg_predictor = ClassifierFreeGuidanceWrapper(
            predictor=flow_predictor,
            strength=cfg_strength,
            interval=cfg_duration,
            rescale=cfg_rescale,
        )

        flow_solver = EulerSolver()
        sparse_structure_latent = noise
        for step_index, step in enumerate(
            flow_solver.iter_steps(
                noise=noise,
                predictor=cfg_predictor,
                steps=steps,
                predictor_args={
                    "cond": condition,
                    "neg_cond": neg_condition,
                },
                sigma_min=1e-5,
                rescale_t=time_step_rescale,
            ),
        ):
            sparse_structure_latent = step.x_t
            if step_index > 0:
                await emit(
                    {
                        "type": "progress",
                        "progress": float(step.t),
                    },
                )

        ss_flow_model.cpu()
        condition = condition.cpu()
        neg_condition = neg_condition.cpu()
        torch.cuda.empty_cache()

        ss_decoder = self.loader.ss_decoder
        ss_decoder.to(DEVICE)
        occ_logits = ss_decoder(sparse_structure_latent.to(DEVICE))
        occ = occ_logits > 0
        pool_size = occ_logits.shape[-1] // 32
        occ_32 = torch.nn.functional.max_pool3d(occ.float(), pool_size, pool_size, 0) > 0.5
        occ_coordinates = trellis2_sparse_structure_logits_to_coords(
            logits=occ_logits,
            target_resolution=32,
        )
        occ_visualization_mesh = trellis2_occ_to_visualization_mesh(occ_32[0, 0])

        ss_decoder.cpu()
        sparse_structure_latent = sparse_structure_latent.cpu()
        occ_coordinates = occ_coordinates.cpu()
        torch.cuda.empty_cache()

        sparse_structure_latent_path = context.work_dir / "sparse_structure_latent.pt"
        occ_coordinates_path = context.work_dir / "occ_coordinates.pt"
        occ_visualization_mesh_path = context.work_dir / "occ_visualization_mesh.glb"

        torch.save(sparse_structure_latent, sparse_structure_latent_path)
        torch.save(occ_coordinates, occ_coordinates_path)
        occ_visualization_mesh.export(occ_visualization_mesh_path)

        voxel_count = int(occ_coordinates.shape[0])
        return OperationResult(
            outputs=[
                OperationOutput(
                    role=SPARSE_STRUCTURE_LATENT,
                    path=sparse_structure_latent_path,
                    filename="sparse_structure_latent.pt",
                ),
                OperationOutput(
                    role=OCC_VISUALIZATION_MESH,
                    path=occ_visualization_mesh_path,
                    filename="occ_visualization_mesh.glb",
                    metadata={
                        "gridSize": 32,
                        "voxelCount": voxel_count,
                    },
                ),
                OperationOutput(
                    role=OCC_COORDINATES,
                    path=occ_coordinates_path,
                    filename="occ_coordinates.pt",
                    metadata={
                        "gridSize": 32,
                        "voxelCount": voxel_count,
                    },
                ),
            ],
            metadata={
                "voxelCount": voxel_count,
            },
        )
