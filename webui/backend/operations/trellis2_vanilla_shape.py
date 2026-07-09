import asyncio
from pathlib import Path
from typing import Any

import torch
from trellis2.modules.sparse import SparseTensor

from inference.trellis2 import (
    TRELLIS2FlowPredictor,
    TRELLIS2ShapeLatentNoiseSampler,
    trelli2_mesh_to_glb,
    trellis2_shape_latent_to_sparse_view,
)
from symtrellis.flow import ClassifierFreeGuidanceWrapper, EulerSolver

from ..loaders.trellis2 import DEVICE, TRELLIS2Loader
from . import Emit, Operation, OperationContext, OperationInputs, OperationOutput, OperationResult

SHAPE_LATENT = "shape_latent"
SHAPE_VISUALIZATION_MESH = "shape_visualization_mesh"


class Trellis2VanillaShape(Operation):
    operation_id = "trellis2.shape.vanilla"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SHAPE_LATENT, SHAPE_VISUALIZATION_MESH)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        image_condition_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            "trellis2.image_condition",
        )
        if image_condition_record is None:
            raise ValueError("trellis2.shape.vanilla requires trellis2.image_condition")

        sparse_structure_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            "trellis2.sparse_structure.vanilla",
        )
        if sparse_structure_record is None:
            raise ValueError("trellis2.shape.vanilla requires trellis2.sparse_structure.vanilla")

        condition_512 = image_condition_record["outputs"].get("image_condition_512")
        if condition_512 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        condition_1024 = image_condition_record["outputs"].get("image_condition_1024")
        if condition_1024 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_1024")

        occ_coordinates = sparse_structure_record["outputs"].get("occ_coordinates")
        if occ_coordinates is None:
            raise ValueError("trellis2.sparse_structure.vanilla did not produce occ_coordinates")

        return OperationInputs(
            records={
                "image_condition": image_condition_record,
                "sparse_structure": sparse_structure_record,
            },
            paths={
                "image_condition_512": Path(condition_512["path"]),
                "image_condition_1024": Path(condition_1024["path"]),
                "occ_coordinates": Path(occ_coordinates["path"]),
            },
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_roles": ["image_condition_512", "image_condition_1024"],
            "sparse_structure_node_run_key": inputs.records["sparse_structure"]["node_run_key"],
            "sparse_structure_roles": ["occ_coordinates"],
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

        mode = params["mode"]
        max_tokens = int(params["maxTokens"])

        condition_512 = torch.load(inputs.paths["image_condition_512"], map_location="cpu")
        occ_coordinates = torch.load(inputs.paths["occ_coordinates"], map_location="cpu")

        shape_flow_model = self.loader.shape_flow_model_512
        shape_flow_model.to(DEVICE)
        condition_512 = condition_512.to(DEVICE)
        occ_coordinates = occ_coordinates.to(DEVICE)
        neg_condition_512 = torch.zeros_like(condition_512)

        noise_sampler = TRELLIS2ShapeLatentNoiseSampler()
        shape_noise = noise_sampler.sample(
            sp_class=SparseTensor,
            coords=occ_coordinates,
            feat_dim=32,
            grid_size=32,
            seed=seed,
            device=DEVICE,
        )

        flow_predictor = TRELLIS2FlowPredictor(model=shape_flow_model)
        cfg_predictor = ClassifierFreeGuidanceWrapper(
            predictor=flow_predictor,
            strength=cfg_strength,
            interval=cfg_duration,
            rescale=cfg_rescale,
        )

        flow_solver = EulerSolver()
        shape_latent = shape_noise

        await emit({"type": "progress", "progress": 0.0})
        for step_index, step in enumerate(
            flow_solver.iter_steps(
                noise=shape_noise,
                predictor=cfg_predictor,
                steps=steps,
                predictor_args={
                    "cond": condition_512,
                    "neg_cond": neg_condition_512,
                },
                sigma_min=1e-5,
                rescale_t=time_step_rescale,
            ),
        ):
            shape_latent = step.x_t
            if step_index > 0:
                await emit({"type": "progress", "progress": float(step.t)})

        shape_latent = shape_latent.replace(
            trellis2_shape_latent_to_sparse_view(shape_latent),
        )

        shape_flow_model.cpu()
        condition_512 = condition_512.cpu()
        neg_condition_512 = neg_condition_512.cpu()
        torch.cuda.empty_cache()

        if mode == "512":
            shape_latent_grid_size = 32
            shape_grid_size = 512
        else:
            condition_1024 = torch.load(inputs.paths["image_condition_1024"], map_location="cpu")
            condition_1024 = condition_1024.to(DEVICE)
            neg_condition_1024 = torch.zeros_like(condition_1024)

            shape_decoder = self.loader.shape_decoder
            shape_decoder.to(DEVICE)

            hr_coords = shape_decoder.upsample(shape_latent, upsample_times=4)

            shape_latent = shape_latent.cpu()
            shape_decoder.cpu()
            torch.cuda.empty_cache()

            cascade_coordinates = None
            shape_latent_grid_size = 64
            shape_grid_size = 1024

            cascade_coordinates = hr_coords
            for candidate_shape_latent_grid_size in (96, 88, 80, 72, 64):
                quantized_coordinates = torch.cat(
                    [
                        hr_coords[:, :1],
                        ((hr_coords[:, 1:] + 0.5) / 512 * candidate_shape_latent_grid_size).int(),
                    ],
                    dim=1,
                )
                candidate_coordinates = quantized_coordinates.unique(dim=0)

                if candidate_coordinates.shape[0] < max_tokens or candidate_shape_latent_grid_size == 64:
                    cascade_coordinates = candidate_coordinates
                    shape_latent_grid_size = candidate_shape_latent_grid_size
                    shape_grid_size = candidate_shape_latent_grid_size * 16
                    break

            shape_flow_model = self.loader.shape_flow_model_1024
            shape_flow_model.to(DEVICE)

            noise_sampler = TRELLIS2ShapeLatentNoiseSampler()
            shape_noise = noise_sampler.sample(
                sp_class=SparseTensor,
                coords=cascade_coordinates,
                feat_dim=32,
                grid_size=shape_latent_grid_size,
                seed=seed,
                device=DEVICE,
            )

            flow_predictor = TRELLIS2FlowPredictor(model=shape_flow_model)
            cfg_predictor = ClassifierFreeGuidanceWrapper(
                predictor=flow_predictor,
                strength=cfg_strength,
                interval=cfg_duration,
                rescale=cfg_rescale,
            )

            flow_solver = EulerSolver()
            shape_latent = shape_noise

            await emit({"type": "progress", "progress": 0.0})
            for step_index, step in enumerate(
                flow_solver.iter_steps(
                    noise=shape_noise,
                    predictor=cfg_predictor,
                    steps=steps,
                    predictor_args={
                        "cond": condition_1024,
                        "neg_cond": neg_condition_1024,
                    },
                    sigma_min=1e-5,
                    rescale_t=time_step_rescale,
                ),
            ):
                shape_latent = step.x_t
                if step_index > 0:
                    await emit({"type": "progress", "progress": float(step.t)})

            shape_latent = shape_latent.replace(
                trellis2_shape_latent_to_sparse_view(shape_latent),
            )

            shape_flow_model.cpu()
            condition_1024 = condition_1024.cpu()
            neg_condition_1024 = neg_condition_1024.cpu()
            torch.cuda.empty_cache()

        shape_decoder = self.loader.shape_decoder
        shape_decoder.set_resolution(shape_grid_size)
        shape_decoder.to(DEVICE)

        shape_mesh = shape_decoder(shape_latent, return_subs=False)[0]

        shape_latent = shape_latent.cpu()
        shape_decoder.cpu()
        torch.cuda.empty_cache()

        shape_latent_path = context.work_dir / "shape_latent.pt"
        shape_visualization_mesh_path = context.work_dir / "shape_visualization_mesh.glb"

        torch.save(
            {
                "coords": shape_latent.coords,
                "feats": shape_latent.feats,
            },
            shape_latent_path,
        )

        loop = asyncio.get_running_loop()

        def report_glb(update: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(
                emit(
                    {
                        "type": "progress",
                        "progress": float(update["progress"]),
                        "stage": update["stage"],
                    },
                ),
                loop,
            )

        await emit({"type": "progress", "progress": 0.0})
        shape_visualization_mesh = await asyncio.to_thread(
            trelli2_mesh_to_glb,
            shape_mesh=shape_mesh,
            res=shape_grid_size,
            device=torch.device(DEVICE),
            texture_size=None,
            pbr_voxel=None,
            remesh=True,
            decimation_target=100_000,
            remesh_band=1.0,
            remesh_project=0.0,
            report=report_glb,
        )
        shape_visualization_mesh.export(shape_visualization_mesh_path)

        voxel_count = int(shape_latent.coords.shape[0])

        return OperationResult(
            outputs=[
                OperationOutput(
                    role=SHAPE_LATENT,
                    path=shape_latent_path,
                    filename="shape_latent.pt",
                    metadata={
                        "shapeLatentGridSize": shape_latent_grid_size,
                        "oVoxelGridSize": shape_grid_size,
                        "voxelCount": voxel_count,
                    },
                ),
                OperationOutput(
                    role=SHAPE_VISUALIZATION_MESH,
                    path=shape_visualization_mesh_path,
                    filename="shape_visualization_mesh.glb",
                    metadata={
                        "shapeLatentGridSize": shape_latent_grid_size,
                        "oVoxelGridSize": shape_grid_size,
                        "voxelCount": voxel_count,
                    },
                ),
            ],
            metadata={
                "shapeLatentGridSize": shape_latent_grid_size,
                "oVoxelGridSize": shape_grid_size,
                "voxelCount": voxel_count,
            },
        )
