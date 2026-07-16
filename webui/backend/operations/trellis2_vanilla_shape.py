from collections.abc import Callable
from functools import partial
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

from ..loaders.trellis2 import DEVICE, TRELLIS2Runtime
from . import Operation, OperationContext, OperationInputs, OperationOutput, OperationResult, forward_glb_progress

SHAPE_LATENT = "shape_latent"
SHAPE_RAW_MESH = "shape_raw_mesh"
SHAPE_VISUALIZATION_MESH = "shape_visualization_mesh"


class Trellis2VanillaShape(Operation):
    operation_id = "trellis2.shape.vanilla"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SHAPE_LATENT, SHAPE_RAW_MESH, SHAPE_VISUALIZATION_MESH)

    def __init__(self, runtime: TRELLIS2Runtime):
        self.runtime = runtime

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

        mode = params["mode"]
        max_tokens = int(params["maxTokens"])

        condition_512 = torch.load(inputs.paths["image_condition_512"], map_location="cpu")
        occ_coordinates = torch.load(inputs.paths["occ_coordinates"], map_location="cpu")

        shape_flow_model = self.runtime.shape_flow_model_512
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

        flow_512_scale = 0.55 if mode == "512" else 0.25
        progress(0.0, desc="shape_flow_512")
        for step in flow_solver.iter_steps(
            noise=shape_noise,
            predictor=cfg_predictor,
            steps=steps,
            predictor_args={
                "cond": condition_512,
                "neg_cond": neg_condition_512,
            },
            sigma_min=1e-5,
            rescale_t=time_step_rescale,
        ):
            shape_latent = step.x_t
            progress(
                flow_512_scale * float(step.t),
                desc="shape_flow_512",
            )

        shape_latent = shape_latent.replace(
            trellis2_shape_latent_to_sparse_view(shape_latent),
        )

        shape_latent = shape_latent.cpu()
        shape_flow_model.cpu()
        del condition_512, neg_condition_512, occ_coordinates
        del noise_sampler, shape_noise, flow_predictor, cfg_predictor, flow_solver, step
        torch.cuda.empty_cache()

        if mode == "512":
            shape_latent_grid_size = 32
            shape_grid_size = 512
        else:
            condition_1024 = torch.load(inputs.paths["image_condition_1024"], map_location="cpu")

            shape_decoder = self.runtime.shape_decoder
            shape_decoder.to(DEVICE)

            shape_latent = shape_latent.to(DEVICE)
            hr_coords = shape_decoder.upsample(shape_latent, upsample_times=4)

            hr_coords = hr_coords.cpu()
            shape_latent = shape_latent.cpu()
            shape_decoder.cpu()
            torch.cuda.empty_cache()
            progress(0.30, desc="shape_upsample")

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

            shape_flow_model = self.runtime.shape_flow_model_1024
            shape_flow_model.to(DEVICE)
            condition_1024 = condition_1024.to(DEVICE)
            neg_condition_1024 = torch.zeros_like(condition_1024)
            cascade_coordinates = cascade_coordinates.to(DEVICE)

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
            normalized_shape_latent = shape_noise

            for step in flow_solver.iter_steps(
                noise=shape_noise,
                predictor=cfg_predictor,
                steps=steps,
                predictor_args={
                    "cond": condition_1024,
                    "neg_cond": neg_condition_1024,
                },
                sigma_min=1e-5,
                rescale_t=time_step_rescale,
            ):
                normalized_shape_latent = step.x_t
                progress(
                    0.30 + 0.30 * float(step.t),
                    desc="shape_flow_1024",
                )

            shape_latent = SparseTensor(
                feats=trellis2_shape_latent_to_sparse_view(normalized_shape_latent),
                coords=normalized_shape_latent.coords,
            )

            shape_latent = shape_latent.cpu()
            shape_flow_model.cpu()
            del condition_1024, neg_condition_1024, cascade_coordinates, hr_coords
            del noise_sampler, shape_noise, normalized_shape_latent
            del flow_predictor, cfg_predictor, flow_solver, step
            torch.cuda.empty_cache()

        shape_decoder = self.runtime.shape_decoder
        shape_decoder.set_resolution(shape_grid_size)
        shape_decoder.to(DEVICE)

        shape_latent = shape_latent.to(DEVICE)
        progress(0.55 if mode == "512" else 0.60, desc="shape_decode")
        shape_meshes, shape_subs = shape_decoder(shape_latent, return_subs=True)
        shape_mesh = shape_meshes[0].cpu()
        shape_subs = [sub.cpu() for sub in shape_subs]

        shape_latent = shape_latent.cpu()
        shape_raw_mesh_vertices = shape_mesh.vertices.contiguous()
        shape_raw_mesh_faces = shape_mesh.faces.contiguous()
        shape_decoder.cpu()
        del shape_meshes
        torch.cuda.empty_cache()
        progress(0.70, desc="shape_decode")

        shape_latent_path = context.work_dir / "shape_latent.pt"
        shape_raw_mesh_path = context.work_dir / "shape_raw_mesh.pt"
        shape_visualization_mesh_path = context.work_dir / "shape_visualization_mesh.glb"

        torch.save(
            {
                "coords": shape_latent.coords.cpu().contiguous(),
                "feats": shape_latent.feats.cpu().contiguous(),
            },
            shape_latent_path,
        )

        torch.save(
            {
                "vertices": shape_raw_mesh_vertices.cpu().contiguous(),
                "faces": shape_raw_mesh_faces.cpu().contiguous(),
                "shape_subs": [
                    {
                        "coords": sub.coords.contiguous(),
                        "feats": sub.feats.contiguous(),
                    }
                    for sub in shape_subs
                ],
            },
            shape_raw_mesh_path,
        )

        shape_visualization_mesh = trelli2_mesh_to_glb(
            shape_mesh=shape_mesh,
            res=shape_grid_size,
            device=torch.device(DEVICE),
            texture_size=None,
            pbr_voxel=None,
            remesh=True,
            decimation_target=100_000,
            remesh_band=1.0,
            remesh_project=0.0,
            report=partial(forward_glb_progress, progress, 0.70, 0.30),
        )
        shape_visualization_mesh.export(shape_visualization_mesh_path)
        del shape_mesh
        torch.cuda.empty_cache()

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
                    role=SHAPE_RAW_MESH,
                    path=shape_raw_mesh_path,
                    filename="shape_raw_mesh.pt",
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
