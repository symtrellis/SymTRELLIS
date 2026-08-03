from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from inference.trellis2 import (
    TRELLIS2FlowPredictor,
    TRELLIS2SparseStructureLatentNoiseSampler,
    TRELLIS2SparseStructureSymmetryProjectionNoiseSampler,
    TRELLIS2SparseStructureView,
    trellis2_dense_grid_coords,
    trellis2_occ_to_visualization_mesh,
    trellis2_sparse_structure_logits_to_coords,
)
from symtrellis.flow import ClassifierFreeGuidanceWrapper, EulerSolver, SymmetryProjectionGuidanceWrapper
from symtrellis.mapper import SymmetryProjector, concat_coeff
from symtrellis.symmetry import build_symmetry_relation_inputs, get_3d_point_group

from ..loaders.trellis2 import DEVICE, TRELLIS2Runtime
from . import Operation, OperationContext, OperationInputs, OperationOutput, OperationResult
from .symmetry import ConfirmDetectedSymmetry, ConfirmManualSymmetry
from .trellis2_vanilla_sparse_structure import OCC_COORDINATES, OCC_VISUALIZATION_MESH, SPARSE_STRUCTURE_LATENT

SS_MAPPER_RELATION_CHUNK_SIZE = 4


class Trellis2SymmetrySparseStructure(Operation):
    operation_id = "trellis2.sparse_structure.symmetry"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SPARSE_STRUCTURE_LATENT, OCC_VISUALIZATION_MESH, OCC_COORDINATES)

    def __init__(self, runtime: TRELLIS2Runtime):
        self.runtime = runtime

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        image_condition_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            "trellis2.image_condition",
        )
        if image_condition_record is None:
            raise ValueError("trellis2.sparse_structure.symmetry requires trellis2.image_condition")

        condition_output = image_condition_record["outputs"].get("image_condition_512")
        if condition_output is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        symmetry_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            ConfirmDetectedSymmetry.operation_id,
        )
        if symmetry_record is None:
            symmetry_record = coordinator.find_lineage_node_run(
                request.parent_run_keys,
                ConfirmManualSymmetry.operation_id,
            )
        if symmetry_record is None:
            raise ValueError("trellis2.sparse_structure.symmetry requires a confirmed symmetry tuple")

        return OperationInputs(
            records={
                "image_condition": image_condition_record,
                "symmetry": symmetry_record,
            },
            paths={
                "condition": Path(condition_output["path"]),
            },
            metadata={
                "symmetry": symmetry_record["json_result"],
            },
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_role": "image_condition_512",
            "symmetry_node_run_key": inputs.records["symmetry"]["node_run_key"],
            "symmetry": inputs.metadata["symmetry"],
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

        noise_symmetry_projection_strength = float(params["noiseSymmetryProjectionStrength"])
        symmetry_projection_strength = float(params["symmetryProjectionStrength"])
        symmetry_projection_duration = (
            float(params["symmetryProjectionDuration"][0]),
            float(params["symmetryProjectionDuration"][1]),
        )

        device = torch.device(DEVICE)
        symmetry = inputs.metadata["symmetry"]
        symmetry_label = str(symmetry["label"])
        symmetry_center = torch.tensor(symmetry["center"], device=device, dtype=torch.float32)
        symmetry_major_axis = torch.tensor(symmetry["majorAxis"], device=device, dtype=torch.float32)
        symmetry_minor_axis = torch.tensor(symmetry["minorAxis"], device=device, dtype=torch.float32)

        condition = torch.load(inputs.paths["condition"], map_location="cpu")

        ss_flow_model = self.runtime.ss_flow_model

        batch_size = condition.shape[0]
        grid_size = ss_flow_model.resolution
        feat_dim = ss_flow_model.in_channels

        O_dst2src, t_dst2src, s_dst2src = get_3d_point_group(
            label=symmetry_label,
            center=symmetry_center,
            major_axis=symmetry_major_axis,
            minor_axis=symmetry_minor_axis,
            include_identity=False,
        )

        ss_coords = trellis2_dense_grid_coords(
            batch_size=batch_size,
            grid_size=grid_size,
            device=DEVICE,
        )
        ss_mapper = self.runtime.ss_mapper
        ss_coeff_chunks = []
        ss_rows_src_chunks = []
        ss_rows_dst_chunks = []

        num_relation_chunks = (O_dst2src.shape[0] + SS_MAPPER_RELATION_CHUNK_SIZE - 1) // SS_MAPPER_RELATION_CHUNK_SIZE
        try:
            ss_mapper.to(DEVICE)
            progress(0.0, desc="sparse_structure_mapper")
            for chunk_index, relation_start in enumerate(
                range(0, O_dst2src.shape[0], SS_MAPPER_RELATION_CHUNK_SIZE),
            ):
                relation_end = relation_start + SS_MAPPER_RELATION_CHUNK_SIZE
                chunk_relations = [
                    (
                        O_dst2src[relation_start:relation_end],
                        t_dst2src[relation_start:relation_end],
                        s_dst2src[relation_start:relation_end],
                    )
                    for _ in range(batch_size)
                ]

                ss_coords_src, ss_coords_dst, ss_rows_src_chunk, ss_rows_dst_chunk, ss_O, ss_t, ss_s = build_symmetry_relation_inputs(
                    coords=ss_coords,
                    relations=chunk_relations,
                    grid_size=grid_size,
                )
                ss_coeff_chunk = ss_mapper(
                    coords_src=ss_coords_src,
                    coords_dst=ss_coords_dst,
                    O_dst2src=ss_O,
                    t_dst2src=ss_t,
                    s_dst2src=ss_s,
                )
                ss_coeff_chunks.append(ss_coeff_chunk.to(torch.device("cpu")))
                ss_rows_src_chunks.append(ss_rows_src_chunk.cpu())
                ss_rows_dst_chunks.append(ss_rows_dst_chunk.cpu())
                del ss_coeff_chunk, ss_coords_src, ss_coords_dst, ss_O, ss_t, ss_s
                del ss_rows_src_chunk, ss_rows_dst_chunk, chunk_relations
                progress(
                    0.15 * (chunk_index + 1) / num_relation_chunks,
                    desc="sparse_structure_mapper",
                )
        finally:
            ss_mapper.cpu()
            torch.cuda.empty_cache()

        ss_coeff = concat_coeff(ss_coeff_chunks).to(device)
        ss_rows_src = torch.cat(ss_rows_src_chunks).to(device)
        ss_rows_dst = torch.cat(ss_rows_dst_chunks).to(device)

        ss_projector = SymmetryProjector(
            num_rows=ss_coords.shape[0],
            rows_src=ss_rows_src,
            rows_dst=ss_rows_dst,
            coeff=ss_coeff,
        )
        ss_view = TRELLIS2SparseStructureView(
            coords=ss_coords,
            grid_size=grid_size,
            batch_size=batch_size,
        )

        try:
            ss_flow_model.to(DEVICE)
            condition = condition.to(DEVICE)
            neg_condition = torch.zeros_like(condition)

            noise_sampler = TRELLIS2SparseStructureSymmetryProjectionNoiseSampler(
                sampler=TRELLIS2SparseStructureLatentNoiseSampler(),
                symmetry_strength=noise_symmetry_projection_strength,
            )
            noise = noise_sampler.sample(
                batch_size=batch_size,
                grid_size=grid_size,
                feat_dim=feat_dim,
                seed=seed,
                device=DEVICE,
                projector=ss_projector,
                to_sparse_view=ss_view.to_sparse_view,
                to_original_view=ss_view.to_original_view,
                self_include=True,
            )

            flow_predictor = TRELLIS2FlowPredictor(model=ss_flow_model)
            cfg_predictor = ClassifierFreeGuidanceWrapper(
                predictor=flow_predictor,
                strength=cfg_strength,
                interval=cfg_duration,
                rescale=cfg_rescale,
            )
            spg_predictor = SymmetryProjectionGuidanceWrapper(
                predictor=cfg_predictor,
                strength=symmetry_projection_strength,
                interval=symmetry_projection_duration,
                symmetrize_target="x_start",
                rescale=1.0,
            )

            flow_solver = EulerSolver()
            sparse_structure_latent = noise
            for step in flow_solver.iter_steps(
                noise=noise,
                predictor=spg_predictor,
                steps=steps,
                predictor_args={
                    "cond": condition,
                    "neg_cond": neg_condition,
                    "projector": ss_projector,
                    "to_sparse_view": ss_view.to_sparse_view,
                    "to_original_view": ss_view.to_original_view,
                    "self_include": True,
                },
                sigma_min=1e-5,
                rescale_t=time_step_rescale,
            ):
                sparse_structure_latent = step.x_t
                progress(
                    0.15 + 0.70 * float(step.t),
                    desc="sparse_structure_flow",
                )

            sparse_structure_latent = sparse_structure_latent.cpu()
            del ss_projector, ss_coeff, ss_coeff_chunks, ss_coords
            del ss_rows_src, ss_rows_dst, ss_rows_src_chunks, ss_rows_dst_chunks
            del ss_view, condition, neg_condition, noise, noise_sampler
            del flow_predictor, cfg_predictor, spg_predictor, flow_solver, step
            del O_dst2src, t_dst2src, s_dst2src, symmetry_center, symmetry_major_axis, symmetry_minor_axis
        finally:
            ss_flow_model.cpu()
            torch.cuda.empty_cache()

        ss_decoder = self.runtime.ss_decoder
        try:
            ss_decoder.to(DEVICE)
            sparse_structure_latent = sparse_structure_latent.to(DEVICE)
            occ_logits = ss_decoder(sparse_structure_latent)
            occ = occ_logits > 0
            pool_size = occ_logits.shape[-1] // 32
            occ_32 = torch.nn.functional.max_pool3d(occ.float(), pool_size, pool_size, 0) > 0.5
            occ_coordinates = trellis2_sparse_structure_logits_to_coords(
                logits=occ_logits,
                target_resolution=32,
            )
            occ_visualization_mesh = trellis2_occ_to_visualization_mesh(occ_32[0, 0], y_up=True)

            sparse_structure_latent = sparse_structure_latent.cpu()
            occ_coordinates = occ_coordinates.cpu()
            del occ_logits, occ, occ_32
        finally:
            ss_decoder.cpu()
            torch.cuda.empty_cache()
        progress(1.0, desc="sparse_structure_decode")

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
