import asyncio
from pathlib import Path
from typing import Any

import torch
from trellis2.modules.sparse import SparseTensor

from inference.trellis2 import (
    TRELLIS2FlowPredictor,
    TRELLIS2ShapeLatentNoiseSampler,
    TRELLIS2ShapeLatentView,
    trelli2_mesh_to_glb,
    trellis2_shape_latent_to_sparse_view,
)
from symtrellis.flow import (
    ClassifierFreeGuidanceWrapper,
    EulerSolver,
    SymmetryProjectionGuidanceWrapper,
    SymmetryProjectionNoiseSampler,
)
from symtrellis.mapper import SymmetryProjector, concat_coeff
from symtrellis.symmetry import build_symmetry_relation_inputs, get_3d_point_group

from ..loaders.trellis2 import DEVICE, TRELLIS2Loader
from . import Emit, Operation, OperationContext, OperationInputs, OperationOutput, OperationResult
from .symmetry import ConfirmDetectedSymmetry, ConfirmManualSymmetry
from .trellis2_symmetry_sparse_structure import Trellis2SymmetrySparseStructure
from .trellis2_vanilla_shape import SHAPE_LATENT, SHAPE_RAW_MESH, SHAPE_VISUALIZATION_MESH

SHAPE_MAPPER_RELATION_CHUNK_SIZE = 1


class Trellis2SymmetryShape(Operation):
    operation_id = "trellis2.shape.symmetry"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SHAPE_LATENT, SHAPE_RAW_MESH, SHAPE_VISUALIZATION_MESH)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        image_condition_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            "trellis2.image_condition",
        )
        if image_condition_record is None:
            raise ValueError("trellis2.shape.symmetry requires trellis2.image_condition")

        sparse_structure_record = coordinator.find_lineage_node_run(
            request.parent_run_keys,
            Trellis2SymmetrySparseStructure.operation_id,
        )
        if sparse_structure_record is None:
            raise ValueError("trellis2.shape.symmetry requires trellis2.sparse_structure.symmetry")

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
            raise ValueError("trellis2.shape.symmetry requires a confirmed symmetry tuple")

        condition_512 = image_condition_record["outputs"].get("image_condition_512")
        if condition_512 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        condition_1024 = image_condition_record["outputs"].get("image_condition_1024")
        if condition_1024 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_1024")

        occ_coordinates = sparse_structure_record["outputs"].get("occ_coordinates")
        if occ_coordinates is None:
            raise ValueError("trellis2.sparse_structure.symmetry did not produce occ_coordinates")

        return OperationInputs(
            records={
                "image_condition": image_condition_record,
                "sparse_structure": sparse_structure_record,
                "symmetry": symmetry_record,
            },
            paths={
                "image_condition_512": Path(condition_512["path"]),
                "image_condition_1024": Path(condition_1024["path"]),
                "occ_coordinates": Path(occ_coordinates["path"]),
            },
            metadata={
                "symmetry": symmetry_record["json_result"],
            },
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_roles": ["image_condition_512", "image_condition_1024"],
            "sparse_structure_node_run_key": inputs.records["sparse_structure"]["node_run_key"],
            "sparse_structure_roles": ["occ_coordinates"],
            "symmetry_node_run_key": inputs.records["symmetry"]["node_run_key"],
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

        noise_symmetry_projection_strength = float(params["noiseSymmetryProjectionStrength"])
        symmetry_projection_strength = float(params["symmetryProjectionStrength"])
        symmetry_projection_duration = (
            float(params["symmetryProjectionDuration"][0]),
            float(params["symmetryProjectionDuration"][1]),
        )

        mode = params["mode"]
        max_tokens = int(params["maxTokens"])

        symmetry = inputs.metadata["symmetry"]
        symmetry_label = str(symmetry["label"])
        symmetry_center = torch.tensor(symmetry["center"], device=DEVICE, dtype=torch.float32)
        symmetry_major_axis = torch.tensor(symmetry["majorAxis"], device=DEVICE, dtype=torch.float32)
        symmetry_minor_axis = torch.tensor(symmetry["minorAxis"], device=DEVICE, dtype=torch.float32)

        condition_512 = torch.load(inputs.paths["image_condition_512"], map_location="cpu")
        occ_coordinates = torch.load(inputs.paths["occ_coordinates"], map_location="cpu")

        shape_flow_model = self.loader.shape_flow_model_512
        occ_coordinates = occ_coordinates.to(DEVICE)

        O_dst2src, t_dst2src, s_dst2src = get_3d_point_group(
            label=symmetry_label,
            center=symmetry_center,
            major_axis=symmetry_major_axis,
            minor_axis=symmetry_minor_axis,
            include_identity=False,
        )

        shape_mapper = self.loader.shape_mapper
        shape_mapper.to(DEVICE)

        shape_coeff_chunks = []
        shape_rows_src_chunks = []
        shape_rows_dst_chunks = []

        for relation_start in range(0, O_dst2src.shape[0], SHAPE_MAPPER_RELATION_CHUNK_SIZE):
            relation_end = relation_start + SHAPE_MAPPER_RELATION_CHUNK_SIZE
            chunk_relations = [
                (
                    O_dst2src[relation_start:relation_end],
                    t_dst2src[relation_start:relation_end],
                    s_dst2src[relation_start:relation_end],
                )
                for _ in range(condition_512.shape[0])
            ]

            shape_coords_src, shape_coords_dst, shape_rows_src_chunk, shape_rows_dst_chunk, shape_O, shape_t, shape_s = build_symmetry_relation_inputs(
                coords=occ_coordinates,
                relations=chunk_relations,
                grid_size=32,
            )
            shape_coeff_chunk = shape_mapper(
                coords_src=shape_coords_src,
                coords_dst=shape_coords_dst,
                O_dst2src=shape_O,
                t_dst2src=shape_t,
                s_dst2src=shape_s,
            )
            shape_coeff_chunks.append(shape_coeff_chunk.to(torch.device("cpu")))
            shape_rows_src_chunks.append(shape_rows_src_chunk.cpu())
            shape_rows_dst_chunks.append(shape_rows_dst_chunk.cpu())
            del shape_coeff_chunk
            del shape_coords_src, shape_coords_dst, shape_O, shape_t, shape_s
            del shape_rows_src_chunk, shape_rows_dst_chunk, chunk_relations

        shape_mapper.cpu()
        torch.cuda.empty_cache()

        shape_coeff = concat_coeff(shape_coeff_chunks).to(torch.device(DEVICE))
        shape_rows_src = torch.cat(shape_rows_src_chunks).to(DEVICE)
        shape_rows_dst = torch.cat(shape_rows_dst_chunks).to(DEVICE)
        del shape_coeff_chunks, shape_rows_src_chunks, shape_rows_dst_chunks

        shape_projector = SymmetryProjector(
            num_rows=occ_coordinates.shape[0],
            rows_src=shape_rows_src,
            rows_dst=shape_rows_dst,
            coeff=shape_coeff,
        )
        shape_view = TRELLIS2ShapeLatentView(
            coords=occ_coordinates,
            sp_class=SparseTensor,
        )

        shape_flow_model.to(DEVICE)
        condition_512 = condition_512.to(DEVICE)
        neg_condition_512 = torch.zeros_like(condition_512)

        noise_sampler = SymmetryProjectionNoiseSampler(
            sampler=TRELLIS2ShapeLatentNoiseSampler(),
            symmetry_strength=noise_symmetry_projection_strength,
        )
        shape_noise = noise_sampler.sample(
            sp_class=SparseTensor,
            coords=occ_coordinates,
            feat_dim=32,
            grid_size=32,
            seed=seed,
            device=DEVICE,
            projector=shape_projector,
            to_sparse_view=shape_view.to_sparse_view,
            to_original_view=shape_view.to_original_view,
            self_include=True,
        )

        flow_predictor = TRELLIS2FlowPredictor(model=shape_flow_model)
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
            rescale=0.0,
        )

        flow_solver = EulerSolver()
        shape_latent = shape_noise

        await emit({"type": "progress", "progress": 0.0})
        for step_index, step in enumerate(
            flow_solver.iter_steps(
                noise=shape_noise,
                predictor=spg_predictor,
                steps=steps,
                predictor_args={
                    "cond": condition_512,
                    "neg_cond": neg_condition_512,
                    "projector": shape_projector,
                    "to_sparse_view": shape_view.to_sparse_view,
                    "to_original_view": shape_view.to_original_view,
                    "self_include": True,
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
        del shape_projector, shape_coeff
        del shape_rows_src, shape_rows_dst, shape_view
        torch.cuda.empty_cache()

        if mode == "512":
            shape_latent_grid_size = 32
            shape_grid_size = 512
        else:
            condition_1024 = torch.load(inputs.paths["image_condition_1024"], map_location="cpu")

            shape_decoder = self.loader.shape_decoder
            shape_decoder.to(DEVICE)

            hr_coords = shape_decoder.upsample(shape_latent, upsample_times=4)

            shape_latent = shape_latent.cpu()
            shape_decoder.cpu()
            torch.cuda.empty_cache()

            cascade_coordinates = hr_coords
            shape_latent_grid_size = 64
            shape_grid_size = 1024

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

            shape_mapper = self.loader.shape_mapper
            shape_mapper.to(DEVICE)

            shape_coeff_chunks = []
            shape_rows_src_chunks = []
            shape_rows_dst_chunks = []

            for relation_start in range(0, O_dst2src.shape[0], SHAPE_MAPPER_RELATION_CHUNK_SIZE):
                relation_end = relation_start + SHAPE_MAPPER_RELATION_CHUNK_SIZE
                chunk_relations = [
                    (
                        O_dst2src[relation_start:relation_end],
                        t_dst2src[relation_start:relation_end],
                        s_dst2src[relation_start:relation_end],
                    )
                    for _ in range(condition_1024.shape[0])
                ]

                shape_coords_src, shape_coords_dst, shape_rows_src_chunk, shape_rows_dst_chunk, shape_O, shape_t, shape_s = build_symmetry_relation_inputs(
                    coords=cascade_coordinates,
                    relations=chunk_relations,
                    grid_size=shape_latent_grid_size,
                )
                shape_coeff_chunk = shape_mapper(
                    coords_src=shape_coords_src,
                    coords_dst=shape_coords_dst,
                    O_dst2src=shape_O,
                    t_dst2src=shape_t,
                    s_dst2src=shape_s,
                )
                shape_coeff_chunks.append(shape_coeff_chunk.to(torch.device("cpu")))
                shape_rows_src_chunks.append(shape_rows_src_chunk.cpu())
                shape_rows_dst_chunks.append(shape_rows_dst_chunk.cpu())
                del shape_coeff_chunk
                del shape_coords_src, shape_coords_dst, shape_O, shape_t, shape_s
                del shape_rows_src_chunk, shape_rows_dst_chunk, chunk_relations

            shape_mapper.cpu()
            torch.cuda.empty_cache()

            shape_coeff = concat_coeff(shape_coeff_chunks).to(torch.device(DEVICE))
            shape_rows_src = torch.cat(shape_rows_src_chunks).to(DEVICE)
            shape_rows_dst = torch.cat(shape_rows_dst_chunks).to(DEVICE)
            del shape_coeff_chunks, shape_rows_src_chunks, shape_rows_dst_chunks

            shape_projector = SymmetryProjector(
                num_rows=cascade_coordinates.shape[0],
                rows_src=shape_rows_src,
                rows_dst=shape_rows_dst,
                coeff=shape_coeff,
            )
            shape_view = TRELLIS2ShapeLatentView(
                coords=cascade_coordinates,
                sp_class=SparseTensor,
            )

            shape_flow_model = self.loader.shape_flow_model_1024
            shape_flow_model.to(DEVICE)
            condition_1024 = condition_1024.to(DEVICE)
            neg_condition_1024 = torch.zeros_like(condition_1024)

            noise_sampler = SymmetryProjectionNoiseSampler(
                sampler=TRELLIS2ShapeLatentNoiseSampler(),
                symmetry_strength=noise_symmetry_projection_strength,
            )
            shape_noise = noise_sampler.sample(
                sp_class=SparseTensor,
                coords=cascade_coordinates,
                feat_dim=32,
                grid_size=shape_latent_grid_size,
                seed=seed,
                device=DEVICE,
                projector=shape_projector,
                to_sparse_view=shape_view.to_sparse_view,
                to_original_view=shape_view.to_original_view,
                self_include=True,
            )

            flow_predictor = TRELLIS2FlowPredictor(model=shape_flow_model)
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
                rescale=0.0,
            )

            flow_solver = EulerSolver()
            normalized_shape_latent = shape_noise

            await emit({"type": "progress", "progress": 0.0})
            for step_index, step in enumerate(
                flow_solver.iter_steps(
                    noise=shape_noise,
                    predictor=spg_predictor,
                    steps=steps,
                    predictor_args={
                        "cond": condition_1024,
                        "neg_cond": neg_condition_1024,
                        "projector": shape_projector,
                        "to_sparse_view": shape_view.to_sparse_view,
                        "to_original_view": shape_view.to_original_view,
                        "self_include": True,
                    },
                    sigma_min=1e-5,
                    rescale_t=time_step_rescale,
                ),
            ):
                normalized_shape_latent = step.x_t
                if step_index > 0:
                    await emit({"type": "progress", "progress": float(step.t)})

            shape_latent = SparseTensor(
                feats=trellis2_shape_latent_to_sparse_view(normalized_shape_latent),
                coords=normalized_shape_latent.coords,
            )

            shape_flow_model.cpu()
            condition_1024 = condition_1024.cpu()
            neg_condition_1024 = neg_condition_1024.cpu()
            del shape_projector, shape_coeff
            del shape_rows_src, shape_rows_dst, shape_view
            torch.cuda.empty_cache()

        shape_decoder = self.loader.shape_decoder
        shape_decoder.set_resolution(shape_grid_size)
        shape_decoder.to(DEVICE)

        shape_meshes, shape_subs = shape_decoder(shape_latent, return_subs=True)
        shape_mesh = shape_meshes[0]

        shape_latent = shape_latent.cpu()
        shape_raw_mesh_vertices = shape_mesh.vertices.cpu().contiguous()
        shape_raw_mesh_faces = shape_mesh.faces.cpu().contiguous()
        shape_decoder.cpu()
        torch.cuda.empty_cache()

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
                        "coords": sub.coords.cpu().contiguous(),
                        "feats": sub.feats.cpu().contiguous(),
                    }
                    for sub in shape_subs
                ],
            },
            shape_raw_mesh_path,
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
