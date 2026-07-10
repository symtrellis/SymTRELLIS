import asyncio
from pathlib import Path
from typing import Any

import torch
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh

from inference.trellis2 import trelli2_mesh_to_glb

from ..loaders.trellis2 import DEVICE
from . import Emit, Operation, OperationContext, OperationInputs, OperationOutput, OperationResult
from .trellis2_image_condition import IMAGE_CONDITION_512, IMAGE_CONDITION_1024, IMAGE_PNG, Trellis2ImageCondition
from .trellis2_symmetry_shape import Trellis2SymmetryShape
from .trellis2_symmetry_sparse_structure import Trellis2SymmetrySparseStructure
from .trellis2_texture import PBR_VOXEL, TEXTURE_LATENT, Trellis2Texture
from .trellis2_vanilla_shape import SHAPE_LATENT, SHAPE_RAW_MESH, Trellis2VanillaShape
from .trellis2_vanilla_sparse_structure import SPARSE_STRUCTURE_LATENT, Trellis2VanillaSparseStructure

EXPORT_GLB = "glb"
SHAPE_OPERATION_IDS = (
    Trellis2VanillaShape.operation_id,
    Trellis2SymmetryShape.operation_id,
)
SPARSE_STRUCTURE_OPERATION_IDS = (
    Trellis2VanillaSparseStructure.operation_id,
    Trellis2SymmetrySparseStructure.operation_id,
)


class Trellis2ExportGlb(Operation):
    operation_id = "trellis2.export_glb"
    execution_kind = "action"
    queue_kind = "gpu"
    output_roles = (EXPORT_GLB,)

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        source_record = coordinator.storage.read_node_run(request.source_node_run_key)
        if source_record is None:
            raise ValueError(f"Export source node run not found: {request.source_node_run_key}")

        source_operation_id = source_record["operation_id"]
        if source_operation_id in SHAPE_OPERATION_IDS:
            shape_record = source_record
            texture_record = None
        elif source_operation_id == Trellis2Texture.operation_id:
            texture_record = source_record
            shape_record = None
            for run_key in reversed(source_record["ancestor_run_keys"]):
                run = coordinator.storage.read_node_run(run_key)
                if run is not None and run["operation_id"] in SHAPE_OPERATION_IDS:
                    shape_record = run
                    break
            if shape_record is None:
                raise ValueError("trellis2.export_glb requires a shape node before texture")
        else:
            raise ValueError(f"trellis2.export_glb cannot export from {source_operation_id}")

        sparse_structure_record = None
        for run_key in reversed(shape_record["ancestor_run_keys"]):
            run = coordinator.storage.read_node_run(run_key)
            if run is not None and run["operation_id"] in SPARSE_STRUCTURE_OPERATION_IDS:
                sparse_structure_record = run
                break
        if sparse_structure_record is None:
            raise ValueError("trellis2.export_glb requires a sparse structure node before shape")

        image_condition_record = None
        for run_key in reversed(source_record["ancestor_run_keys"]):
            run = coordinator.storage.read_node_run(run_key)
            if run is not None and run["operation_id"] == Trellis2ImageCondition.operation_id:
                image_condition_record = run
                break
        if image_condition_record is None:
            raise ValueError("trellis2.export_glb requires trellis2.image_condition")

        image_png = image_condition_record["outputs"].get(IMAGE_PNG)
        if image_png is None:
            raise ValueError("trellis2.image_condition did not produce image_png")

        image_condition_512 = image_condition_record["outputs"].get(IMAGE_CONDITION_512)
        if image_condition_512 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_512")

        image_condition_1024 = image_condition_record["outputs"].get(IMAGE_CONDITION_1024)
        if image_condition_1024 is None:
            raise ValueError("trellis2.image_condition did not produce image_condition_1024")

        sparse_structure_latent = sparse_structure_record["outputs"].get(SPARSE_STRUCTURE_LATENT)
        if sparse_structure_latent is None:
            raise ValueError(f"{sparse_structure_record['operation_id']} did not produce sparse_structure_latent")

        shape_latent = shape_record["outputs"].get(SHAPE_LATENT)
        if shape_latent is None:
            raise ValueError(f"{shape_record['operation_id']} did not produce shape_latent")

        shape_raw_mesh = shape_record["outputs"].get(SHAPE_RAW_MESH)
        if shape_raw_mesh is None:
            raise ValueError(f"{shape_record['operation_id']} did not produce shape_raw_mesh")

        records = {
            "source": source_record,
            "image_condition": image_condition_record,
            "sparse_structure": sparse_structure_record,
            "shape": shape_record,
        }
        paths = {
            IMAGE_PNG: Path(image_png["path"]),
            IMAGE_CONDITION_512: Path(image_condition_512["path"]),
            IMAGE_CONDITION_1024: Path(image_condition_1024["path"]),
            SPARSE_STRUCTURE_LATENT: Path(sparse_structure_latent["path"]),
            SHAPE_LATENT: Path(shape_latent["path"]),
            SHAPE_RAW_MESH: Path(shape_raw_mesh["path"]),
        }

        if texture_record is not None:
            texture_latent = texture_record["outputs"].get(TEXTURE_LATENT)
            if texture_latent is None:
                raise ValueError("trellis2.texture.generate did not produce texture_latent")

            pbr_voxel = texture_record["outputs"].get(PBR_VOXEL)
            if pbr_voxel is None:
                raise ValueError("trellis2.texture.generate did not produce pbr_voxel")

            records["texture"] = texture_record
            paths[TEXTURE_LATENT] = Path(texture_latent["path"])
            paths[PBR_VOXEL] = Path(pbr_voxel["path"])

        return OperationInputs(records=records, paths=paths)

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        texture_record = inputs.records.get("texture")

        return {
            "source_node_run_key": inputs.records["source"]["node_run_key"],
            "source_operation_id": inputs.records["source"]["operation_id"],
            "image_condition_node_run_key": inputs.records["image_condition"]["node_run_key"],
            "image_condition_roles": [
                IMAGE_PNG,
                IMAGE_CONDITION_512,
                IMAGE_CONDITION_1024,
            ],
            "sparse_structure_node_run_key": inputs.records["sparse_structure"]["node_run_key"],
            "sparse_structure_operation_id": inputs.records["sparse_structure"]["operation_id"],
            "sparse_structure_roles": [SPARSE_STRUCTURE_LATENT],
            "shape_node_run_key": inputs.records["shape"]["node_run_key"],
            "shape_operation_id": inputs.records["shape"]["operation_id"],
            "shape_roles": [
                SHAPE_LATENT,
                SHAPE_RAW_MESH,
            ],
            "texture_node_run_key": texture_record["node_run_key"] if texture_record is not None else None,
            "texture_operation_id": texture_record["operation_id"] if texture_record is not None else None,
            "texture_roles": (
                [
                    TEXTURE_LATENT,
                    PBR_VOXEL,
                ]
                if texture_record is not None
                else []
            ),
        }

    async def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        emit: Emit,
    ) -> OperationResult:
        face_decimation_target = int(params["faceDecimationTarget"])
        remesh = bool(params["remesh"])
        remesh_band = float(params["remeshBand"])
        remesh_project = float(params["remeshProject"])

        has_texture = PBR_VOXEL in inputs.paths
        texture_size = int(params["textureSize"]) if has_texture else None

        shape_metadata = inputs.records["shape"]["outputs"][SHAPE_RAW_MESH]["metadata"]
        o_voxel_grid_size = int(shape_metadata["oVoxelGridSize"])

        shape_raw_mesh_payload = torch.load(inputs.paths[SHAPE_RAW_MESH], map_location="cpu")
        shape_mesh = Mesh(
            vertices=shape_raw_mesh_payload["vertices"],
            faces=shape_raw_mesh_payload["faces"],
        )

        pbr_voxel = None
        if has_texture:
            pbr_voxel_payload = torch.load(inputs.paths[PBR_VOXEL], map_location="cpu")
            pbr_voxel = SparseTensor(
                coords=pbr_voxel_payload["coords"].to(DEVICE),
                feats=pbr_voxel_payload["feats"].to(DEVICE),
            )

        glb_path = context.work_dir / "model.glb"

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

        await emit({"type": "progress", "progress": 0.0, "stage": "export_start"})
        glb_mesh = await asyncio.to_thread(
            trelli2_mesh_to_glb,
            shape_mesh=shape_mesh,
            res=o_voxel_grid_size,
            device=torch.device(DEVICE),
            texture_size=texture_size,
            pbr_voxel=pbr_voxel,
            remesh=remesh,
            decimation_target=face_decimation_target,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            report=report_glb,
        )
        glb_mesh.export(glb_path)

        pbr_voxel = None
        torch.cuda.empty_cache()

        bundle_root = f"{context.key}_bundle"
        image_condition_key = inputs.records["image_condition"]["node_run_key"]
        sparse_structure_key = inputs.records["sparse_structure"]["node_run_key"]
        shape_key = inputs.records["shape"]["node_run_key"]

        bundle = [
            {
                "source": "node_run",
                "key": image_condition_key,
                "role": IMAGE_PNG,
                "filename": f"{bundle_root}/image.png",
            },
            {
                "source": "node_run",
                "key": image_condition_key,
                "role": IMAGE_CONDITION_512,
                "filename": f"{bundle_root}/image_condition_512.pt",
            },
            {
                "source": "node_run",
                "key": image_condition_key,
                "role": IMAGE_CONDITION_1024,
                "filename": f"{bundle_root}/image_condition_1024.pt",
            },
            {
                "source": "node_run",
                "key": sparse_structure_key,
                "role": SPARSE_STRUCTURE_LATENT,
                "filename": f"{bundle_root}/sparse_structure_latent.pt",
            },
            {
                "source": "node_run",
                "key": shape_key,
                "role": SHAPE_LATENT,
                "filename": f"{bundle_root}/shape_latent.pt",
            },
            {
                "source": "node_run",
                "key": shape_key,
                "role": SHAPE_RAW_MESH,
                "filename": f"{bundle_root}/shape_raw_mesh.pt",
            },
        ]

        texture_record = inputs.records.get("texture")
        if texture_record is not None:
            texture_key = texture_record["node_run_key"]
            bundle.extend(
                [
                    {
                        "source": "node_run",
                        "key": texture_key,
                        "role": TEXTURE_LATENT,
                        "filename": f"{bundle_root}/texture_latent.pt",
                    },
                    {
                        "source": "node_run",
                        "key": texture_key,
                        "role": PBR_VOXEL,
                        "filename": f"{bundle_root}/pbr_voxel.pt",
                    },
                ],
            )

        bundle.append(
            {
                "source": "action",
                "key": context.key,
                "role": EXPORT_GLB,
                "filename": f"{bundle_root}/model.glb",
            },
        )

        metadata = {
            "faceDecimationTarget": face_decimation_target,
            "hasTexture": has_texture,
            "oVoxelGridSize": o_voxel_grid_size,
            "remesh": remesh,
            "remeshBand": remesh_band,
            "remeshProject": remesh_project,
            "textureSize": texture_size,
        }

        return OperationResult(
            outputs=[
                OperationOutput(
                    role=EXPORT_GLB,
                    path=glb_path,
                    filename="model.glb",
                    metadata=metadata,
                ),
            ],
            metadata=metadata,
            json_result={
                "bundle": bundle,
            },
        )
