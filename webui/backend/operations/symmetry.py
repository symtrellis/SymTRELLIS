from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import trimesh
from pytorch3d.structures import Meshes

from symtrellis.detection.detectors import (
    detect_c2_axes_perpendicular_to_axis,
    detect_reflection_planes,
    detect_reflection_planes_containing_axis,
    detect_reflection_planes_perpendicular_to_axis,
    detect_rotation_axes,
)
from symtrellis.detection.intrinsic import build_intrinsic_basis
from symtrellis.detection.sampling import sample_mesh_farthest_points

from ..loaders.trellis2 import DEVICE
from . import Operation, OperationContext, OperationInputs, OperationResult
from .trellis2_vanilla_shape import SHAPE_VISUALIZATION_MESH, Trellis2VanillaShape

DETECTION_NUM_SAMPLES = 8192
DETECTION_INTRINSIC_DIM = 64
DETECTION_NUM_ICP_ITER = 128
DETECTION_NUM_ICP_INIT = 512
DETECTION_MAX_FOLD = 30


def load_detection_basis(
    mesh_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mesh = trimesh.load_mesh(mesh_path, file_type="glb")
    vertices_y_up = torch.as_tensor(
        mesh.vertices,
        dtype=torch.float32,
        device=device,
    )
    vertices = torch.stack(
        [
            vertices_y_up[:, 0],
            -vertices_y_up[:, 2],
            vertices_y_up[:, 1],
        ],
        dim=1,
    )
    faces = torch.as_tensor(
        mesh.faces,
        dtype=torch.long,
        device=device,
    )
    samples = sample_mesh_farthest_points(
        Meshes(vertices[None], faces[None]),
        num_points=DETECTION_NUM_SAMPLES,
    )
    samples, phi, Gphi = build_intrinsic_basis(
        verts=vertices,
        faces=faces,
        samples=samples,
        intrinsic_dim=DETECTION_INTRINSIC_DIM,
    )
    return samples, phi, Gphi


class DetectRotationSymmetry(Operation):
    operation_id = "symmetry.detect_rotation_symmetry"
    execution_kind = "action"
    queue_kind = "gpu"

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        source_record = coordinator.storage.read_node_run(request.source_node_run_key)
        if source_record["operation_id"] != Trellis2VanillaShape.operation_id:
            raise ValueError(
                f"{self.operation_id} requires {Trellis2VanillaShape.operation_id} as its source",
            )

        shape_visualization_mesh = source_record["outputs"][SHAPE_VISUALIZATION_MESH]
        return OperationInputs(
            records={"vanilla_shape": source_record},
            paths={SHAPE_VISUALIZATION_MESH: Path(shape_visualization_mesh["path"])},
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any] | None,
    ) -> OperationResult:
        device = torch.device(DEVICE)
        samples, phi, Gphi = load_detection_basis(
            inputs.paths[SHAPE_VISUALIZATION_MESH],
            device,
        )

        candidates = detect_rotation_axes(
            samples=samples,
            phi=phi,
            Gphi=Gphi,
            num_icp_iter=DETECTION_NUM_ICP_ITER,
            num_icp_init=DETECTION_NUM_ICP_INIT,
            max_fold=DETECTION_MAX_FOLD,
        )

        del samples, phi, Gphi
        torch.cuda.empty_cache()

        return OperationResult(json_result=candidates)


class DetectReflectionPlanes(Operation):
    operation_id = "symmetry.detect_reflection_planes"
    execution_kind = "action"
    queue_kind = "gpu"

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        source_record = coordinator.storage.read_node_run(request.source_node_run_key)
        if source_record["operation_id"] != Trellis2VanillaShape.operation_id:
            raise ValueError(
                f"{self.operation_id} requires {Trellis2VanillaShape.operation_id} as its source",
            )

        shape_visualization_mesh = source_record["outputs"][SHAPE_VISUALIZATION_MESH]
        return OperationInputs(
            records={"vanilla_shape": source_record},
            paths={SHAPE_VISUALIZATION_MESH: Path(shape_visualization_mesh["path"])},
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any] | None,
    ) -> OperationResult:
        device = torch.device(DEVICE)
        samples, phi, Gphi = load_detection_basis(
            inputs.paths[SHAPE_VISUALIZATION_MESH],
            device,
        )

        candidates = detect_reflection_planes(
            samples=samples,
            phi=phi,
            Gphi=Gphi,
            num_icp_iter=DETECTION_NUM_ICP_ITER,
            num_icp_init=DETECTION_NUM_ICP_INIT,
        )

        del samples, phi, Gphi
        torch.cuda.empty_cache()

        return OperationResult(json_result=candidates)


class DetectFinerSymmetry(Operation):
    operation_id = "symmetry.detect_finer_symmetry"
    execution_kind = "action"
    queue_kind = "gpu"

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        source_record = coordinator.storage.read_node_run(request.source_node_run_key)
        if source_record["operation_id"] != Trellis2VanillaShape.operation_id:
            raise ValueError(
                f"{self.operation_id} requires {Trellis2VanillaShape.operation_id} as its source",
            )

        shape_visualization_mesh = source_record["outputs"][SHAPE_VISUALIZATION_MESH]
        return OperationInputs(
            records={"vanilla_shape": source_record},
            paths={SHAPE_VISUALIZATION_MESH: Path(shape_visualization_mesh["path"])},
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any] | None,
    ) -> OperationResult:
        device = torch.device(DEVICE)
        samples, phi, Gphi = load_detection_basis(
            inputs.paths[SHAPE_VISUALIZATION_MESH],
            device,
        )

        major_axis = torch.tensor(
            params["majorAxis"],
            dtype=samples.dtype,
            device=device,
        )
        center = torch.tensor(
            params["center"],
            dtype=samples.dtype,
            device=device,
        )

        reflection_planes_containing_axis = detect_reflection_planes_containing_axis(
            samples=samples,
            phi=phi,
            Gphi=Gphi,
            axis=major_axis,
            q=center,
            num_icp_iter=DETECTION_NUM_ICP_ITER,
            num_icp_init=DETECTION_NUM_ICP_INIT,
            max_fold=DETECTION_MAX_FOLD,
        )
        reflection_planes_perpendicular_to_axis = detect_reflection_planes_perpendicular_to_axis(
            samples=samples,
            phi=phi,
            Gphi=Gphi,
            axis=major_axis,
            q=center,
            num_icp_iter=DETECTION_NUM_ICP_ITER,
            num_icp_init=DETECTION_NUM_ICP_INIT,
        )
        c2_axes_perpendicular_to_axis = detect_c2_axes_perpendicular_to_axis(
            samples=samples,
            phi=phi,
            Gphi=Gphi,
            axis=major_axis,
            q=center,
            num_icp_iter=DETECTION_NUM_ICP_ITER,
            num_icp_init=DETECTION_NUM_ICP_INIT,
            max_fold=DETECTION_MAX_FOLD,
        )

        del samples, phi, Gphi, major_axis, center
        torch.cuda.empty_cache()

        return OperationResult(
            json_result={
                "c2_axes_perpendicular_to_axis": c2_axes_perpendicular_to_axis,
                "reflection_planes_containing_axis": reflection_planes_containing_axis,
                "reflection_planes_perpendicular_to_axis": reflection_planes_perpendicular_to_axis,
            },
        )


class ConfirmSymmetry(Operation):
    execution_kind = "node_run"
    queue_kind = "inline"

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        return OperationInputs()

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "symmetry": params["symmetry"],
        }

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any] | None,
    ) -> OperationResult:
        return OperationResult(
            json_result=params["symmetry"],
            metadata={
                "symmetry": params["symmetry"],
            },
        )


class ConfirmDetectedSymmetry(ConfirmSymmetry):
    operation_id = "symmetry.confirm_detected_tuple"


class ConfirmManualSymmetry(ConfirmSymmetry):
    operation_id = "symmetry.confirm_manual_tuple"
