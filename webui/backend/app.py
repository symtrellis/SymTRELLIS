import os
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio
import spaces
import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .coordinator import Coordinator, ExecutionRequest
from .loaders.trellis2 import TRELLIS2Runtime
from .operations.symmetry import (
    ConfirmDetectedSymmetry,
    ConfirmManualSymmetry,
    DetectFinerSymmetry,
    DetectReflectionPlanes,
    DetectRotationSymmetry,
)
from .operations.trellis2_export_glb import Trellis2ExportGlb
from .operations.trellis2_image_condition import Trellis2ImageCondition
from .operations.trellis2_symmetry_shape import Trellis2SymmetryShape
from .operations.trellis2_symmetry_sparse_structure import Trellis2SymmetrySparseStructure
from .operations.trellis2_texture import Trellis2Texture
from .operations.trellis2_vanilla_shape import Trellis2VanillaShape
from .operations.trellis2_vanilla_sparse_structure import Trellis2VanillaSparseStructure
from .storage import Storage

STORAGE_ROOT = Path(os.environ.get("SYMTRELLIS_WEBUI_STORAGE_ROOT", "/tmp/symtrellis_webui"))
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

torch.set_grad_enabled(False)

server = gradio.Server(title="SymTRELLIS WebUI Backend")
storage = Storage(STORAGE_ROOT)
trellis2_runtime = TRELLIS2Runtime()
operations: dict[str, Any] = {
    Trellis2ImageCondition.operation_id: Trellis2ImageCondition(trellis2_runtime),
    Trellis2VanillaSparseStructure.operation_id: Trellis2VanillaSparseStructure(trellis2_runtime),
    Trellis2SymmetrySparseStructure.operation_id: Trellis2SymmetrySparseStructure(trellis2_runtime),
    Trellis2VanillaShape.operation_id: Trellis2VanillaShape(trellis2_runtime),
    Trellis2SymmetryShape.operation_id: Trellis2SymmetryShape(trellis2_runtime),
    Trellis2Texture.operation_id: Trellis2Texture(trellis2_runtime),
    Trellis2ExportGlb.operation_id: Trellis2ExportGlb(),
    DetectRotationSymmetry.operation_id: DetectRotationSymmetry(),
    DetectReflectionPlanes.operation_id: DetectReflectionPlanes(),
    DetectFinerSymmetry.operation_id: DetectFinerSymmetry(),
    ConfirmDetectedSymmetry.operation_id: ConfirmDetectedSymmetry(),
    ConfirmManualSymmetry.operation_id: ConfirmManualSymmetry(),
}
coordinator = Coordinator(storage=storage, operations=operations)


@server.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@server.get("/")
def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIST / "index.html")


server.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST / "assets"),
)


@server.api(name="upload", queue=False)
def upload(file: gradio.FileData) -> dict[str, Any]:
    file = gradio.FileData.model_validate(file)
    filename = file.orig_name or "upload"
    mime_type = file.mime_type or "application/octet-stream"

    with Path(file.path).open("rb") as fileobj:
        record = storage.save_upload(
            fileobj=fileobj,
            filename=filename,
            mime_type=mime_type,
        )

    return {
        "upload_key": record["upload_key"],
        "content_hash": record["content_hash"],
        "filename": record["filename"],
        "mime_type": record["mime_type"],
    }


@server.api(name="prepare_execution", queue=False)
def prepare_execution(payload: dict[str, Any]) -> dict[str, Any]:
    request = ExecutionRequest(**payload)
    return coordinator.prepare(request)


def gpu_duration(request: ExecutionRequest, _progress: gradio.Progress) -> int:
    operation_id = request.operation_id

    if operation_id == Trellis2ImageCondition.operation_id:
        return 10

    if operation_id in (
        DetectRotationSymmetry.operation_id,
        DetectReflectionPlanes.operation_id,
    ):
        return 20

    if operation_id == DetectFinerSymmetry.operation_id:
        return 40

    if operation_id == Trellis2ExportGlb.operation_id:
        source_record = storage.read_node_run(request.source_node_run_key)
        if source_record["metadata"]["oVoxelGridSize"] == 512:
            return 120
        return 180

    steps = int(request.params["steps"])
    extra_steps = max(0, steps - 32)

    if operation_id in (
        Trellis2VanillaSparseStructure.operation_id,
        Trellis2SymmetrySparseStructure.operation_id,
    ):
        return 24 + extra_steps

    if operation_id in (
        Trellis2VanillaShape.operation_id,
        Trellis2SymmetryShape.operation_id,
    ):
        if request.params["mode"] == "cascade":
            return 105 + 2 * extra_steps
        return 24 + extra_steps

    if operation_id == Trellis2Texture.operation_id:
        shape_record = storage.read_node_run(request.parent_run_keys[-1])
        if shape_record["metadata"]["shapeLatentGridSize"] > 32:
            return 75 + 2 * extra_steps
        return 35 + extra_steps

    raise ValueError(f"Unknown GPU operation_id: {operation_id}")


@spaces.GPU(duration=gpu_duration)
@torch.inference_mode()
def run_gpu(
    request: ExecutionRequest,
    progress: gradio.Progress,
) -> dict[str, Any]:
    return coordinator.execute_gpu(request, progress)


@server.api(
    name="execute_execution",
    queue=True,
    concurrency_id="symtrellis-gpu",
    concurrency_limit=1,
)
def execute_execution(payload: dict[str, Any]) -> dict[str, Any]:
    request = ExecutionRequest(**payload)
    progress = gradio.Progress()
    return run_gpu(request, progress)


@server.api(name="restore_session", queue=False)
def restore_session(
    session_id: str,
    key: str | None = None,
) -> dict[str, Any]:
    restored = coordinator.restore_session(session_id=session_id, key=key)
    if restored is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return restored


@server.get("/node-runs/{node_run_key}/outputs/{role}")
def download_node_run_output(node_run_key: str, role: str) -> FileResponse:
    record = storage.read_node_run(node_run_key)
    if record is None:
        raise HTTPException(status_code=404, detail="Node run not found")

    output = record["outputs"].get(role)
    if output is None:
        raise HTTPException(status_code=404, detail="Output role not found")

    path = Path(output["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(path, filename=output["filename"])


@server.get("/actions/{action_key}/outputs/{role}")
def download_action_output(action_key: str, role: str) -> FileResponse:
    record = storage.read_action(action_key)
    if record is None:
        raise HTTPException(status_code=404, detail="Action not found")

    output = record["outputs"].get(role)
    if output is None:
        raise HTTPException(status_code=404, detail="Output role not found")

    path = Path(output["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(path, filename=output["filename"])


@server.get("/actions/{action_key}/bundle")
def download_action_bundle(action_key: str) -> FileResponse:
    record = storage.read_action(action_key)
    if record is None:
        raise HTTPException(status_code=404, detail="Action not found")

    bundle = record["json_result"]["bundle"]
    bundle_path = storage.bundles_dir / f"{action_key}.zip"

    if not bundle_path.exists():
        temp_path = storage.bundles_dir / f"{action_key}.{uuid4().hex}.zip.tmp"

        with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in bundle:
                if item["source"] == "node_run":
                    source_record = storage.read_node_run(item["key"])
                elif item["source"] == "action":
                    source_record = storage.read_action(item["key"])
                else:
                    raise HTTPException(status_code=400, detail="Invalid bundle source")

                if source_record is None:
                    raise HTTPException(status_code=404, detail="Bundle source not found")

                output = source_record["outputs"].get(item["role"])
                if output is None:
                    raise HTTPException(status_code=404, detail="Bundle output role not found")

                path = Path(output["path"])
                if not path.exists():
                    raise HTTPException(status_code=404, detail="Bundle file not found")

                archive.write(path, arcname=item["filename"])

        os.replace(temp_path, bundle_path)

    return FileResponse(bundle_path, filename=f"{action_key}_bundle.zip")
