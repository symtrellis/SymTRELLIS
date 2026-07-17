import asyncio
import logging
import os
import time
import zipfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio
import spaces
import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .coordinator import Coordinator, ExecutionRequest, GpuExecutionInput
from .loaders.trellis2 import TRELLIS2Runtime
from .operations import OperationContext, OperationResult
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
from .storage import SessionExpiredError, Storage

STORAGE_ROOT = Path(os.environ.get("SYMTRELLIS_WEBUI_STORAGE_ROOT", "/tmp/symtrellis_webui"))
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
SESSION_TIMEOUT_SECONDS = int(os.environ.get("SYMTRELLIS_WEBUI_SESSION_TIMEOUT_SECONDS", "3600"))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("SYMTRELLIS_WEBUI_CLEANUP_INTERVAL_SECONDS", "30"))
PREPARED_RESERVATION_SECONDS = 120
RUNTIME_ID = uuid4().hex
LOGGER = logging.getLogger(__name__)

torch.set_grad_enabled(False)

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
coordinator = Coordinator(
    storage=storage,
    operations=operations,
    runtime_id=RUNTIME_ID,
    prepared_reservation_seconds=PREPARED_RESERVATION_SECONDS,
)


async def cleanup_storage_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(
                storage.cleanup_expired_sessions,
                time.time(),
                SESSION_TIMEOUT_SECONDS,
            )
        except Exception:
            LOGGER.exception("SymTRELLIS storage cleanup failed")


@asynccontextmanager
async def server_lifespan(_server: Any):
    await asyncio.to_thread(storage.fail_stale_runtime_tasks, RUNTIME_ID, time.time())
    await asyncio.to_thread(
        storage.cleanup_expired_sessions,
        time.time(),
        SESSION_TIMEOUT_SECONDS,
    )
    cleanup_task = asyncio.create_task(cleanup_storage_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


server = gradio.Server(
    title="SymTRELLIS WebUI Backend",
    lifespan=server_lifespan,
)


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
def upload(
    file: gradio.FileData,
    filename: str,
    mime_type: str,
) -> dict[str, Any]:
    file = gradio.FileData.model_validate(file)

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
    try:
        return coordinator.prepare(request)
    except SessionExpiredError:
        return {"status": "session_expired"}


def gpu_duration(gpu_input: GpuExecutionInput, _progress: gradio.Progress) -> int:
    operation_id = gpu_input.operation_id

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
        source_record = gpu_input.inputs.records["source"]
        if source_record["metadata"]["oVoxelGridSize"] == 512:
            return 120
        return 180

    steps = int(gpu_input.params["steps"])
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
        if gpu_input.params["mode"] == "cascade":
            return 105 + 2 * extra_steps
        return 24 + extra_steps

    if operation_id == Trellis2Texture.operation_id:
        shape_record = gpu_input.inputs.records["shape"]
        if shape_record["metadata"]["shapeLatentGridSize"] > 32:
            return 75 + 2 * extra_steps
        return 35 + extra_steps

    raise ValueError(f"Unknown GPU operation_id: {operation_id}")


@spaces.GPU(duration=gpu_duration)
@torch.inference_mode()
def run_gpu(
    gpu_input: GpuExecutionInput,
    progress: gradio.Progress,
) -> OperationResult:
    operation = operations[gpu_input.operation_id]
    context = OperationContext(
        key=gpu_input.key,
        work_dir=gpu_input.work_dir,
    )
    return operation.run(
        gpu_input.inputs,
        gpu_input.params,
        context,
        progress,
    )


@server.api(name="mark_execution_queued", queue=False)
def mark_execution_queued(task_id: str, event_id: str) -> dict[str, bool]:
    storage.mark_execution_queued(
        task_id=task_id,
        event_id=event_id,
        queue_lease_seconds=SESSION_TIMEOUT_SECONDS,
    )
    return {"ok": True}


@server.api(name="renew_execution_queue", queue=False)
def renew_execution_queue(task_id: str, event_id: str) -> dict[str, bool]:
    queued = storage.renew_execution_queue(
        task_id=task_id,
        event_id=event_id,
        queue_lease_seconds=SESSION_TIMEOUT_SECONDS,
    )
    return {"queued": queued}


@server.api(
    name="execute_execution",
    queue=True,
    concurrency_id="symtrellis-gpu",
    concurrency_limit=1,
)
def execute_execution(task_id: str) -> dict[str, Any]:
    progress = gradio.Progress()
    try:
        return coordinator.execute(task_id, progress, run_gpu)
    except SessionExpiredError:
        return {"status": "session_expired"}


@server.api(name="restore_session", queue=False)
def restore_session(
    session_id: str,
    key: str | None = None,
) -> dict[str, Any]:
    return coordinator.restore_session(session_id=session_id, key=key)


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

        try:
            with zipfile.ZipFile(
                temp_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
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
        finally:
            temp_path.unlink(missing_ok=True)

    return FileResponse(bundle_path, filename=f"{action_key}_bundle.zip")
