import logging
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse

from .coordinator import Coordinator
from .loaders.trellis2 import TRELLIS2Loader
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
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

app = FastAPI(title="SymTRELLIS WebUI Backend")
storage = Storage(STORAGE_ROOT)
trellis2_loader = TRELLIS2Loader()
operations: dict[str, Any] = {
    Trellis2ImageCondition.operation_id: Trellis2ImageCondition(trellis2_loader),
    Trellis2VanillaSparseStructure.operation_id: Trellis2VanillaSparseStructure(trellis2_loader),
    Trellis2SymmetrySparseStructure.operation_id: Trellis2SymmetrySparseStructure(trellis2_loader),
    Trellis2VanillaShape.operation_id: Trellis2VanillaShape(trellis2_loader),
    Trellis2SymmetryShape.operation_id: Trellis2SymmetryShape(trellis2_loader),
    Trellis2Texture.operation_id: Trellis2Texture(trellis2_loader),
    Trellis2ExportGlb.operation_id: Trellis2ExportGlb(),
    DetectRotationSymmetry.operation_id: DetectRotationSymmetry(),
    DetectReflectionPlanes.operation_id: DetectReflectionPlanes(),
    DetectFinerSymmetry.operation_id: DetectFinerSymmetry(),
    ConfirmDetectedSymmetry.operation_id: ConfirmDetectedSymmetry(),
    ConfirmManualSymmetry.operation_id: ConfirmManualSymmetry(),
}
coordinator = Coordinator(storage=storage, operations=operations)
websockets: set[WebSocket] = set()


class ZipStreamWriter:
    def __init__(self):
        self.buffer = bytearray()

    def write(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def flush_bytes(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


def execution_request(payload: dict[str, Any], execution_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=payload.get("request_id", f"request_{uuid4().hex}"),
        execution_kind=execution_kind,
        operation_id=payload["operation_id"],
        model_id=payload.get("model_id"),
        session_id=payload.get("session_id"),
        parent_run_keys=payload.get("parent_run_keys", []),
        input_upload_keys=payload.get("input_upload_keys", []),
        source_node_run_key=payload.get("source_node_run_key"),
        params=payload.get("params", {}),
    )


async def emit(update: dict) -> None:
    for websocket in list(websockets):
        await websocket.send_json(update)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/uploads")
async def upload(file: UploadFile) -> dict:
    record = storage.save_upload(
        fileobj=file.file,
        filename=file.filename or "upload",
        mime_type=file.content_type or "application/octet-stream",
    )
    return {
        "upload_key": record["upload_key"],
        "content_hash": record["content_hash"],
        "filename": record["filename"],
        "mime_type": record["mime_type"],
    }


@app.post("/node-runs")
async def submit_node_run(payload: dict[str, Any]) -> Any:
    try:
        return await coordinator.submit_execution(execution_request(payload, "node_run"), emit)
    except Exception as error:
        logger.exception("Node run failed")
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(error),
                "operation_id": payload.get("operation_id"),
                "request_id": payload.get("request_id"),
            },
        ) from error


@app.post("/actions")
async def submit_action(payload: dict[str, Any]) -> Any:
    try:
        return await coordinator.submit_execution(execution_request(payload, "action"), emit)
    except Exception as error:
        logger.exception("Action failed")
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(error),
                "operation_id": payload.get("operation_id"),
                "request_id": payload.get("request_id"),
            },
        ) from error


@app.get("/sessions/{session_id}")
def restore_session(session_id: str, key: str | None = None) -> Any:
    restored = coordinator.restore_session(session_id=session_id, key=key)
    if restored is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return restored


@app.get("/node-runs/{node_run_key}/outputs/{role}")
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


@app.get("/actions/{action_key}/outputs/{role}")
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


@app.get("/actions/{action_key}/bundle")
def download_action_bundle(action_key: str) -> StreamingResponse:
    record = storage.read_action(action_key)
    if record is None:
        raise HTTPException(status_code=404, detail="Action not found")

    bundle = (record.get("json_result") or {}).get("bundle")
    if not isinstance(bundle, list):
        raise HTTPException(status_code=404, detail="Action bundle not found")

    files = []
    for item in bundle:
        source = item["source"]
        key = item["key"]
        role = item["role"]
        filename = item["filename"]

        if source == "node_run":
            source_record = storage.read_node_run(key)
        elif source == "action":
            source_record = storage.read_action(key)
        else:
            raise HTTPException(status_code=400, detail="Invalid bundle source")

        if source_record is None:
            raise HTTPException(status_code=404, detail="Bundle source not found")

        output = source_record["outputs"].get(role)
        if output is None:
            raise HTTPException(status_code=404, detail="Bundle output role not found")

        path = Path(output["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Bundle file not found")

        files.append({"path": path, "filename": filename})

    def stream_bundle():
        writer = ZipStreamWriter()

        with zipfile.ZipFile(writer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file["path"], arcname=file["filename"])
                data = writer.flush_bytes()
                if data:
                    yield data

        data = writer.flush_bytes()
        if data:
            yield data

    return StreamingResponse(
        stream_bundle(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{action_key}_bundle.zip"',
        },
    )


@app.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    websockets.add(websocket)
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            websockets.discard(websocket)
            return
