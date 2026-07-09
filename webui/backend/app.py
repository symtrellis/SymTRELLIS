import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse

from .coordinator import Coordinator
from .loaders.trellis2 import TRELLIS2Loader
from .operations.trellis2_image_condition import Trellis2ImageCondition
from .operations.trellis2_vanilla_sparse_structure import Trellis2VanillaSparseStructure
from .storage import Storage

STORAGE_ROOT = Path(os.environ.get("SYMTRELLIS_WEBUI_STORAGE_ROOT", "/tmp/symtrellis_webui"))

torch.set_grad_enabled(False)

app = FastAPI(title="SymTRELLIS WebUI Backend")
storage = Storage(STORAGE_ROOT)
trellis2_loader = TRELLIS2Loader()
operations: dict[str, Any] = {
    Trellis2ImageCondition.operation_id: Trellis2ImageCondition(trellis2_loader),
    Trellis2VanillaSparseStructure.operation_id: Trellis2VanillaSparseStructure(trellis2_loader),
}
coordinator = Coordinator(storage=storage, operations=operations)
websockets: set[WebSocket] = set()


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
    return await coordinator.submit_execution(execution_request(payload, "node_run"), emit)


@app.post("/actions")
async def submit_action(payload: dict[str, Any]) -> Any:
    return await coordinator.submit_execution(execution_request(payload, "action"), emit)


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


@app.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    websockets.add(websocket)
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            websockets.discard(websocket)
            return
