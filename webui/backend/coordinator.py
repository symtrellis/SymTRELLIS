import asyncio
import hashlib
import json
import shutil
from functools import partial
from typing import Any
from uuid import uuid4

from .operations import OperationContext


class Coordinator:
    def __init__(self, storage: Any, operations: dict[str, Any]):
        self.storage = storage
        self.operations = operations
        self.gpu_lock = asyncio.Lock()

    async def submit_execution(self, request: Any, emit: Any) -> Any:
        operation = self.operations.get(request.operation_id)
        if operation is None:
            raise ValueError(f"Unknown operation_id: {request.operation_id}")
        if operation.execution_kind != request.execution_kind:
            raise ValueError(
                f"Operation {operation.operation_id} expects {operation.execution_kind}, "
                f"got {request.execution_kind}",
            )

        if operation.creates_session:
            if request.session_id:
                session = self.storage.read_session(request.session_id)
                if session is None:
                    raise ValueError(f"Session not found: {request.session_id}")
            else:
                session = {
                    "session_id": f"session_{uuid4().hex}",
                    "model_id": request.model_id,
                    "active_run_keys": [],
                    "actions_by_source": {},
                }
        else:
            if not request.session_id:
                raise ValueError(f"Operation {operation.operation_id} requires session_id")
            session = self.storage.read_session(request.session_id)
            if session is None:
                raise ValueError(f"Session not found: {request.session_id}")

        if request.execution_kind == "node_run":
            if request.model_id != session["model_id"]:
                raise ValueError(f"Session {session['session_id']} belongs to model {session['model_id']}")
            if request.parent_run_keys != session["active_run_keys"][: len(request.parent_run_keys)]:
                raise ValueError("Parent node runs are not on the active session path")
            for parent_key in request.parent_run_keys:
                if self.storage.read_node_run(parent_key) is None:
                    raise ValueError(f"Parent node run not found: {parent_key}")
            for upload_key in request.input_upload_keys:
                if not self.storage.upload_path(upload_key).exists():
                    raise ValueError(f"Input upload not found: {upload_key}")
        else:
            if request.source_node_run_key not in session["active_run_keys"]:
                raise ValueError("Action source node run is not on the active session path")
            if self.storage.read_node_run(request.source_node_run_key) is None:
                raise ValueError(f"Source node run not found: {request.source_node_run_key}")

        inputs = operation.resolve_inputs(self, request)

        key = self.build_execution_key(operation, request, inputs)

        if request.execution_kind == "node_run":
            cached_record = self.storage.read_node_run(key)
        else:
            cached_record = self.storage.read_action(key)

        if cached_record is not None:
            if request.execution_kind == "node_run":
                session["active_run_keys"] = [*request.parent_run_keys, key]
            else:
                source_key = request.source_node_run_key
                session["actions_by_source"].setdefault(source_key, [])
                if key not in session["actions_by_source"][source_key]:
                    session["actions_by_source"][source_key].append(key)
            self.storage.write_session(session["session_id"], session)
            return {
                "key": key,
                "session_id": session["session_id"],
                "cached": True,
                "outputs": {
                    role: {
                        "filename": output["filename"],
                        "metadata": output["metadata"],
                    }
                    for role, output in cached_record["outputs"].items()
                },
                "metadata": cached_record["metadata"],
                "json_result": cached_record["json_result"],
            }

        context = OperationContext(
            request_id=request.request_id,
            key=key,
            work_dir=self.storage.execution_work_dir(key),
        )
        emit_update = partial(self.emit_update, emit, request.request_id, key, session["session_id"])

        if operation.queue_kind == "gpu":
            async with self.gpu_lock:
                result = await operation.run(inputs, request.params, context, emit_update)
        else:
            result = await operation.run(inputs, request.params, context, emit_update)

        return self.commit_execution(
            request=request,
            operation=operation,
            session=session,
            key=key,
            context=context,
            result=result,
            cached=False,
        )

    def restore_session(self, session_id: str, key: str | None = None) -> Any:
        session = self.storage.read_session(session_id)
        if session is None:
            return None

        active_run_keys = session["active_run_keys"]
        if key is not None and key in active_run_keys:
            active_run_keys = active_run_keys[: active_run_keys.index(key) + 1]

        node_runs = []
        for run_key in active_run_keys:
            run = self.storage.read_node_run(run_key)
            if run is not None:
                node_runs.append(
                    {
                        **run,
                        "outputs": {
                            role: {
                                "filename": output["filename"],
                                "metadata": output["metadata"],
                            }
                            for role, output in run["outputs"].items()
                        },
                    },
                )

        actions = {}
        for source_key, action_keys in session["actions_by_source"].items():
            if source_key not in active_run_keys:
                continue
            actions[source_key] = []
            for action_key in action_keys:
                action = self.storage.read_action(action_key)
                if action is not None:
                    actions[source_key].append(
                        {
                            **action,
                            "outputs": {
                                role: {
                                    "filename": output["filename"],
                                    "metadata": output["metadata"],
                                }
                                for role, output in action["outputs"].items()
                            },
                        },
                    )

        return {
            "session": {
                **session,
                "active_run_keys": active_run_keys,
            },
            "node_runs": node_runs,
            "actions": actions,
        }

    async def emit_update(self, emit: Any, request_id: str, key: str, session_id: str, update: dict) -> None:
        await emit(
            {
                **update,
                "request_id": request_id,
                "key": key,
                "session_id": session_id,
            },
        )

    def find_lineage_node_run(self, parent_run_keys: list[str], operation_id: str) -> Any:
        ancestor_keys: list[str] = []
        for parent_key in parent_run_keys:
            parent = self.storage.read_node_run(parent_key)
            if parent is None:
                raise ValueError(f"Parent node run not found: {parent_key}")
            for ancestor_key in [*parent["ancestor_run_keys"], parent["node_run_key"]]:
                if ancestor_key not in ancestor_keys:
                    ancestor_keys.append(ancestor_key)

        for run_key in reversed(ancestor_keys):
            run = self.storage.read_node_run(run_key)
            if run is not None and run["operation_id"] == operation_id:
                return run

        return None

    def build_execution_key(self, operation: Any, request: Any, inputs: Any) -> str:
        payload = {
            "execution_kind": request.execution_kind,
            "operation_id": operation.operation_id,
            "operation_version": operation.operation_version,
            "params": request.params,
            "key_parts": operation.key_parts(inputs, request.params),
        }

        if request.execution_kind == "node_run":
            payload.update(
                {
                    "model_id": request.model_id,
                    "parent_run_keys": request.parent_run_keys,
                    "input_upload_keys": request.input_upload_keys,
                },
            )
            prefix = "node_run"
        else:
            payload.update(
                {
                    "source_node_run_key": request.source_node_run_key,
                },
            )
            prefix = "action"

        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:32]}"

    def commit_execution(
        self,
        request: Any,
        operation: Any,
        session: Any,
        key: str,
        context: Any,
        result: Any,
        cached: bool,
    ) -> Any:
        output_roles = [output.role for output in result.outputs]
        for role in operation.output_roles:
            if role not in output_roles:
                raise ValueError(f"Operation {operation.operation_id} did not produce output role: {role}")

        outputs = {}
        work_dir = context.work_dir.resolve()
        for output in result.outputs:
            output_path = output.path.resolve()
            if not output_path.exists():
                raise ValueError(f"Operation output not found: {output.role}")
            if output_path != work_dir and work_dir not in output_path.parents:
                raise ValueError(f"Operation output is outside work_dir: {output.role}")

        for output in result.outputs:
            if request.execution_kind == "node_run":
                final_path = self.storage.node_run_output_path(key, output.role)
            else:
                final_path = self.storage.action_output_path(key, output.role)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output.path), str(final_path))
            outputs[output.role] = {
                "filename": output.filename,
                "path": str(final_path),
                "metadata": output.metadata,
            }

        if request.execution_kind == "node_run":
            ancestor_run_keys = []
            for parent_key in request.parent_run_keys:
                parent = self.storage.read_node_run(parent_key)
                if parent is None:
                    raise ValueError(f"Parent node run not found: {parent_key}")
                for ancestor_key in [*parent["ancestor_run_keys"], parent_key]:
                    if ancestor_key not in ancestor_run_keys:
                        ancestor_run_keys.append(ancestor_key)

            record = {
                "node_run_key": key,
                "execution_kind": "node_run",
                "operation_id": operation.operation_id,
                "operation_version": operation.operation_version,
                "model_id": request.model_id,
                "parent_run_keys": request.parent_run_keys,
                "ancestor_run_keys": ancestor_run_keys,
                "input_upload_keys": request.input_upload_keys,
                "params": request.params,
                "outputs": outputs,
                "metadata": result.metadata,
                "json_result": result.json_result,
            }
            self.storage.write_node_run(key, record)
            session["active_run_keys"] = [*request.parent_run_keys, key]
        else:
            record = {
                "action_key": key,
                "execution_kind": "action",
                "operation_id": operation.operation_id,
                "operation_version": operation.operation_version,
                "source_node_run_key": request.source_node_run_key,
                "params": request.params,
                "outputs": outputs,
                "metadata": result.metadata,
                "json_result": result.json_result,
            }
            self.storage.write_action(key, record)
            source_key = request.source_node_run_key
            session["actions_by_source"].setdefault(source_key, [])
            if key not in session["actions_by_source"][source_key]:
                session["actions_by_source"][source_key].append(key)

        self.storage.write_session(session["session_id"], session)

        return {
            "key": key,
            "session_id": session["session_id"],
            "cached": cached,
            "outputs": {
                role: {
                    "filename": output["filename"],
                    "metadata": output["metadata"],
                }
                for role, output in outputs.items()
            },
            "metadata": result.metadata,
            "json_result": result.json_result,
        }
