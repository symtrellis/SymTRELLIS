import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import Any

from .operations import Operation, OperationContext, OperationInputs, OperationResult


@dataclass
class ExecutionRequest:
    request_id: str
    execution_kind: str
    operation_id: str
    model_id: str | None
    session_id: str | None
    session_revision: int
    parent_run_keys: list[str]
    input_upload_keys: list[str]
    source_node_run_key: str | None
    params: dict[str, Any]


@dataclass
class PreparedExecution:
    request: ExecutionRequest
    operation: Operation
    session: dict[str, Any]
    inputs: OperationInputs
    key: str
    cached_record: dict[str, Any] | None


class Coordinator:
    def __init__(self, storage: Any, operations: dict[str, Operation]):
        self.storage = storage
        self.operations = operations

    def _resolve(self, request: ExecutionRequest) -> PreparedExecution:
        operation = self.operations.get(request.operation_id)
        if operation is None:
            raise ValueError(f"Unknown operation_id: {request.operation_id}")
        if operation.execution_kind != request.execution_kind:
            raise ValueError(
                f"Operation {operation.operation_id} expects {operation.execution_kind}, " f"got {request.execution_kind}",
            )

        if operation.creates_session and request.session_id is None:
            if request.session_revision != 0:
                raise ValueError("A new session must start at revision 0")
            session = self.storage.create_session(request.model_id)
        else:
            if request.session_id is None:
                raise ValueError(f"Operation {operation.operation_id} requires session_id")
            session = self.storage.read_session(request.session_id)
            if session is None:
                raise ValueError(f"Session not found: {request.session_id}")

        if request.session_revision != session["revision"]:
            raise ValueError(f"Session revision conflict: {session['session_id']}")
        if request.model_id != session["model_id"]:
            raise ValueError(
                f"Session {session['session_id']} belongs to model {session['model_id']}",
            )

        if request.execution_kind == "node_run":
            if request.parent_run_keys != session["active_run_keys"][: len(request.parent_run_keys)]:
                raise ValueError("Parent node runs are not on the active session path")
            for parent_key in request.parent_run_keys:
                if self.storage.read_node_run(parent_key) is None:
                    raise ValueError(f"Parent node run not found: {parent_key}")
            for upload_key in request.input_upload_keys:
                if self.storage.read_upload(upload_key) is None:
                    raise ValueError(f"Input upload record not found: {upload_key}")
                if not self.storage.upload_path(upload_key).exists():
                    raise ValueError(f"Input upload file not found: {upload_key}")
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

        return PreparedExecution(
            request=request,
            operation=operation,
            session=session,
            inputs=inputs,
            key=key,
            cached_record=cached_record,
        )

    def prepare(self, request: ExecutionRequest) -> dict[str, Any]:
        prepared = self._resolve(request)

        if prepared.cached_record is not None:
            return {
                "status": "completed",
                "result": self._cached_response(prepared),
            }

        if prepared.operation.queue_kind == "inline":
            context = OperationContext(
                request_id=request.request_id,
                key=prepared.key,
                work_dir=self.storage.execution_work_dir(request.request_id),
            )
            result = prepared.operation.run(
                prepared.inputs,
                request.params,
                context,
                None,
            )
            return {
                "status": "completed",
                "result": self.commit_execution(prepared, result),
            }

        return {
            "status": "gpu_required",
            "session_id": prepared.session["session_id"],
            "session_revision": prepared.session["revision"],
        }

    def execute_gpu(self, request: ExecutionRequest, progress: Any) -> dict[str, Any]:
        prepared = self._resolve(request)

        if prepared.cached_record is not None:
            return self._cached_response(prepared)

        if prepared.operation.queue_kind != "gpu":
            raise ValueError(f"Operation {prepared.operation.operation_id} is not a GPU operation")

        context = OperationContext(
            request_id=request.request_id,
            key=prepared.key,
            work_dir=self.storage.execution_work_dir(request.request_id),
        )
        result = prepared.operation.run(
            prepared.inputs,
            request.params,
            context,
            progress,
        )
        return self.commit_execution(prepared, result)

    def _cached_response(self, prepared: PreparedExecution) -> dict[str, Any]:
        request = prepared.request
        if request.execution_kind == "node_run":
            prepared.session["active_run_keys"] = [*request.parent_run_keys, prepared.key]
        else:
            source_key = request.source_node_run_key
            prepared.session["actions_by_source"].setdefault(source_key, [])
            if prepared.key not in prepared.session["actions_by_source"][source_key]:
                prepared.session["actions_by_source"][source_key].append(prepared.key)

        actual_record, new_revision = self.storage.commit_execution(
            execution_kind=request.execution_kind,
            key=prepared.key,
            record=prepared.cached_record,
            session=prepared.session,
            expected_revision=request.session_revision,
        )

        return {
            "key": prepared.key,
            "session_id": prepared.session["session_id"],
            "session_revision": new_revision,
            "cached": True,
            "outputs": {
                role: {
                    "filename": output["filename"],
                    "metadata": output["metadata"],
                }
                for role, output in actual_record["outputs"].items()
            },
            "metadata": actual_record["metadata"],
            "json_result": actual_record["json_result"],
        }

    def commit_execution(
        self,
        prepared: PreparedExecution,
        result: OperationResult,
    ) -> dict[str, Any]:
        request = prepared.request
        output_roles = [output.role for output in result.outputs]
        for role in prepared.operation.output_roles:
            if role not in output_roles:
                raise ValueError(
                    f"Operation {prepared.operation.operation_id} did not produce output role: {role}",
                )

        outputs = {}
        expected_work_dir = (self.storage.attempts_dir / request.request_id).resolve()
        for output in result.outputs:
            output_path = output.path.resolve()
            if not output_path.exists():
                raise ValueError(f"Operation output not found: {output.role}")
            if output_path != expected_work_dir and expected_work_dir not in output_path.parents:
                raise ValueError(f"Operation output is outside work_dir: {output.role}")

        for output in result.outputs:
            final_path = self.storage.output_path(
                request.execution_kind,
                prepared.key,
                output.role,
            )
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
                "node_run_key": prepared.key,
                "execution_kind": "node_run",
                "operation_id": prepared.operation.operation_id,
                "operation_version": prepared.operation.operation_version,
                "model_id": request.model_id,
                "parent_run_keys": request.parent_run_keys,
                "ancestor_run_keys": ancestor_run_keys,
                "input_upload_keys": request.input_upload_keys,
                "params": request.params,
                "outputs": outputs,
                "metadata": result.metadata,
                "json_result": result.json_result,
            }
            prepared.session["active_run_keys"] = [*request.parent_run_keys, prepared.key]
        else:
            record = {
                "action_key": prepared.key,
                "execution_kind": "action",
                "operation_id": prepared.operation.operation_id,
                "operation_version": prepared.operation.operation_version,
                "source_node_run_key": request.source_node_run_key,
                "params": request.params,
                "outputs": outputs,
                "metadata": result.metadata,
                "json_result": result.json_result,
            }
            source_key = request.source_node_run_key
            prepared.session["actions_by_source"].setdefault(source_key, [])
            if prepared.key not in prepared.session["actions_by_source"][source_key]:
                prepared.session["actions_by_source"][source_key].append(prepared.key)

        actual_record, new_revision = self.storage.commit_execution(
            execution_kind=request.execution_kind,
            key=prepared.key,
            record=record,
            session=prepared.session,
            expected_revision=request.session_revision,
        )

        return {
            "key": prepared.key,
            "session_id": prepared.session["session_id"],
            "session_revision": new_revision,
            "cached": False,
            "outputs": {
                role: {
                    "filename": output["filename"],
                    "metadata": output["metadata"],
                }
                for role, output in actual_record["outputs"].items()
            },
            "metadata": actual_record["metadata"],
            "json_result": actual_record["json_result"],
        }

    def restore_session(
        self,
        session_id: str,
        key: str | None = None,
    ) -> dict[str, Any] | None:
        session = self.storage.read_session(session_id)
        if session is None:
            return None

        if key is None:
            active_run_keys = session["active_run_keys"]
        else:
            if key not in session["active_run_keys"]:
                return None
            key_index = session["active_run_keys"].index(key)
            active_run_keys = session["active_run_keys"][: key_index + 1]

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

    def find_lineage_node_run(
        self,
        parent_run_keys: list[str],
        operation_id: str,
    ) -> dict[str, Any] | None:
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

    def build_execution_key(
        self,
        operation: Operation,
        request: ExecutionRequest,
        inputs: OperationInputs,
    ) -> str:
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
