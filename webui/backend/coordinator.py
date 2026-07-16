import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .operations import Operation, OperationContext, OperationInputs, OperationResult
from .storage import SessionExpiredError


@dataclass
class ExecutionRequest:
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
class GpuExecutionInput:
    operation_id: str
    inputs: OperationInputs
    params: dict[str, Any]
    key: str
    attempt_id: str
    work_dir: Path


@dataclass
class PreparedExecution:
    request: ExecutionRequest
    operation: Operation
    session: dict[str, Any]
    inputs: OperationInputs
    key: str
    cached_record: dict[str, Any] | None


class Coordinator:
    def __init__(
        self,
        storage: Any,
        operations: dict[str, Operation],
        runtime_id: str,
        prepared_reservation_seconds: int,
    ):
        self.storage = storage
        self.operations = operations
        self.runtime_id = runtime_id
        self.prepared_reservation_seconds = prepared_reservation_seconds

    def _resolve(
        self,
        request: ExecutionRequest,
        session: dict[str, Any],
    ) -> PreparedExecution:
        operation = self.operations.get(request.operation_id)
        if operation is None:
            raise ValueError(f"Unknown operation_id: {request.operation_id}")
        if operation.execution_kind != request.execution_kind:
            raise ValueError(
                f"Operation {operation.operation_id} expects {operation.execution_kind}, " f"got {request.execution_kind}",
            )

        if request.session_id != session["session_id"]:
            raise ValueError(f"Task session mismatch: {session['session_id']}")
        if session["status"] != "active":
            raise SessionExpiredError(f"Session expired: {session['session_id']}")
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
        operation = self.operations.get(request.operation_id)
        if operation is None:
            raise ValueError(f"Unknown operation_id: {request.operation_id}")
        if operation.execution_kind != request.execution_kind:
            raise ValueError(
                f"Operation {operation.operation_id} expects {operation.execution_kind}, " f"got {request.execution_kind}",
            )

        task, session = self.storage.begin_session_task(
            request=asdict(request),
            creates_session=operation.creates_session,
            runtime_id=self.runtime_id,
            prepared_reservation_seconds=self.prepared_reservation_seconds,
        )
        task_id = task["task_id"]

        try:
            stored_request = ExecutionRequest(**task["request"])
            prepared = self._resolve(stored_request, session)

            if prepared.cached_record is not None:
                self.storage.mark_task_committing(task_id)
                return {
                    "status": "completed",
                    "result": self._cached_response(prepared, task_id),
                }

            if prepared.operation.queue_kind == "inline":
                self.storage.start_inline_execution(task_id)
                _, work_dir = self.storage.create_execution_attempt(task_id)
                context = OperationContext(
                    key=prepared.key,
                    work_dir=work_dir,
                )
                result = prepared.operation.run(
                    prepared.inputs,
                    stored_request.params,
                    context,
                    None,
                )
                return {
                    "status": "completed",
                    "result": self.commit_execution(
                        prepared,
                        result,
                        task_id,
                        work_dir,
                    ),
                }

            if prepared.operation.queue_kind != "gpu":
                raise ValueError(
                    f"Unknown queue kind for {prepared.operation.operation_id}: " f"{prepared.operation.queue_kind}",
                )

            return {
                "status": "gpu_required",
                "task_id": task_id,
            }
        except Exception:
            self.storage.fail_execution(task_id)
            raise

    def execute(
        self,
        task_id: str,
        progress: Any,
        gpu_runner: Callable[[GpuExecutionInput, Any], OperationResult],
    ) -> dict[str, Any]:
        try:
            task = self.storage.start_queued_execution(task_id)
            request = ExecutionRequest(**task["request"])
            session = self.storage.read_session(task["session_id"])
            if session is None:
                raise ValueError(f"Session not found: {task['session_id']}")

            prepared = self._resolve(request, session)
            if prepared.cached_record is not None:
                self.storage.mark_task_committing(task_id)
                return self._cached_response(prepared, task_id)

            if prepared.operation.queue_kind != "gpu":
                raise ValueError(
                    f"Operation {prepared.operation.operation_id} is not a GPU operation",
                )

            attempt_id, work_dir = self.storage.create_execution_attempt(task_id)
            gpu_input = GpuExecutionInput(
                operation_id=prepared.operation.operation_id,
                inputs=prepared.inputs,
                params=request.params,
                key=prepared.key,
                attempt_id=attempt_id,
                work_dir=work_dir,
            )
            result = gpu_runner(gpu_input, progress)
            return self.commit_execution(
                prepared,
                result,
                task_id,
                work_dir,
            )
        except Exception:
            self.storage.fail_execution(task_id)
            raise

    def _cached_response(
        self,
        prepared: PreparedExecution,
        task_id: str,
    ) -> dict[str, Any]:
        request = prepared.request
        if request.execution_kind == "node_run":
            prepared.session["active_run_keys"] = [*request.parent_run_keys, prepared.key]
        else:
            source_key = request.source_node_run_key
            prepared.session["actions_by_source"].setdefault(source_key, [])
            if prepared.key not in prepared.session["actions_by_source"][source_key]:
                prepared.session["actions_by_source"][source_key].append(prepared.key)

        actual_record, new_revision, _ = self.storage.commit_execution(
            task_id=task_id,
            execution_kind=request.execution_kind,
            key=prepared.key,
            record=prepared.cached_record,
            session=prepared.session,
            expected_revision=request.session_revision,
            actual_work_dir=None,
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
        task_id: str,
        actual_work_dir: Path,
    ) -> dict[str, Any]:
        request = prepared.request
        output_roles = [output.role for output in result.outputs]
        for role in prepared.operation.output_roles:
            if role not in output_roles:
                raise ValueError(
                    f"Operation {prepared.operation.operation_id} did not produce output role: {role}",
                )

        expected_work_dir = actual_work_dir.resolve()
        attempts_dir = self.storage.attempts_dir.resolve()
        if expected_work_dir == attempts_dir or attempts_dir not in expected_work_dir.parents:
            raise ValueError("Execution work_dir is outside attempts directory")

        final_dir = self.storage.outputs_dir / ("node_runs" if request.execution_kind == "node_run" else "actions") / prepared.key
        outputs = {}
        for output in result.outputs:
            output_path = output.path.resolve()
            if not output_path.is_file():
                raise ValueError(f"Operation output not found: {output.role}")
            if output_path != expected_work_dir and expected_work_dir not in output_path.parents:
                raise ValueError(f"Operation output is outside work_dir: {output.role}")
            final_path = final_dir / output_path.relative_to(expected_work_dir)
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

        self.storage.mark_task_committing(task_id)
        actual_record, new_revision, cache_hit = self.storage.commit_execution(
            task_id=task_id,
            execution_kind=request.execution_kind,
            key=prepared.key,
            record=record,
            session=prepared.session,
            expected_revision=request.session_revision,
            actual_work_dir=actual_work_dir,
        )

        return {
            "key": prepared.key,
            "session_id": prepared.session["session_id"],
            "session_revision": new_revision,
            "cached": cache_hit,
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
    ) -> dict[str, Any]:
        snapshot = self.storage.read_session_restore_snapshot(session_id)
        if snapshot["status"] == "not_found":
            return {"status": "not_found"}
        if snapshot["status"] == "session_expired":
            return {"status": "session_expired"}

        session = snapshot["session"]
        if key is None:
            active_run_keys = session["active_run_keys"]
        else:
            if key not in session["active_run_keys"]:
                return {"status": "not_found"}
            key_index = session["active_run_keys"].index(key)
            active_run_keys = session["active_run_keys"][: key_index + 1]

        node_runs = []
        for run_key in active_run_keys:
            run = snapshot["node_runs_by_key"][run_key]
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
                action = snapshot["actions_by_key"][action_key]
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
            "status": "restored",
            "restored": {
                "session": {
                    **session,
                    "active_run_keys": active_run_keys,
                },
                "node_runs": node_runs,
                "actions": actions,
            },
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
