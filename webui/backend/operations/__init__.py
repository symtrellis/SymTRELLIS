from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OperationInputs:
    records: dict[str, Any] = field(default_factory=dict)
    uploads: dict[str, Path] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationContext:
    key: str
    work_dir: Path


@dataclass
class OperationOutput:
    role: str
    path: Path
    filename: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationResult:
    outputs: list[OperationOutput] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    json_result: Any = None


class Operation:
    operation_id: str
    operation_version = "1"
    execution_kind: str
    queue_kind: str
    creates_session = False
    output_roles: tuple[str, ...] = ()

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        raise NotImplementedError

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        progress: Callable[..., Any] | None,
    ) -> OperationResult:
        raise NotImplementedError


def forward_glb_progress(
    progress: Callable[..., Any],
    offset: float,
    scale: float,
    update: dict[str, Any],
) -> None:
    total_progress = offset + scale * update["progress"]
    stage = update["stage"]
    progress(total_progress, desc=stage)
