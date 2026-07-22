import inspect
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from tqdm import tqdm

__all__ = ["Task", "Stage", "Pipeline"]


class Task:
    def __init__(
        self,
        payload: Mapping[str, Any],
        root_id: int,
        start: float,
        end: float,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("task payload must be a mapping")
        if not all(isinstance(name, str) and name for name in payload):
            raise TypeError("task payload keys must be non-empty strings")
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(f"invalid task interval ({start}, {end})")

        self.payload = dict(payload)
        self.root_id = root_id
        self.start = float(start)
        self.end = float(end)

    def derive(self, result: Any, inherit: bool) -> list["Task"]:
        if result is None:
            return []
        if isinstance(result, Mapping):
            patches = [result]
        elif isinstance(result, list):
            if not all(isinstance(patch, Mapping) for patch in result):
                raise TypeError("every emitted result must be a mapping")
            patches = result
        else:
            raise TypeError("a stage must return None, a mapping, or a list of mappings")

        payloads = []
        for patch in patches:
            if inherit:
                payload = dict(self.payload)
                payload.update(patch)
            else:
                payload = dict(patch)
            payloads.append(payload)
        return self.split(payloads)

    def split(self, payloads: list[Mapping[str, Any]]) -> list["Task"]:
        if not payloads:
            return []

        width = (self.end - self.start) / len(payloads)
        return [
            Task(
                payload,
                self.root_id,
                self.start + index * width,
                (self.end if index + 1 == len(payloads) else self.start + (index + 1) * width),
            )
            for index, payload in enumerate(payloads)
        ]


class ItemPolicy:
    def accept(self, task: Task, now: float) -> list[tuple[Task, ...]]:
        return [(task,)]

    def poll(self, now: float) -> list[tuple[Task, ...]]:
        return []

    def close_root(self, root_id: int, bounded: bool) -> list[tuple[Task, ...]]:
        return []

    def close_input(self) -> list[tuple[Task, ...]]:
        return []

    def task_argument(self, unit: tuple[Task, ...], name: str) -> Any:
        return unit[0].payload[name]

    def map_result(self, unit: tuple[Task, ...], result: Any) -> list[Task]:
        return unit[0].derive(result, inherit=True)


class BatchPolicy:
    def __init__(self, batch_size: int, timeout_s: float | None) -> None:
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.buffer: list[Task] = []
        self.started_at: float | None = None

    def accept(self, task: Task, now: float) -> list[tuple[Task, ...]]:
        if not self.buffer:
            self.started_at = now
        self.buffer.append(task)
        if len(self.buffer) < self.batch_size:
            return []

        unit = tuple(self.buffer)
        self.buffer = []
        self.started_at = None
        return [unit]

    def poll(self, now: float) -> list[tuple[Task, ...]]:
        if self.timeout_s is None or not self.buffer:
            return []
        assert self.started_at is not None
        if now - self.started_at < self.timeout_s:
            return []

        unit = tuple(self.buffer)
        self.buffer = []
        self.started_at = None
        return [unit]

    def close_root(self, root_id: int, bounded: bool) -> list[tuple[Task, ...]]:
        if not bounded or not any(task.root_id == root_id for task in self.buffer):
            return []

        unit = tuple(self.buffer)
        self.buffer = []
        self.started_at = None
        return [unit]

    def close_input(self) -> list[tuple[Task, ...]]:
        if not self.buffer:
            return []

        unit = tuple(self.buffer)
        self.buffer = []
        self.started_at = None
        return [unit]

    def task_argument(self, unit: tuple[Task, ...], name: str) -> Any:
        present = [name in task.payload for task in unit]
        if all(present):
            return [task.payload[name] for task in unit]
        if any(present):
            raise ValueError(f"batch argument {name!r} is missing from part of the batch")
        raise KeyError(name)

    def map_result(self, unit: tuple[Task, ...], result: Any) -> list[Task]:
        if result is None:
            return []
        if not isinstance(result, list):
            raise TypeError("a batch stage must return a list aligned with its inputs")
        if len(result) != len(unit):
            raise ValueError(
                f"batch returned {len(result)} result slots for {len(unit)} inputs",
            )

        tasks = []
        for parent, slot in zip(unit, result, strict=True):
            tasks.extend(parent.derive(slot, inherit=True))
        return tasks


class GroupPolicy:
    def __init__(
        self,
        key_field: str,
        size_field: str,
        position_field: str,
        timeout_s: float | None,
    ) -> None:
        self.key_field = key_field
        self.size_field = size_field
        self.position_field = position_field
        self.timeout_s = timeout_s
        self.slots: dict[tuple[int, Any], list[Task | None]] = {}
        self.counts: dict[tuple[int, Any], int] = {}
        self.sizes: dict[tuple[int, Any], int] = {}
        self.started_at: dict[tuple[int, Any], float] = {}

    def accept(self, task: Task, now: float) -> list[tuple[Task, ...]]:
        payload = task.payload
        try:
            key = payload[self.key_field]
            size = payload[self.size_field]
            position = payload[self.position_field]
        except KeyError as error:
            raise ValueError(f"group metadata field {error.args[0]!r} is missing") from error

        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"group size must be a positive integer, got {size!r}")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"group position must be an integer, got {position!r}")
        if not 0 <= position < size:
            raise ValueError(f"group position {position} is outside [0, {size})")

        group_id = (task.root_id, key)
        if group_id not in self.slots:
            self.slots[group_id] = [None] * size
            self.counts[group_id] = 0
            self.sizes[group_id] = size
            self.started_at[group_id] = now
        elif self.sizes[group_id] != size:
            raise ValueError(
                f"group {key!r} changed size from {self.sizes[group_id]} to {size}",
            )

        if self.slots[group_id][position] is not None:
            raise ValueError(f"group {key!r} received duplicate position {position}")

        self.slots[group_id][position] = task
        self.counts[group_id] += 1
        if self.counts[group_id] < size:
            return []

        slots = self.slots.pop(group_id)
        self.counts.pop(group_id)
        self.sizes.pop(group_id)
        self.started_at.pop(group_id)
        return [tuple(slots)]  # type: ignore[arg-type]

    def poll(self, now: float) -> list[tuple[Task, ...]]:
        if self.timeout_s is None:
            return []

        for group_id, started_at in self.started_at.items():
            if now - started_at >= self.timeout_s:
                root_id, key = group_id
                raise TimeoutError(
                    f"group {key!r} for root {root_id} timed out: " f"received {self.counts[group_id]} of {self.sizes[group_id]}",
                )
        return []

    def close_root(self, root_id: int, bounded: bool) -> list[tuple[Task, ...]]:
        incomplete = [group_id for group_id in self.slots if group_id[0] == root_id]
        if incomplete:
            _, key = incomplete[0]
            raise RuntimeError(
                f"root {root_id} closed with incomplete group {key!r}: " f"received {self.counts[incomplete[0]]} of {self.sizes[incomplete[0]]}",
            )
        return []

    def close_input(self) -> list[tuple[Task, ...]]:
        if self.slots:
            group_id = next(iter(self.slots))
            root_id, key = group_id
            raise RuntimeError(
                f"input closed with incomplete group {key!r} for root {root_id}: " f"received {self.counts[group_id]} of {self.sizes[group_id]}",
            )
        return []

    def task_argument(self, unit: tuple[Task, ...], name: str) -> Any:
        if name == self.key_field or name == self.size_field:
            return unit[0].payload[name]

        present = [name in task.payload for task in unit]
        if all(present):
            return [task.payload[name] for task in unit]
        if any(present):
            raise ValueError(f"group argument {name!r} is missing from part of the group")
        raise KeyError(name)

    def map_result(self, unit: tuple[Task, ...], result: Any) -> list[Task]:
        root_ids = {task.root_id for task in unit}
        if len(root_ids) != 1:
            raise RuntimeError("a group execution unit contains more than one root")

        aggregate = Task(
            {},
            unit[0].root_id,
            min(task.start for task in unit),
            max(task.end for task in unit),
        )
        return aggregate.derive(result, inherit=False)


class Stage:
    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        mode: str = "item",
        workers: int = 1,
        queue_size: int = 0,
        work_queue_size: int = 0,
        batch_size: int | None = None,
        batch_timeout_s: float | None = None,
        group_key: str | None = None,
        group_size: str | None = None,
        group_position: str | None = None,
        group_timeout_s: float | None = None,
        params: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("stage name must not be empty")
        if not callable(fn):
            raise TypeError(f"stage {name!r} function must be callable")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError(f"stage {name!r} workers must be a positive integer")
        queue_sizes = (queue_size, work_queue_size)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in queue_sizes):
            raise ValueError(f"stage {name!r} queue sizes must be non-negative integers")

        self.name = name
        self.fn = fn
        self.mode = mode
        self.workers = workers
        self.queue_size = queue_size
        self.work_queue_size = work_queue_size

        match mode:
            case "item":
                if any(
                    value is not None
                    for value in (
                        batch_size,
                        batch_timeout_s,
                        group_key,
                        group_size,
                        group_position,
                        group_timeout_s,
                    )
                ):
                    raise ValueError(f"item stage {name!r} received batch/group options")
                self.policy = ItemPolicy()
            case "batch":
                if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                    raise ValueError(f"batch stage {name!r} requires a positive batch_size")
                if any(
                    value is not None
                    for value in (
                        group_key,
                        group_size,
                        group_position,
                        group_timeout_s,
                    )
                ):
                    raise ValueError(f"batch stage {name!r} received group options")
                if batch_timeout_s is not None and batch_timeout_s <= 0:
                    raise ValueError(f"batch stage {name!r} timeout must be positive")
                self.policy = BatchPolicy(batch_size, batch_timeout_s)
            case "group":
                if batch_size is not None or batch_timeout_s is not None:
                    raise ValueError(f"group stage {name!r} received batch options")
                fields = (group_key, group_size, group_position)
                if not all(isinstance(field, str) and field for field in fields):
                    raise ValueError(
                        f"group stage {name!r} requires group_key, group_size, and group_position",
                    )
                assert group_key is not None
                assert group_size is not None
                assert group_position is not None
                if group_timeout_s is not None and group_timeout_s <= 0:
                    raise ValueError(f"group stage {name!r} timeout must be positive")
                self.policy = GroupPolicy(
                    group_key,
                    group_size,
                    group_position,
                    group_timeout_s,
                )
            case _:
                raise ValueError(f"stage {name!r} has unsupported mode {mode!r}")

        self.params = dict(params or {})
        self.resources = dict(resources or {})
        overlap = set(self.params).intersection(self.resources)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"stage {name!r} duplicates params/resources: {names}")
        self.fixed_arguments = dict(self.params)
        self.fixed_arguments.update(self.resources)

        signature = inspect.signature(fn)
        self.positional_plan: list[tuple[str, Any]] = []
        self.keyword_plan: list[tuple[str, Any]] = []
        self.accepts_kwargs = False
        declared_names = set()
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(f"stage {name!r} function may not declare *args")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                self.accepts_kwargs = True
                continue
            declared_names.add(parameter.name)
            entry = (parameter.name, parameter.default)
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                self.positional_plan.append(entry)
            else:
                self.keyword_plan.append(entry)

        unknown_fixed = set(self.fixed_arguments).difference(declared_names)
        if unknown_fixed and not self.accepts_kwargs:
            names = ", ".join(sorted(unknown_fixed))
            raise ValueError(f"stage {name!r} function does not accept fixed arguments: {names}")

        self.pipeline: Pipeline | None = None
        self.index: int | None = None
        self.next_stage: Stage | None = None
        self.input_queue: queue.Queue[tuple[str, Any]] | None = None
        self.work_queue: queue.Queue[tuple[Task, ...] | None] | None = None
        self.threads: list[threading.Thread] = []
        self.stopping = threading.Event()
        self.state_lock = threading.Lock()
        self.pending_by_root: defaultdict[int, int] = defaultdict(int)
        self.closed_roots: set[int] = set()
        self.forwarded_roots: set[int] = set()
        self.input_closed = False
        self.input_forwarded = False

    def bind(
        self,
        pipeline: "Pipeline",
        index: int,
        next_stage: "Stage | None",
    ) -> None:
        self.pipeline = pipeline
        self.index = index
        self.next_stage = next_stage
        self.input_queue = queue.Queue(maxsize=self.queue_size)
        self.work_queue = queue.Queue(maxsize=self.work_queue_size)
        self.threads = []
        self.stopping = threading.Event()
        self.state_lock = threading.Lock()
        self.pending_by_root = defaultdict(int)
        self.closed_roots = set()
        self.forwarded_roots = set()
        self.input_closed = False
        self.input_forwarded = False

    def start(self) -> None:
        if self.pipeline is None or self.input_queue is None or self.work_queue is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        coordinator = threading.Thread(
            target=self.coordinator_loop,
            name=f"{self.name}-coordinator",
            daemon=True,
        )
        self.threads.append(coordinator)
        coordinator.start()

        for worker_index in range(self.workers):
            worker = threading.Thread(
                target=self.worker_loop,
                name=f"{self.name}-worker-{worker_index}",
                daemon=True,
            )
            self.threads.append(worker)
            worker.start()

    def submit(self, kind: str, value: Any = None) -> bool:
        if self.pipeline is None or self.input_queue is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        while not self.pipeline.stop_event.is_set() and not self.stopping.is_set():
            try:
                self.input_queue.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def coordinator_loop(self) -> None:
        pipeline = self.pipeline
        input_queue = self.input_queue
        if pipeline is None or input_queue is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        context: Any = None
        try:
            while not pipeline.stop_event.is_set() and not self.stopping.is_set():
                self.queue_units(self.policy.poll(time.monotonic()))
                if self.input_forwarded:
                    return

                try:
                    kind, value = input_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                context = {"event": kind, "value": value}
                match kind:
                    case "task":
                        task = value
                        with self.state_lock:
                            self.pending_by_root[task.root_id] += 1
                        units = self.policy.accept(task, time.monotonic())
                    case "root_closed":
                        with self.state_lock:
                            self.closed_roots.add(value)
                        units = self.policy.close_root(
                            value,
                            pipeline.max_active_roots is not None,
                        )
                    case "input_closed":
                        with self.state_lock:
                            self.input_closed = True
                        units = self.policy.close_input()
                    case "stop":
                        return
                    case _:
                        raise ValueError(f"stage {self.name!r} received unknown event {kind!r}")

                self.queue_units(units)
                self.advance_completion()
        except Exception as error:
            if not pipeline.stop_event.is_set():
                pipeline.fail(error, self.name, "coordinator", context)

    def queue_units(self, units: list[tuple[Task, ...]]) -> None:
        if self.pipeline is None or self.work_queue is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        for unit in units:
            while not self.pipeline.stop_event.is_set() and not self.stopping.is_set():
                try:
                    self.work_queue.put(unit, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def worker_loop(self) -> None:
        pipeline = self.pipeline
        work_queue = self.work_queue
        index = self.index
        if pipeline is None or work_queue is None or index is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        unit: tuple[Task, ...] | None = None
        try:
            while not pipeline.stop_event.is_set() and not self.stopping.is_set():
                try:
                    unit = work_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if unit is None:
                    return

                args, kwargs = self.build_arguments(unit)
                result = self.fn(*args, **kwargs)
                output_tasks = self.policy.map_result(unit, result)
                pipeline.update_progress(
                    index,
                    max(task.end for task in unit),
                )
                self.complete_unit(unit, output_tasks)
        except Exception as error:
            if not pipeline.stop_event.is_set():
                task_context = None
                if unit is not None:
                    task_context = [
                        {
                            "root_id": task.root_id,
                            "interval": (task.start, task.end),
                            "payload_keys": sorted(task.payload),
                        }
                        for task in unit
                    ]
                pipeline.fail(error, self.name, "worker", task_context)

    def build_arguments(self, unit: tuple[Task, ...]) -> tuple[list[Any], dict[str, Any]]:
        args = []
        kwargs = {}

        for name in self.fixed_arguments:
            if any(name in task.payload for task in unit):
                raise ValueError(f"argument {name!r} exists in both task data and stage configuration")

        for name, default in self.positional_plan:
            if name in self.fixed_arguments:
                args.append(self.fixed_arguments[name])
                continue
            try:
                args.append(self.policy.task_argument(unit, name))
            except KeyError:
                if default is inspect.Parameter.empty:
                    raise ValueError(f"required argument {name!r} is missing") from None
                args.append(default)

        for name, default in self.keyword_plan:
            if name in self.fixed_arguments:
                kwargs[name] = self.fixed_arguments[name]
                continue
            try:
                kwargs[name] = self.policy.task_argument(unit, name)
            except KeyError:
                if default is inspect.Parameter.empty:
                    raise ValueError(f"required argument {name!r} is missing") from None

        if self.accepts_kwargs:
            planned_names = {name for name, _ in self.positional_plan + self.keyword_plan}
            payload_names = set().union(*(task.payload.keys() for task in unit))
            for name in payload_names.difference(planned_names, self.fixed_arguments):
                kwargs[name] = self.policy.task_argument(unit, name)
            for name, value in self.fixed_arguments.items():
                if name not in planned_names:
                    kwargs[name] = value
        return args, kwargs

    def complete_unit(self, unit: tuple[Task, ...], output_tasks: list[Task]) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        for task in output_tasks:
            if self.next_stage is None:
                self.pipeline.collect_result(task.payload)
            else:
                self.next_stage.submit("task", task)

        with self.state_lock:
            for task in unit:
                self.pending_by_root[task.root_id] -= 1
                if self.pending_by_root[task.root_id] == 0:
                    self.pending_by_root.pop(task.root_id)
        self.advance_completion()

    def advance_completion(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"stage {self.name!r} is not bound")

        with self.state_lock:
            for root_id in self.closed_roots.difference(self.forwarded_roots):
                if self.pending_by_root.get(root_id, 0) != 0:
                    continue
                if self.next_stage is None:
                    self.pipeline.root_finished(root_id)
                else:
                    self.next_stage.submit("root_closed", root_id)
                self.forwarded_roots.add(root_id)

            roots_complete = self.closed_roots.issubset(self.forwarded_roots)
            if self.input_closed and not self.input_forwarded and not self.pending_by_root and roots_complete:
                if self.next_stage is None:
                    self.pipeline.input_finished()
                else:
                    self.next_stage.submit("input_closed")
                self.input_forwarded = True

    def stop(self) -> None:
        self.stopping.set()

        if self.input_queue is not None:
            try:
                self.input_queue.put_nowait(("stop", None))
            except queue.Full:
                pass
        if self.work_queue is not None:
            for _ in range(self.workers):
                try:
                    self.work_queue.put_nowait(None)
                except queue.Full:
                    break

        current = threading.current_thread()
        for thread in self.threads:
            if thread is not current:
                thread.join()


class Pipeline:
    def __init__(
        self,
        stages: Iterable[Stage],
        *,
        max_active_roots: int | None = None,
        progress: bool = True,
        total: int | None = None,
    ) -> None:
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("a pipeline requires at least one stage")
        if max_active_roots is not None and (isinstance(max_active_roots, bool) or not isinstance(max_active_roots, int) or max_active_roots <= 0):
            raise ValueError("max_active_roots must be None or a positive integer")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("total must be None or a non-negative integer")

        self.max_active_roots = max_active_roots
        self.progress = progress
        self.total = total
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.progress_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.results: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.active_roots = 0
        self.input_complete = False
        self.closed = False
        self.progress_values = [0.0] * len(self.stages)
        self.progress_bars = []
        if progress:
            name_width = max(len(stage.name) for stage in self.stages)
            bar_format = (
                "{desc} |{bar}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}]"
            )
            for index, stage in enumerate(self.stages):
                self.progress_bars.append(
                    tqdm(
                        total=1.0,
                        desc=stage.name.ljust(name_width),
                        position=index,
                        leave=True,
                        dynamic_ncols=True,
                        bar_format=bar_format,
                    ),
                )

    def run(self, inputs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        try:
            if self.closed:
                raise RuntimeError("a pipeline instance can only run once")
            if self.total is None:
                try:
                    self.total = len(inputs)  # type: ignore[arg-type]
                except TypeError as error:
                    raise ValueError("inputs must have len(), or total must be provided") from error

            for index, stage in enumerate(self.stages):
                next_stage = self.stages[index + 1] if index + 1 < len(self.stages) else None
                stage.bind(self, index, next_stage)
            for stage in self.stages:
                stage.start()

            count = 0
            for root_id, payload in enumerate(inputs):
                if root_id >= self.total:
                    raise ValueError(f"inputs contains more than the declared total {self.total}")
                self.wait_for_slot()
                self.feed_root(root_id, payload)
                count += 1
            if count != self.total:
                raise ValueError(f"inputs contains {count} items but total is {self.total}")

            self.stages[0].submit("input_closed")
            with self.condition:
                while not self.input_complete and self.error is None:
                    self.condition.wait(timeout=0.1)
            if self.error is not None:
                raise self.error
            return list(self.results)
        finally:
            self.close()

    def wait_for_slot(self) -> None:
        if self.max_active_roots is None:
            return

        with self.condition:
            while self.active_roots >= self.max_active_roots and self.error is None:
                self.condition.wait(timeout=0.1)
        if self.error is not None:
            raise self.error

    def feed_root(self, root_id: int, payload: Mapping[str, Any]) -> None:
        if self.total is None or self.total <= 0:
            raise RuntimeError("cannot feed a root without a positive total")

        task = Task(
            payload,
            root_id,
            root_id / self.total,
            (root_id + 1) / self.total,
        )
        with self.condition:
            self.active_roots += 1
        self.stages[0].submit("task", task)
        self.stages[0].submit("root_closed", root_id)

    def root_finished(self, root_id: int) -> None:
        with self.condition:
            self.active_roots -= 1
            if self.active_roots < 0:
                raise RuntimeError(f"root {root_id} completed more than once")
            self.condition.notify_all()

    def input_finished(self) -> None:
        with self.condition:
            self.input_complete = True
            self.condition.notify_all()

    def update_progress(self, index: int, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        with self.progress_lock:
            current = self.progress_values[index]
            if value <= current:
                return
            self.progress_values[index] = value
            if index < len(self.progress_bars):
                self.progress_bars[index].update(value - current)

    def collect_result(self, payload: Mapping[str, Any]) -> None:
        with self.result_lock:
            self.results.append(dict(payload))

    def fail(
        self,
        error: BaseException,
        stage: str | None,
        boundary: str,
        task_context: Any,
    ) -> None:
        with self.condition:
            if self.error is None:
                location = f"stage {stage!r} {boundary}" if stage is not None else boundary
                detail = f"; task={task_context!r}" if task_context is not None else ""
                wrapped = RuntimeError(f"pipeline failed at {location}{detail}: {error}")
                wrapped.__cause__ = error
                self.error = wrapped
            self.stop_event.set()
            self.condition.notify_all()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()

        for stage in reversed(self.stages):
            stage.stop()
        for bar in self.progress_bars:
            bar.close()
