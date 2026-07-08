# WebUI Frontend And Backend Refactor Design

## Core Goal

The WebUI should support multiple model workflows, including TRELLIS.2, TRELLIS, and SAM-3D-Object. Each model can have its own DAG, while sharing common rendering, symmetry interaction, generation controls, viewer behavior, and backend execution infrastructure.

The design must keep these concepts separate:

```text
model
DAG topology
DAG layout
DAG renderer
node instance
node kind
node run
non-node action
execution
backend operation
artifact
key
lineage
display label
DAG short label
```

The frontend owns model-specific workflow structure and UI routing. The backend owns operation execution, artifact storage, key calculation support, cache lookup, and event delivery.

## DAG Model

The DAG is model-specific.

Each model decides:

```text
which node instances exist
how node instances connect
which node kind each instance uses
which backend operation each instance calls
how this model DAG is visually arranged
```

The DAG renderer is model-agnostic. It only draws the graph that it receives.

The renderer knows:

```text
node boxes or mobile dots
edge paths
inactive / current / completed status
desktop labels
mobile compact rendering
rounded edge and node styling
```

The renderer must not know:

```text
TRELLIS.2
manual symmetry semantics
texture generation semantics
backend operation names
artifact types
```

## Topology And Layout

DAG topology and DAG layout must be separate.

Topology describes the real workflow graph:

```ts
nodes: [
  { id: 'image_condition', kind: 'trellis2_image_condition' },
  { id: 'manual_symmetry', kind: 'manual_symmetry' },
  { id: 'detect_adjust_symmetry', kind: 'detect_adjust_symmetry' },
  { id: 'vanilla_sparse_structure', kind: 'trellis2_vanilla_sparse_structure' },
];

edges: [
  { source: 'image_condition', target: 'manual_symmetry' },
  { source: 'image_condition', target: 'vanilla_sparse_structure' },
  { source: 'vanilla_sparse_structure', target: 'detect_adjust_symmetry' },
];
```

Topology answers:

```text
what nodes exist
which node connects to which node
which next nodes are reachable
```

Layout describes how the topology is drawn:

```ts
nodeLayout: {
  image_condition: { lane: 'main', rank: 0 },
  vanilla_sparse_structure: { lane: 'main', rank: 1 },
  vanilla_shape: { lane: 'main', rank: 2 },
  manual_symmetry: { lane: 'left', rank: 3 },
  detect_adjust_symmetry: { lane: 'main', rank: 3 },
  symmetry_sparse_structure: { lane: 'main', rank: 4 },
};

edgeLayout: {
  'image_condition->manual_symmetry': { route: 'side_branch' },
  'manual_symmetry->symmetry_sparse_structure': { route: 'side_merge' },
  'vanilla_shape->texture': { route: 'bypass' },
};
```

Layout answers:

```text
which lane a node is drawn in
which rank a node is drawn at
whether manual symmetry is drawn on the left
how each edge route is drawn
how desktop and mobile DAGs are arranged
```

Manual symmetry is a shared node kind. Drawing it on the left side is a model-specific DAG layout choice.

## Frontend DAG Storage

The frontend stores model-specific DAG topology and layout.

The frontend needs this because it owns:

```text
model selection
DAG rendering
current node navigation
available next-node choices
model-specific workflow customization
```

The backend does not need the full UI DAG for this interaction model. The backend exposes atomic operations and validates whether each requested execution can run.

## Node Classification

### Fully Model-Agnostic Nodes

These node kinds are shared across models:

```text
manual_symmetry
detect_adjust_symmetry
```

`manual_symmetry` owns:

```text
major axis input
minor axis input
center input
axis shortcut buttons
axial / T / O / I family selection
fold input
point group label selection
confirm proposed symmetry
symmetry tuple output
symmetry preview
```

`detect_adjust_symmetry` owns:

```text
detect major axis
axis candidate selection
detect finer type
C2 axis selection
reflection plane selection
minor axis derivation
point group label selection
confirm proposed symmetry
symmetry tuple output
symmetry preview
```

The symmetry detection algorithm is shared. Different models can connect this node to different upstream and downstream nodes, but they do not need separate detection node implementations.

### Shared UI With Model-Specific Execution

`image_condition` has shared UI but model-specific execution.

Shared UI:

```text
image upload
paste image
drag and drop
image preview
generate condition button
condition ready state
next node choices
```

Model-specific execution:

```text
background removal strategy
vision encoder
condition artifact type
backend operation
available downstream nodes
```

### Model-Specific Generation Nodes

Generation nodes are model-specific because their parameter schemas, artifacts, checkpoints, and backend execution semantics can differ.

TRELLIS.2 examples:

```text
trellis2_vanilla_sparse_structure
trellis2_symmetry_sparse_structure
trellis2_vanilla_shape
trellis2_symmetry_shape
trellis2_texture
```

Future models can define their own corresponding node kinds.

## Reusable Parts Inside Model-Specific Nodes

Model-specific generation nodes still contain reusable model-agnostic pieces.

Reusable panel and control pieces:

```text
node panel shell
title / instruction / back button / scroll body
seed control
steps control
time step rescale control
classifier-free guidance strength control
classifier-free guidance duration control
classifier-free guidance rescale control
duration range control
progress display
metadata display
confirm and generate action
go to next step action
```

Reusable symmetry pieces:

```text
readonly symmetry tuple display
symmetry projection parameter controls
noise symmetry projection strength
symmetry projection strength
symmetry projection duration
symmetry preview
```

Model-specific pieces:

```text
backend operation name
input artifact types
output artifact types
parameter schema
default parameter values
shape mode logic
cascade upscale logic
max token logic
VRAM estimate logic
progress metadata fields
artifact routing
position in the model DAG
```

The shared layer should contain controls and reusable state patterns. It should not hide the explicit business layout of each concrete node panel.

## Node Run, Non-Node Action, And Execution

Backend communication is organized around `execution`.

There are two execution kinds:

```text
DAG node run
non-node action
```

Both can:

```text
produce a deterministic key
hit cache
run on the backend worker
emit started / progress / cached / completed / failed events
write artifacts or JSON results
write records
```

They differ in workflow meaning.

### DAG Node Run

A DAG node run is the formal execution of a DAG node.

It:

```text
belongs to the model DAG
produces a node_run_key
enters node lineage
can be used as input by downstream DAG nodes
can change the current workflow state
```

Examples:

```text
image condition generation
manual symmetry confirm
detect and adjust symmetry confirm
vanilla sparse structure generation
symmetry sparse structure generation
vanilla shape generation
symmetry shape generation
texture generation
```

For `detect_adjust_symmetry`, the intermediate detection probes are not node runs. Only `confirm proposed symmetry` produces the formal node run result.

### Non-Node Action

A non-node action is a backend execution attached to an existing run or artifact. It is not a DAG node.

It:

```text
does not appear in DAG topology
does not appear in DAG layout
does not change the current node
does not decide next-node routing
produces an action_key
can be cached
can produce JSON or artifacts
```

Examples:

```text
symmetry.detect_rotation_axes
symmetry.detect_reflection_planes_containing_axis
symmetry.detect_reflection_planes_perpendicular_to_axis
symmetry.detect_c2_axes_perpendicular_to_axis
trellis2.export_glb
```

Export is a non-node action. It is not a DAG node. It can appear as a sub-panel inside `vanilla_shape`, `symmetry_shape`, and `texture`, but it never enters DAG topology.

Export produces its own key:

```text
export_key = hash(source run/artifact key + export operation + export params)
```

Detection probes also produce action keys:

```text
detection_action_key = hash(source geometry key + detection operation + detection params)
```

## Event Stream

Node runs and non-node actions share the same event stream shape.

Frontend command types:

```text
execution.submit
execution.cancel
```

Backend event types:

```text
execution.started
execution.progress
execution.cached
execution.completed
execution.failed
```

Each event includes:

```ts
{
  type: 'execution.progress',
  requestId: string,
  sessionId: string,
  executionKind: 'node_run' | 'action',
  key?: string,
  payload: object,
}
```

Cache hits do not need fake progress. A cache hit can emit:

```text
execution.cached
execution.completed
```

## File Transfer And Artifact Storage

Large files should not travel through WebSocket.

Use:

```text
WebSocket:
  control messages
  execution submit/cancel
  started/progress/cached/completed/failed events
  JSON metadata
  artifact keys and URLs

HTTP:
  image upload
  GLB download/view
  exported files
  large artifact bytes
```

Frontend to backend file flow:

```text
browser File / Blob
-> HTTP multipart upload
-> backend artifact store
-> artifact_key
-> execution.submit references artifact_key
```

Backend to frontend file flow:

```text
execution.completed event
-> artifact_key / artifact_url / metadata
-> frontend loads artifact content with HTTP GET
```

For example, an uploaded image should become an input artifact before image condition generation:

```text
POST /api/artifacts/upload
  file=image.png
  kind=input_image

return:
  artifact_key
  content_hash
  mime_type
  width / height
```

Then the node run request only references the uploaded artifact:

```json
{
  "executionKind": "node_run",
  "operation": "trellis2.image_condition",
  "payload": {
    "inputImageKey": "artifact:..."
  }
}
```

The browser can keep a local `File` / `Blob` for preview before submission. Once the file participates in a formal execution, it should be uploaded and referenced by artifact key.

Fully in-memory storage is possible, but it should not be the formal workflow storage path because it breaks cache reuse, lineage recovery, refresh recovery, back navigation, and large artifact reuse. Memory should be used only for temporary frontend preview and temporary backend execution buffers.

The formal backend artifact store should use the local filesystem by default.

Default storage root:

```text
/tmp/symtrellis_webui
```

The root must be configurable:

```text
SYMTRELLIS_WEBUI_STORAGE=/tmp/symtrellis_webui
```

Recommended storage layout:

```text
/tmp/symtrellis_webui/
  sessions/
    {session_id}.json

  node_runs/
    ab/
      cd/
        {node_run_key}/
          record.json

  actions/
    ef/
      12/
        {action_key}/
          record.json

  artifacts/
    34/
      56/
        {artifact_key}/
          meta.json
          content.glb

  tmp/
    {request_id}/
```

The `ab/cd` directory shards come from the key prefix. This prevents too many files in one directory.

Internal records should use folders, not zip files.

Use:

```text
{node_run_key}/record.json
{action_key}/record.json
{artifact_key}/meta.json
{artifact_key}/content.*
```

Do not use zip as the internal storage format because records need local metadata reads, status updates, atomic writes, and easy debugging. Zip can be used later for user-facing package download, not for internal execution storage.

Artifact metadata should record:

```json
{
  "artifact_key": "...",
  "kind": "glb",
  "mime_type": "model/gltf-binary",
  "filename": "content.glb",
  "size_bytes": 123456,
  "created_at": "...",
  "source_execution_key": "..."
}
```

Writing should be atomic:

```text
write content into tmp
finish and close the file
rename into the final artifact directory
write artifact metadata
write completed execution record last
```

The completed record should be written only after all referenced artifacts are fully available.

## Key Design

Every successful node run or action produces a deterministic key.

The key must be based on canonical execution inputs, not on incidental UI state.

Keys should not include:

```text
session_id
request_id
timestamp
UI history stack
temporary panel state
```

Node run key common inputs:

```text
execution_kind = node_run
operation_id
operation_version
schema_version
model_id
model/checkpoint version
immediate parent node run keys
input artifact keys
normalized params
```

Action key common inputs:

```text
execution_kind = action
operation_id
operation_version
schema_version
source node run key
source artifact key
normalized params
```

Direct parent run keys are enough for key calculation because each parent key already commits to its own inputs and lineage.

The full lineage should still be stored in the execution record:

```text
parent_run_keys
ancestor_run_keys
source_artifact_keys
```

The UI history stack should not be used directly to compute keys. Two identical executions should produce the same key even if the user reached them through different UI backtracking steps.

Key calculation should use normalized params:

```text
stable field order
normalized floats where needed
canonical vectors
canonical symmetry tuple
explicit operation and schema versions
explicit model/checkpoint versions
```

Artifact keys should be content-addressed when possible:

```text
artifact_key = hash(bytes + artifact kind + canonical metadata that affects interpretation)
```

This lets different sessions share the same uploaded image or generated file reference when the content is identical.

## Session And Lineage

A session is a user workflow trial over a selected model DAG.

The session is mutable user interaction state. It should be randomly generated and should not be content-addressed.

Different sessions can share the same:

```text
artifact_key
node_run_key
action_key
```

Session state should record:

```text
session_id
selected_model_id
current_node_instance_id
current_node_run_key
node run history
action records attached to node runs or artifacts
selected DAG route choices
```

Node run records are global deterministic cache records. A session references them, but the record itself is not unique to one session.

Node run records represent formal workflow results:

```text
node_run_key
model_id
node_instance_id
node_kind
operation_id
operation_version
parent_run_keys
ancestor_run_keys
input_artifact_keys
params_hash
artifact_keys
metadata
status
created_at
```

Action records are also global deterministic cache records. They are attached to source runs or artifacts, not to DAG topology.

Action records represent auxiliary executions:

```text
action_key
model_id
action_kind
operation_id
operation_version
source_node_run_key
source_artifact_key
params_hash
artifact_keys
json_result
metadata
status
created_at
```

Back navigation should use session node history or node lineage. Export actions and detection probe actions do not participate in DAG back navigation.

## Backend Architecture

The backend should be organized around an execution registry, not around a mutable DAG object.

Rough structure:

```text
webui/backend/
  app.py
  settings.py
  schemas.py
  sessions.py
  records.py
  artifacts.py
  keys.py
  events.py
  worker.py
  executions.py
  operations.py

  node_runs/
    __init__.py
    image_condition.py
    manual_symmetry.py
    detect_adjust_symmetry.py
    trellis2_vanilla_sparse_structure.py
    trellis2_symmetry_sparse_structure.py
    trellis2_vanilla_shape.py
    trellis2_symmetry_shape.py
    trellis2_texture.py

  actions/
    __init__.py
    symmetry_detection.py
    trellis2_export_glb.py

  runtime/
    __init__.py
    trellis2.py
    detection.py
    previews.py
```

Responsibilities:

```text
app.py:
  FastAPI app, WebSocket endpoint, upload/download endpoints

settings.py:
  storage root, runtime paths, device, environment-specific configuration

schemas.py:
  Pydantic request/event/result schemas

sessions.py:
  session state, session history, references to node run/action records

records.py:
  global node run records, global action records, status, cache record lookup

artifacts.py:
  artifact storage, metadata, URLs, upload/download, cache lookup

keys.py:
  canonical serialization and hash helpers

events.py:
  event envelope helpers

worker.py:
  single GPU queue, cancellation, progress callback

executions.py:
  execution base classes and shared execution flow

operations.py:
  operation registry

node_runs/:
  DAG node run executors

actions/:
  non-node action executors

runtime/:
  model and algorithm runtime code
```

## Backend Execution Interfaces

Node runs and actions should share one execution base/protocol while keeping distinct semantics.

```python
class Execution:
    execution_kind: Literal["node_run", "action"]
    operation_id: str
    operation_version: str
    schema_version: str

    def normalize_params(self, payload: dict) -> dict:
        ...

    def resolve_inputs(self, context, payload: dict) -> ResolvedInputs:
        ...

    def build_key(self, context, inputs: ResolvedInputs, params: dict) -> str:
        ...

    async def run(
        self,
        context,
        inputs: ResolvedInputs,
        params: dict,
        emit: EventEmitter,
    ) -> ExecutionResult:
        ...
```

```python
class NodeRunExecution(Execution):
    execution_kind = "node_run"
    node_kind: str

    def build_lineage(self, inputs: ResolvedInputs) -> list[str]:
        ...
```

```python
class ActionExecution(Execution):
    execution_kind = "action"
    action_kind: Literal["analysis", "export"]
```

The shared base exists because node runs and actions share cache, queue, key, event, and artifact mechanics.

The subclasses exist because node runs enter DAG lineage and actions do not.

## Backend Submit Flow

Unified execution submit flow:

```python
async def submit_execution(request):
    executor = operation_registry.get(request.operation_id)

    params = executor.normalize_params(request.payload.params)
    inputs = executor.resolve_inputs(context, request.payload)
    key = executor.build_key(context, inputs, params)

    cached = records.get_completed_execution(key)
    if cached is not None:
        await emit("execution.cached", key=key, payload=cached.frontend_payload)
        await emit("execution.completed", key=key, payload=cached.frontend_payload)
        return

    record = records.create_pending_execution(
        key=key,
        request=request,
        inputs=inputs,
        params=params,
    )

    await emit("execution.started", key=key)

    result = await worker.run(
        executor.run,
        context=context,
        inputs=inputs,
        params=params,
        emit=emit,
    )

    artifacts.write_execution_result(key, result)
    records.mark_execution_completed(key, result)
    sessions.attach_execution_reference(request.session_id, key, result)

    await emit("execution.completed", key=key, payload=result.frontend_payload)
```

Operation executors should not decide UI navigation. They only execute a validated request.

## Frontend Architecture Implications

The frontend should have two submit paths:

```text
submitNodeRun()
submitAction()
```

Both send `execution.submit`, but they build different payloads and update different state after completion.

Frontend state should track:

```text
selectedModelId
currentNodeInstanceId
currentNodeRunKey
uploaded input artifact keys
nodeRunByInstance
actionResultsBySource
session history
DAG route choices
viewer state
```

The model DAG spec should include:

```ts
{
  modelId: 'trellis2',
  topology: {
    nodes: [...],
    edges: [...],
  },
  layout: {
    nodeLayout: {...},
    edgeLayout: {...},
  },
  nodes: {
    vanilla_sparse_structure: {
      kind: 'trellis2_vanilla_sparse_structure',
      operation: 'trellis2.sparse_structure.vanilla',
      defaults: {...},
      actions: {},
    },
    texture: {
      kind: 'trellis2_texture',
      operation: 'trellis2.texture.generate',
      defaults: {...},
      actions: {
        exportGlb: {
          operation: 'trellis2.export_glb',
          source: 'current_node_output',
        },
      },
    },
  },
}
```

Shared node panels should call the execution client through typed node/action request builders. Panels should not hand-build raw WebSocket messages.

Panels that accept large files should upload them through HTTP first, then submit executions by artifact key. The execution payload should not include large file bytes.

Node run completion updates:

```text
node run key
node output metadata
viewer content
DAG status
available next nodes
```

Action completion updates:

```text
action key
action result JSON or artifact link
viewer overlays if the action is a detection probe
export download state if the action is export
```

Action completion does not update DAG status.

## Naming Rules

Do not use `nat` or `native` for the vanilla route in the frontend workflow.

Use:

```text
vanilla
symmetry
manual
detect
texture
```

Separate internal names from display names:

```text
NodeInstanceId
NodeKind
OperationId
DisplayLabel
DagShortLabel
```

Model-specific node example:

```ts
{
  id: 'vanilla_sparse_structure',
  kind: 'trellis2_vanilla_sparse_structure',
  operation: 'trellis2.sparse_structure.vanilla',
  label: 'Vanilla sparse structure generation',
  shortLabel: 'VANILLA SS',
}
```

Shared node example:

```ts
{
  id: 'detect_adjust_symmetry',
  kind: 'detect_adjust_symmetry',
  operation: 'symmetry.confirm_detected_tuple',
  label: 'Detect and adjust symmetry',
  shortLabel: 'DETECT SYM',
}
```

Detection probe action example:

```ts
{
  action: 'detectRotationAxes',
  operation: 'symmetry.detect_rotation_axes',
}
```

Export action example:

```ts
{
  action: 'exportGlb',
  operation: 'trellis2.export_glb',
}
```

## Viewer Boundary

The viewer is model-agnostic.

It should not know concrete node kinds such as `trellis2_vanilla_shape` or `trellis2_texture`.

The viewer receives:

```text
ViewerContent
SymmetryOverlay[]
SymmetryPreview | null
selected overlay id
pickable overlay ids
overlay pick callback
```

The current node state decides which artifact or preview is shown. The viewer only renders the provided scene state.

## Final Boundary Summary

```text
DAG topology:
  model-specific workflow graph

DAG layout:
  model-specific visual placement

DAG renderer:
  model-agnostic SVG/UI renderer

Node kind:
  shared or model-specific panel/business behavior

Node run:
  formal DAG execution, produces node_run_key, enters lineage

Non-node action:
  auxiliary execution, produces action_key, does not enter DAG

Execution:
  shared backend abstraction for node runs and actions

Event stream:
  shared transport for started/progress/cached/completed/failed

Artifact store:
  backend-owned storage for large outputs and metadata

Frontend:
  owns model workflow, UI routing, viewer state, request submission

Backend:
  owns operation registry, execution, validation, caching, artifacts, events
```

The key design rule is:

```text
Workflow structure belongs to the frontend model DAG.
Execution correctness belongs to the backend operation system.
Node runs and non-node actions share execution mechanics but keep different workflow semantics.
```
