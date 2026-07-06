下面是一份可以直接交给无上下文 AI 的系统化项目说明。它按“概念边界 → DAG → node 交互 → artifact → 前后端 → 实现顺序”组织，避免把临时检测结果、保存、导出、预览误认为 DAG artifact 或 node。

---

# SymTRELLIS WebUI 子项目设计说明

你要实现的是一个面向 TRELLIS.2 / SymTRELLIS 的交互式 WebUI。当前只实现 TRELLIS.2。TRELLIS 1 和 SAM3D Objects 以后再接入，因为它们对应的 mapper 还没有训练好。

这个项目不是 Gradio demo，不是 notebook wrapper，也不是通用 workflow engine。它是一个交互式 DAG 工作台。用户上传图像后，选择 TRELLIS.2，进入 TRELLIS.2 的 DAG。用户可以沿不同 edge 选择 manual symmetry、native generation、symmetry detection / adjustment、SymTRELLIS generation、texture generation。每个 graph-node 只产生当前阶段结果，用户看完结果后决定 rerun 或进入下一步。

当前目标部署环境是 Hugging Face Docker Space。前端用 React + Vite + React Flow + Three.js。后端用 FastAPI + Uvicorn + single GPU worker + WebSocket。

---

## 1. 核心概念

### 1.1 Pre-DAG state

上传图像不属于 DAG。

用户上传图像后，系统处于 Pre-DAG app state。此时还没有进入某个模型的 DAG。

Pre-DAG state 包括：

```text
uploaded_image
selected_model
```

当前只支持：

```text
selected_model = TRELLIS.2
```

正式 DAG 从 image condition 开始：

```text
trellis2_image_condition
```

它消费 Pre-DAG 的 `uploaded_image`。

不要把 `load_image` 设计成 graph-node。

---

### 1.2 Graph-node

graph-node 是 DAG 中的阶段。所有 graph-node 平级。

一个 graph-node 的职责是：

```text
读取输入 artifact / pre-DAG input
执行当前阶段
产生 pending result
用户接受后 commit 成该 node 的正式 artifact
```

graph-node 不决定下一步走哪里。下一步由 DAG edge 和用户选择决定。

例如：

```text
trellis2_image_condition
  output: image_condition_key / image condition result

possible next nodes:
  manual_symmetry
  trellis2_native_ss
```

这不是两个返回值。`trellis2_image_condition` 只产生 image condition 阶段结果。后续走哪条 edge 是用户在 UI 中选择。

---

### 1.3 Artifact

artifact 只对应 graph-node 的最终阶段结果。

artifact 不是 detection 的中间候选，不是前后端通信 JSON，不是用户下载文件，不是 preview，不是 export 结果。

正式 artifact 的唯一物理形式是：

```text
zip file
```

每个 committed artifact 都是一个 zip。zip 内必须包含一个 JSON 文件，用来说明：

```text
这个 artifact 属于哪个 node
artifact key 是什么
前置 artifact keys 是什么
parameters 是什么
payload 应该如何被后续 node 读取
```

概念结构：

```text
artifact.zip
  metadata.json
  payload files
```

metadata 至少包含：

```text
node_id
artifact_key
previous_artifact_keys
parameters
payload_description
```

payload 内容由对应 node 决定。

例子：

```text
trellis2_native_ss artifact:
  payload = occ / sparse structure information

trellis2_sym_shape artifact:
  payload = shape latent / shape state needed by later stages

trellis2_texture artifact:
  payload = texture latent / texture state needed by export
```

---

### 1.4 Detection intermediate result is not artifact

`detect_adjust_symmetry` 内部会有多轮前后端通信，但这些通信结果不是 artifact。

例如：

```text
rotation axis candidates
axis position
fold
score
reflection plane candidates
C2 axis candidates
candidate group labels
```

这些都只是 JSON 临时数据，用于当前 node 内部交互。它们不写入 artifact zip，不进入 storage，不参与 artifact key。

`detect_adjust_symmetry` 只有在用户最终确认后，才产生该 node 的最终结果。最终结果是四元组：

```text
center
major_axis
minor_axis
label
```

这才是该 node 的正式结果，可以 commit 成 artifact zip。

---

### 1.5 Pending result vs committed artifact

每个 node 运行完成后，结果先留在内存里，叫 pending result。

用户可以在当前 node 内反复尝试参数。新的 pending result 会覆盖旧的 pending result。

只有当用户选择进入下一步时，当前 pending result 才 commit 成 artifact zip，写入 storage。

```text
pending result:
  在内存中
  可以被 rerun 覆盖
  不写入 storage
  不进入 artifact history

committed artifact:
  zip file
  写入 storage
  有 artifact key
  可以作为后续 node 输入
  可以回退到
  可以被 cache 命中
```

这是项目的 rerun 策略。

---

### 1.6 Export

Export 不是 graph-node。

Export 不是 artifact。

Export 是 artifact action。

当前只支持 GLB。

```text
current terminal artifact + export resolution -> downloadable GLB
```

GLB 是用户下载文件，不进入 DAG，不作为后续 node 输入，不是 artifact zip。

不要实现 preview video、MP4、多格式导出。

---

## 2. Graph-node pool

所有 graph-node 平级。每个 node 只带一个属性：是否 model-specific。

当前 graph-node pool 是：

```text
model-agnostic:
  manual_symmetry
  detect_adjust_symmetry

trellis2-specific:
  trellis2_image_condition
  trellis2_native_ss
  trellis2_native_shape
  trellis2_sym_ss
  trellis2_sym_shape
  trellis2_texture
```

`manual_symmetry` 和 `detect_adjust_symmetry` 不属于 TRELLIS.2。它们是通用 symmetry graph-node。

`trellis2_*` 节点依赖 TRELLIS.2 checkpoint、sampler、latent layout、decoder、SymTRELLIS mapper 或 texture model，所以带 `trellis2_` 前缀。

---

## 3. TRELLIS.2 DAG

DAG 是真正的图，不是线性 pipeline。

Pre-DAG：

```text
uploaded_image
selected_model = TRELLIS.2
```

DAG root：

```text
trellis2_image_condition
```

完整 edge list：

```text
trellis2_image_condition -> manual_symmetry
trellis2_image_condition -> trellis2_native_ss

manual_symmetry -> trellis2_sym_ss

trellis2_native_ss -> trellis2_native_shape

trellis2_native_shape -> detect_adjust_symmetry
trellis2_native_shape -> trellis2_texture

detect_adjust_symmetry -> trellis2_sym_ss

trellis2_sym_ss -> trellis2_sym_shape

trellis2_sym_shape -> trellis2_texture
```

`trellis2_sym_ss` 是多输入节点。它需要：

```text
image_condition_key
symmetry artifact
```

symmetry artifact 可以来自：

```text
manual_symmetry
detect_adjust_symmetry
```

所有 DAG node 的 config 中都应该包含第一步 `trellis2_image_condition` 的 key。这样后续 node 不需要长期存储 image condition tensor。后续 node 可以通过 `image_condition_key` 识别它属于哪个 image condition lineage；如果 tensor 不在内存中，可以按需要恢复或重算。

---

## 4. DAG 支持的用户路径

这是同一个 DAG 的不同路径，不是三条独立 pipeline。

### 4.1 Manual symmetry path

```text
uploaded_image
  -> trellis2_image_condition
  -> manual_symmetry
  -> trellis2_sym_ss
  -> trellis2_sym_shape
  -> optional trellis2_texture
```

### 4.2 Detect then enforce path

```text
uploaded_image
  -> trellis2_image_condition
  -> trellis2_native_ss
  -> trellis2_native_shape
  -> detect_adjust_symmetry
  -> trellis2_sym_ss
  -> trellis2_sym_shape
  -> optional trellis2_texture
```

### 4.3 Vanilla TRELLIS.2 path

```text
uploaded_image
  -> trellis2_image_condition
  -> trellis2_native_ss
  -> trellis2_native_shape
  -> optional trellis2_texture
```

这条路径没有 symmetry enforcement。`trellis2_native_shape` 可以直接进入 `trellis2_texture`。

---

## 5. Terminal artifacts

没有 ending node。

用户停在哪个 artifact 上，哪个 artifact 就是当前结果。

当前 terminal artifacts：

```text
trellis2_native_shape
  vanilla shape result

trellis2_sym_shape
  symmetry-enforced shape result

trellis2_texture
  textured result
```

`trellis2_texture` 可以从两类输入产生：

```text
trellis2_native_shape -> trellis2_texture
trellis2_sym_shape    -> trellis2_texture
```

是否 native / symmetry-enforced 不需要额外字段专门标注，可以从 artifact lineage 追溯。

---

## 6. Symmetry 参数

用户层 symmetry 参数只有四个：

```text
center
major_axis
minor_axis
label
```

不要加入 include_identity、confidence、source 等用户层参数。

坐标约定：

```text
z-up
canonical cube = [-0.5, 0.5]^3
```

---

## 7. Node 交互模式

### 7.1 trellis2_image_condition

纯计算节点，无 3D 交互。

交互模式：

```text
运行
显示进度
输出 image_condition_key / pending result
用户接受后 commit
```

该 node 是 DAG root。它消费 Pre-DAG 的 uploaded image。

---

### 7.2 trellis2_native_ss / trellis2_sym_ss

计算 + voxel 可视化，无编辑交互。

交互模式：

```text
用户设置参数
运行
显示进度
输出 ss / occupancy pending result
Three.js 可视化 voxel / occupancy
用户 rerun 或接受进入下一步
```

ss 需要可视化，但没有编辑交互。

---

### 7.3 trellis2_native_shape / trellis2_sym_shape

计算 + shape 可视化，无编辑交互。

交互模式：

```text
用户设置参数
运行
显示进度
输出 shape pending result
Three.js 显示 shape
用户 rerun / 接受进入后继节点 / 停在当前 terminal artifact
```

`trellis2_native_shape` 可以继续到：

```text
detect_adjust_symmetry
trellis2_texture
```

`trellis2_sym_shape` 可以继续到：

```text
trellis2_texture
```

---

### 7.4 trellis2_texture

计算 + textured result 可视化，无编辑交互。

交互模式：

```text
输入 native shape 或 sym shape
运行
显示进度
输出 texture pending result
Three.js 显示 textured result
用户 rerun / 停在当前 terminal artifact / export GLB
```

texture 是 graph-node，不是 render，不是 preview video。

---

### 7.5 manual_symmetry

model-agnostic 手动编辑节点。

交互模式：

```text
用户编辑 center / major_axis / minor_axis / label
前端实时显示 symmetry visualization
用户确认
输出 symmetry pending result
进入下一步时 commit 成 artifact zip
```

manual symmetry 的对称群空间划分、透明 sector、等价类示意，优先作为纯前端 Three.js 行为实现，不需要后端。

---

### 7.6 detect_adjust_symmetry

model-agnostic 多阶段交互节点。它在 DAG 上仍然是一个 graph-node，但内部有多轮前后端 JSON 通信。

#### 阶段 A：detect rotation axes

后端执行 rotation axis detection。

返回 JSON：

```text
axis candidates
axis position
axis direction
fold
score
```

前端负责：

```text
可视化所有候选 rotation axis
显示 fold / score
用户鼠标选择一个 major axis
```

这些候选数据不是 artifact。

#### 阶段 B：axis regularization

用户选中 major axis 后，前端提供选项：

```text
regularize:
  如果 axis 接近 x/y/z 标准轴，则可把它正规化为标准轴

origin as center:
  如果启用，则 center = [0,0,0]
```

这一步主要是前端逻辑。

#### 阶段 C：secondary symmetry detection

后端继续检测：

```text
C2 axes
reflection planes containing major axis
reflection planes perpendicular to major axis
```

返回 JSON candidates。

前端负责：

```text
可视化垂直于主轴的 C2 axes
可视化相关 reflection planes
辅助用户选择 minor_axis
辅助用户判断 label，例如 Cnv / Dn 等
```

这些候选数据也不是 artifact。

#### 阶段 D：special groups

对于特殊多面体群，可以基于 rotation-axis detection 阶段检测到的非 major axes 生成可能的 minor_axis 候选，并让用户选。

#### 阶段 E：final confirmation

label 选择阶段，前端实时渲染该 symmetry group 对应的透明空间划分 / 等价类 / sector visualization。

最终用户确认四元组：

```text
center
major_axis
minor_axis
label
```

确认后，这个 node 才产生最终 pending result。进入下一步时，该 result 才 commit 成 artifact zip。

---

## 8. Artifact key

artifact key 是 SHA256。

规则：

```text
key = sha256(previous_node_keys + parameters)
```

parameters 包括 seed。

不包含：

```text
model version
code version
pipeline version
checkpoint version
```

所有 DAG node 的 config 里都包含第一步 `trellis2_image_condition` 的 key。

---

## 9. Storage

所有 committed artifact 都存入 storage。

storage root 可配置。可以是普通本地目录，也可以是 `/dev/shm`。

所有正式 artifact 都是 zip。

pending result 在内存里，不落盘。

---

## 10. Session

刷新网页默认不保留 session。

如果 URL 中带 session id，可以恢复 session。

Session 记录：

```text
selected_model
uploaded_image
current node
current pending result
committed artifact history
current UI state
```

回退规则：

```text
只允许回退到上一步
可以连续点击上一步
不做 branch manager
放弃的 committed artifact 留在 storage/cache
```

回退只是 session 指针变化，不删除 artifact。

---

## 11. Cache

Cache 通过 artifact key 进行。

运行某 node 前，后端计算：

```text
sha256(previous_node_keys + parameters)
```

如果 storage 中已有对应 artifact zip，则直接复用。

如果没有，则运行 node，产生 pending result。用户接受并进入下一步时，commit 成 artifact zip。

Detection 中间 JSON 不参与 cache，不进入 storage。

---

## 12. Export action

Export 不是 DAG node。

Export 输入：

```text
当前 terminal artifact
export resolution
```

Export 输出：

```text
downloadable GLB
```

Export 和后端通信，因为它需要根据 resolution 生成 GLB。

Export 需要进度事件：

```text
started
progress: generating GLB at resolution ...
progress: packaging GLB
finished: GLB ready
failed
```

Export 不产生 artifact zip，不进入 DAG。

当前只支持 GLB。不要实现 preview video、MP4、多格式导出。

---

## 13. WebSocket

WebSocket 只负责状态和进度，不定义 DAG。

统一事件类型：

```text
started
progress
preview_ready
finished
failed
```

每个 node 的 progress 内容可以不同。

`finished` 后，前端展示 pending result。用户决定 rerun 或接受进入下一步。

不实现 cancel。运行时前端显示 busy。

Detection 中间通信可以用 JSON API 或 WebSocket，但 detection candidates 只是临时数据，不是 artifact。

---

## 14. WebUI 文件结构

WebUI 放在仓库顶层 `webui/`。它是应用代码，不放进 `symtrellis/`、`inference/`、`trainer/`、`dataset/` 或 `examples/`。

```text
webui/
  README.md

  backend/
    app.py
    settings.py
    dag.py
    schemas.py
    sessions.py
    artifacts.py
    worker.py
    export_glb.py

    nodes/
      __init__.py
      manual_symmetry.py
      detect_adjust_symmetry.py
      trellis2_image_condition.py
      trellis2_sparse_structure.py
      trellis2_shape.py
      trellis2_texture.py

    runtime/
      __init__.py
      trellis2.py
      detection.py
      previews.py

  frontend/
    package.json
    vite.config.ts
    index.html
    tsconfig.json

    src/
      main.tsx
      App.tsx
      api.ts
      types.ts
      dag.ts
      state.ts
      styles.css

      layout/
        AppLayout.tsx

      panels/
        DagPanel.tsx
        NodePanel.tsx
        ProgressPanel.tsx

      node_panels/
        ImageConditionPanel.tsx
        ManualSymmetryPanel.tsx
        DetectAdjustSymmetryPanel.tsx
        SparseStructurePanel.tsx
        ShapePanel.tsx
        TexturePanel.tsx

      viewer/
        ThreeViewer.tsx
        viewerUtils.ts
```

不建立额外的应用层级，例如 `apps/webui/`。不建立 `backend/api/`、`backend/core/`、`backend/schemas/`、`backend/static.py`、`backend/events.py`、`backend/trellis2/`、`backend/detection/`。不建立 `frontend/src/api/`、`frontend/src/types/`、`frontend/src/state/`、`frontend/src/graph/`、`frontend/src/viewer/scene.ts` 这类拆分。

---

## 15. 后端文件边界

后端使用：

```text
FastAPI
Uvicorn
single GPU worker
WebSocket
artifact zip storage
```

GPU 执行：

```text
single GPU worker
一次执行一个 GPU-heavy task
多 session 可以存在，但 GPU job 排队
不实现 cancel
```

`backend/app.py`

FastAPI 唯一入口。负责注册 API、WebSocket 和 frontend static serving，并连接 session、artifact、worker、node runner。它不承载 TRELLIS.2 计算细节。

`backend/settings.py`

配置文件。负责 storage root、checkpoint path、frontend dist path、device、HF Space 相关环境变量。它不保存运行状态。

`backend/dag.py`

TRELLIS.2 DAG 的事实来源。负责 node id、edge、terminal node、可用后继 node。它不执行 node，不读写 artifact。

`backend/schemas.py`

Pydantic schema 集中定义。覆盖 session、artifact、node params/result、symmetry tuple、detection candidates、WebSocket event、export request/response。它只定义数据形状，不放业务流程。

`backend/sessions.py`

Session 状态管理。负责 uploaded image、selected model、current node、pending result、artifact history、一步回退。Session 是用户游标，不是 artifact storage。

`backend/artifacts.py`

Artifact zip 管理。负责 key、metadata、zip pack/unpack、storage read/write、cache lookup。Cache 不单独成文件，因为 cache 就是 artifact key lookup。

`backend/worker.py`

Single GPU worker。负责 job queue、busy 状态、progress event 分发、运行 node/export job。它不定义 DAG，也不保存 artifact。

`backend/export_glb.py`

GLB export action。它读取 terminal artifact 并生成 downloadable GLB。Export 不是 DAG node，不产生 artifact zip。

`backend/nodes/`

Graph-node runner 层。这里表达 node 语义：读取输入 artifact / pre-DAG input，调用 runtime，产出 pending result 和 artifact payload。

```text
manual_symmetry.py             # manual_symmetry 最终四元组确认
detect_adjust_symmetry.py      # detect_adjust_symmetry 最终四元组确认
trellis2_image_condition.py    # trellis2_image_condition
trellis2_sparse_structure.py   # trellis2_native_ss / trellis2_sym_ss
trellis2_shape.py              # trellis2_native_shape / trellis2_sym_shape
trellis2_texture.py            # trellis2_texture
```

`manual_symmetry.py` 和 `detect_adjust_symmetry.py` 不负责复杂 Three.js 可视化。它们只负责最终四元组的后端校验和 pending result。

`backend/runtime/`

真实计算层。这里不放 app 状态，不决定 DAG 走向。

```text
trellis2.py    # TRELLIS.2 lazy loading、sampler、mapper/projector、texture 调用
detection.py   # detection 阶段函数，返回临时 JSON candidates
previews.py    # voxel / shape / texture 结果的 preview 数据准备
```

`runtime/detection.py` 的 candidates 不进 artifact，不进 storage，不参与 artifact key。

`runtime/previews.py` 不负责 manual symmetry 的 group sector、透明空间划分、axis/plane/gizmo 可视化。这些属于前端 `ThreeViewer.tsx` 和 `viewerUtils.ts`。

---

## 16. 前端文件边界

前端使用：

```text
React
Vite
React Flow
Three.js
WebSocket client
```

不要使用 Gradio。
不要手写大型 static HTML。
不要使用 server-side template。
不要使用 Next.js 或 SSR。

页面结构：

```text
左侧：DAG panel
中间：当前 node control panel
右侧：Three.js viewer
底部或侧边：progress / logs / artifact actions
```

DAG panel 必须显示真实 graph，不是线性 stepper。当前 node finished 后，前端不自动前进。它显示 pending result 和可用后继 node。用户选择后继 node 时，当前 pending result commit 成 artifact zip，然后进入下一 node。

`frontend/src/api.ts`

REST 和 WebSocket client。负责和后端通信，不保存 UI 状态。

`frontend/src/types.ts`

TypeScript 类型集中定义。负责 API、DAG、artifact、session、symmetry、events、preview 的类型。

`frontend/src/dag.ts`

前端 DAG 展示数据。node id 和 edge 必须与 `backend/dag.py` 一致。

`frontend/src/state.ts`

前端状态。负责 session、current node、pending result、job progress、preview payload、当前选择的 detection candidate / symmetry tuple。

`frontend/src/layout/AppLayout.tsx`

整体布局。负责组织 DAG panel、node panel、viewer、progress panel。

`frontend/src/panels/`

通用面板层：

```text
DagPanel.tsx       # React Flow DAG
NodePanel.tsx      # current node panel 容器
ProgressPanel.tsx  # job progress / logs
```

`frontend/src/node_panels/`

按 node 交互拆分：

```text
ImageConditionPanel.tsx
ManualSymmetryPanel.tsx
DetectAdjustSymmetryPanel.tsx
SparseStructurePanel.tsx
ShapePanel.tsx
TexturePanel.tsx
```

`ManualSymmetryPanel.tsx` 和 `DetectAdjustSymmetryPanel.tsx` 把 axis、plane、sector、candidate selection 状态传给 viewer。检测中间结果只是临时 UI 数据。

`frontend/src/viewer/`

Three.js viewer：

```text
ThreeViewer.tsx   # scene/camera/renderer/controls、voxel/mesh/texture/symmetry/candidates 显示
viewerUtils.ts    # geometry/material/dispose/坐标转换等 viewer 工具
```

Manual symmetry 的 group visualization、透明 sector、axis/plane/gizmo 优先在这里完成，不放到后端 preview。

---

## 17. HF Docker Space 部署

目标部署是 Hugging Face Docker Space。

部署约束：

```text
一个 Docker container
一个对外端口
FastAPI serve frontend + API + WebSocket + GLB download
storage_root 可配置
Docker build 阶段不运行 GPU 代码
运行时才加载 GPU 模型
```

不要设计成必须两个外部服务端口。

---

## 18. 工作推进顺序

工作按交互风险和依赖关系推进。最难的交互先闭环。

```text
1. detect_adjust_symmetry 交互
   使用已有 mesh 或程序构造 mesh
   验证多阶段后端 JSON 通信
   验证 Three.js axis / plane / group visualization
   验证最终四元组确认

2. Three.js symmetry visualization
   major axis
   minor axis
   reflection planes
   C2 axes
   group sector / 等价类透明示意

3. FastAPI + React/Vite + WebSocket 壳
   一个服务
   一个端口
   前端和后端可通信

4. React Flow 显示 TRELLIS.2 DAG
   graph-node 平级
   edge 正确
   当前 node 和可用后继 node 正确

5. session state
   current node
   current pending result
   committed artifact history
   one-step back

6. artifact zip storage
   metadata.json
   payload
   key lookup
   commit artifact

7. pending / commit / back 机制
   node result 先在内存
   进入下一步才落盘
   rerun 覆盖 pending result

8. node runners 连通
   验证 DAG / session / artifact 全流程

9. manual_symmetry
   复用 Three.js symmetry visualization
   输出四元组

10. trellis2_image_condition
    产生 image_condition_key
    后续 node config 包含该 key

11. trellis2_sym_ss

12. trellis2_sym_shape

13. export GLB action

14. trellis2_native_ss

15. trellis2_native_shape

16. 将真实 native_shape 接入 detect_adjust_symmetry

17. trellis2_texture
```

---

## 19. 实现原则

实现时坚持以下原则：

```text
DAG 是数据，不是线性代码流程。
graph-node 平级。
model-specific 是 node 属性，不是层级。
artifact 只对应 node 最终结果。
detection intermediate JSON 不是 artifact。
artifact 只有 zip 一种正式物理形式。
pending result 和 committed artifact 分开。
所有 DAG node config 包含 image_condition_key。
export 是 action，不是 node。
save/download/preview 不是 node。
texture 是 graph-node，不是 render/video。
session 是用户游标，不是 artifact。
前端显示 DAG，后端执行 node。
single GPU worker 排队执行 GPU-heavy task。
```

这就是当前完整设计。
