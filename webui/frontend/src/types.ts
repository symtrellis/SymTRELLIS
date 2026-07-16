export type ThemeMode = 'light' | 'dark';

export type Vector3 = [number, number, number];

export type DagStatus = 'inactive' | 'completed' | 'current';

export type UploadKey = string;

export type NodeRunKey = string;

export type ActionKey = string;

export type RequestId = string;

export type SessionId = string;

export type SessionRevision = number;

export type OutputRole = string;

export type ExecutionProgress = {
  progress: number;
  stage: string;
};

export type UploadRef = {
  contentHash: string;
  filename: string;
  mimeType: string;
  uploadKey: UploadKey;
};

export type OutputRef = {
  filename: string;
  metadata: Record<string, unknown>;
  role: OutputRole;
  url: string;
};

export type NodeRunResult = {
  cached: boolean;
  jsonResult: unknown;
  key: NodeRunKey;
  metadata: Record<string, unknown>;
  outputs: Record<OutputRole, OutputRef>;
  sessionId: SessionId;
  sessionRevision: SessionRevision;
};

export type ActionResult<JsonResult = unknown> = {
  cached: boolean;
  jsonResult: JsonResult;
  key: ActionKey;
  metadata: Record<string, unknown>;
  outputs: Record<OutputRole, OutputRef>;
  sessionId: SessionId;
  sessionRevision: SessionRevision;
};

export type NodeRunRecord = {
  ancestorRunKeys: NodeRunKey[];
  inputUploadKeys: UploadKey[];
  jsonResult: unknown;
  key: NodeRunKey;
  metadata: Record<string, unknown>;
  modelId: string;
  operationId: string;
  operationVersion: string;
  outputs: Record<OutputRole, OutputRef>;
  params: Record<string, unknown>;
  parentRunKeys: NodeRunKey[];
};

export type ActionRecord = {
  jsonResult: unknown;
  key: ActionKey;
  metadata: Record<string, unknown>;
  operationId: string;
  operationVersion: string;
  outputs: Record<OutputRole, OutputRef>;
  params: Record<string, unknown>;
  sourceNodeRunKey: NodeRunKey;
};

export type RestoredSessionRef = {
  actions: Record<NodeRunKey, ActionRecord[]>;
  activeRunKeys: NodeRunKey[];
  modelId: string;
  nodeRuns: NodeRunRecord[];
  sessionId: SessionId;
  sessionRevision: SessionRevision;
};

export type SymmetryFamily = 'axial' | 'T' | 'O' | 'I';

export type SymmetryTuple = {
  center: Vector3;
  label: string;
  majorAxis: Vector3;
  minorAxis: Vector3;
};

export type RotationAxisDetectionResult = {
  axis: Vector3;
  dbscan_label: number;
  fold_e: number;
  fold_i: number;
  q: Vector3;
  ratio: number;
  rmse: number;
};

export type ReflectionPlaneDetectionResult = {
  c: number;
  dbscan_label: number;
  fold_i_val: number;
  n: Vector3;
  ratio: number;
  rmse: number;
};

export type FinerReflectionPlaneDetectionResult = ReflectionPlaneDetectionResult & {
  c_cor: number;
  fold_pred?: number;
  n_cor: Vector3;
};

export type C2AxisDetectionResult = {
  axis: Vector3;
  axis_cor: Vector3;
  dbscan_label: number;
  fold_c2: number;
  fold_i_val: number;
  q: Vector3;
  q_cor: Vector3;
  ratio: number;
  rmse: number;
};

export type FinerSymmetryDetectionResult = {
  c2_axes_perpendicular_to_axis: C2AxisDetectionResult[];
  reflection_planes_containing_axis: FinerReflectionPlaneDetectionResult[];
  reflection_planes_perpendicular_to_axis: FinerReflectionPlaneDetectionResult[];
};

export type RotationAxisCandidate = {
  axis: Vector3;
  center: Vector3;
  color: string;
  dbscanLabel: number;
  foldE: number;
  foldI: number;
  id: string;
  ratio: number;
  rmse: number;
};

export type C2AxisCandidate = {
  axis: Vector3;
  axisCorrected: Vector3;
  center: Vector3;
  centerCorrected: Vector3;
  color: string;
  dbscanLabel: number;
  foldC2: number;
  foldIValidation: number;
  id: string;
  ratio: number;
  rmse: number;
};

export type ReflectionPlaneCandidate = {
  color: string;
  dbscanLabel: number;
  foldIValidation: number;
  foldPred?: number;
  id: string;
  normal: Vector3;
  normalCorrected: Vector3;
  ratio: number;
  rmse: number;
  role: 'contains_major_axis' | 'perpendicular_to_major_axis';
};

export type FinerSymmetryResult = {
  c2AxesPerpendicularToAxis: C2AxisCandidate[];
  reflectionPlanesContainingAxis: ReflectionPlaneCandidate[];
  reflectionPlanesPerpendicularToAxis: ReflectionPlaneCandidate[];
};

export type SymmetryOverlay =
  | {
      axis: Vector3;
      center: Vector3;
      color: string;
      fold?: number;
      id: string;
      kind: 'rotation_axis' | 'c2_axis';
      label: string;
    }
  | {
      center: Vector3;
      color: string;
      id: string;
      kind: 'reflection_plane';
      label: string;
      majorAxis: Vector3;
      normal: Vector3;
      role: ReflectionPlaneCandidate['role'];
      shape: 'disk' | 'square';
    };
