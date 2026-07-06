export type ThemeMode = 'light' | 'dark';

export type Vector3 = [number, number, number];

export type NodeId =
  | 'img_cond'
  | 'nat_ss'
  | 'nat_shape'
  | 'detect_sym'
  | 'manual_sym'
  | 'sym_ss'
  | 'sym_shape'
  | 'texture';

export type DagStatus = 'inactive' | 'completed' | 'current';

export type DagNode = {
  id: NodeId;
  label: string;
  shortLabel: string;
};

export type DagEdge = {
  id: string;
  source: NodeId;
  target: NodeId;
};

export type SymmetryFamily = 'axial' | 'T' | 'O' | 'I';

export type SymmetryTuple = {
  center: Vector3;
  label: string;
  majorAxis: Vector3;
  minorAxis: Vector3;
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
  c2Axes: C2AxisCandidate[];
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
    };
