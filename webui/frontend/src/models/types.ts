import type { DagStatus } from '../types';

export type ModelId = 'trellis' | 'trellis2' | 'sam3d_object';

export type EnabledModelId = 'trellis2';

export type NodeInstanceId = string;

export type NodeKind =
  | 'trellis2_image_condition'
  | 'manual_symmetry'
  | 'detect_adjust_symmetry'
  | 'trellis2_vanilla_sparse_structure'
  | 'trellis2_vanilla_shape'
  | 'trellis2_symmetry_sparse_structure'
  | 'trellis2_symmetry_shape'
  | 'trellis2_texture';

export type OperationId = string;

export type DagLane = 'left' | 'main';

export type DagEdgeRoute = 'straight' | 'side_branch' | 'side_merge' | 'right_bypass';

export type ModelOption = {
  disabled: boolean;
  id: ModelId;
  label: string;
};

export type ModelDagNode = {
  id: NodeInstanceId;
  kind: NodeKind;
  label: string;
  operation: OperationId;
  shortLabel: string;
};

export type ModelDagEdge = {
  id: string;
  routeLabel?: string;
  source: NodeInstanceId;
  target: NodeInstanceId;
};

export type ModelDagLayout = {
  edges: Record<string, { route: DagEdgeRoute }>;
  nodes: Record<NodeInstanceId, { lane: DagLane; rank: number }>;
};

export type ViewerArtifactCandidate = {
  material: 'neutral' | 'source';
  nodeId: NodeInstanceId;
  roles: string[];
};

export type ViewerNodeRule = {
  artifactCandidates: ViewerArtifactCandidate[];
  showConfirmedSymmetryPreview?: boolean;
};

export type ModelViewerRules = Record<NodeInstanceId, ViewerNodeRule>;

export type ModelSpec = {
  dag: {
    edges: ModelDagEdge[];
    entryNodeId: NodeInstanceId;
    layout: ModelDagLayout;
    nodes: ModelDagNode[];
  };
  disabled: boolean;
  id: EnabledModelId;
  label: string;
  viewer: ModelViewerRules;
};

export type DagStatusByNode = Record<NodeInstanceId, DagStatus>;
