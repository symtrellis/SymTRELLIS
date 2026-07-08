import type { ActionKey, ActionRef, DagStatus, NodeRunKey, NodeRunRef } from '../types';
import type { DagStatusByNode, EnabledModelId, ModelDagEdge, ModelDagNode, ModelSpec, NodeInstanceId } from '../models/types';

export type WorkflowHistoryEntry = {
  edgeId: string | null;
  fromNodeId: NodeInstanceId | null;
  nodeId: NodeInstanceId;
};

export type WorkflowChosenEdge = {
  id: string;
  source: NodeInstanceId;
  target: NodeInstanceId;
};

export type WorkflowState = {
  actionRefsByKey: Record<ActionKey, ActionRef>;
  chosenEdges: WorkflowChosenEdge[];
  completedNodeIds: NodeInstanceId[];
  currentNodeId: NodeInstanceId | null;
  currentNodeRunKey: NodeRunKey | null;
  nodeHistory: WorkflowHistoryEntry[];
  nodeRunsByNode: Record<NodeInstanceId, NodeRunRef>;
  selectedModelId: EnabledModelId;
  sessionId: string;
};

export type WorkflowBackTransition = {
  resetNodeIds: NodeInstanceId[];
  state: WorkflowState;
};

export type WorkflowSuccessorRoute = {
  edge: ModelDagEdge;
  label: string;
  node: ModelDagNode;
};

export function createClientId() {
  if (globalThis.crypto.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  return [...bytes].map((byte, index) => {
    const hex = byte.toString(16).padStart(2, '0');
    return [4, 6, 8, 10].includes(index) ? `-${hex}` : hex;
  }).join('');
}

export function createInitialWorkflowState(): WorkflowState {
  return {
    actionRefsByKey: {},
    chosenEdges: [],
    completedNodeIds: [],
    currentNodeId: null,
    currentNodeRunKey: null,
    nodeHistory: [],
    nodeRunsByNode: {},
    selectedModelId: 'trellis2',
    sessionId: createClientId(),
  };
}

export function enterModelDag(state: WorkflowState, modelSpec: ModelSpec): WorkflowState {
  return {
    ...state,
    currentNodeId: modelSpec.dag.entryNodeId,
    nodeHistory: [{ edgeId: null, fromNodeId: null, nodeId: modelSpec.dag.entryNodeId }],
  };
}

export function chooseWorkflowEdge(
  state: WorkflowState,
  modelSpec: ModelSpec,
  targetNodeId: NodeInstanceId,
): WorkflowState {
  const sourceNodeId = state.currentNodeId;

  if (!sourceNodeId) {
    return state;
  }

  const edge = modelSpec.dag.edges.find(
    (candidate) => candidate.source === sourceNodeId && candidate.target === targetNodeId,
  );

  if (!edge) {
    return state;
  }

  const chosenEdge = { id: edge.id, source: edge.source, target: edge.target };
  return {
    ...state,
    chosenEdges: [...state.chosenEdges, chosenEdge],
    currentNodeId: targetNodeId,
    nodeHistory: [
      ...state.nodeHistory,
      { edgeId: edge.id, fromNodeId: sourceNodeId, nodeId: targetNodeId },
    ],
  };
}

export function canGoBack(state: WorkflowState) {
  return state.nodeHistory.length > 1;
}

export function goBackWorkflowNode(state: WorkflowState): WorkflowBackTransition {
  if (!canGoBack(state)) {
    return { resetNodeIds: [], state };
  }

  const prunedEntries = state.nodeHistory.slice(-1);
  const nodeHistory = state.nodeHistory.slice(0, -1);
  const currentEntry = nodeHistory[nodeHistory.length - 1];
  const activeNodeIds = new Set(nodeHistory.map((entry) => entry.nodeId));
  const activeEdgeIds = new Set(
    nodeHistory.flatMap((entry) => (entry.edgeId ? [entry.edgeId] : [])),
  );
  const resetNodeIds = [...new Set([...prunedEntries.map((entry) => entry.nodeId), currentEntry.nodeId])];
  const resetNodeIdSet = new Set(resetNodeIds);
  const nodeRunsByNode = Object.fromEntries(
    Object.entries(state.nodeRunsByNode).filter(
      ([nodeId]) => activeNodeIds.has(nodeId) && !resetNodeIdSet.has(nodeId),
    ),
  );
  const currentNodeRunKey = currentEntry.fromNodeId
    ? nodeRunsByNode[currentEntry.fromNodeId]?.key ?? null
    : null;

  return {
    resetNodeIds,
    state: {
      ...state,
      chosenEdges: state.chosenEdges.filter((edge) => activeEdgeIds.has(edge.id)),
      completedNodeIds: state.completedNodeIds.filter(
        (nodeId) => activeNodeIds.has(nodeId) && !resetNodeIdSet.has(nodeId),
      ),
      currentNodeId: currentEntry.nodeId,
      currentNodeRunKey,
      nodeHistory,
      nodeRunsByNode,
    },
  };
}

export function completeWorkflowNode(
  state: WorkflowState,
  nodeId: NodeInstanceId,
  nodeRun: NodeRunRef,
): WorkflowState {
  const completedNodeIds = state.completedNodeIds.includes(nodeId)
    ? state.completedNodeIds
    : [...state.completedNodeIds, nodeId];

  return {
    ...state,
    completedNodeIds,
    currentNodeRunKey: nodeRun.key,
    nodeRunsByNode: { ...state.nodeRunsByNode, [nodeId]: nodeRun },
  };
}

export function recordWorkflowAction(state: WorkflowState, actionRef: ActionRef): WorkflowState {
  return {
    ...state,
    actionRefsByKey: { ...state.actionRefsByKey, [actionRef.key]: actionRef },
  };
}

export function dagStatusForWorkflow(modelSpec: ModelSpec, state: WorkflowState) {
  return modelSpec.dag.nodes.reduce<DagStatusByNode>((statusByNode, node) => {
    const status: DagStatus =
      node.id === state.currentNodeId
        ? 'current'
        : state.completedNodeIds.includes(node.id)
          ? 'completed'
          : 'inactive';
    statusByNode[node.id] = status;
    return statusByNode;
  }, {});
}

export function currentNodeCompleted(state: WorkflowState) {
  return Boolean(state.currentNodeId && state.completedNodeIds.includes(state.currentNodeId));
}

export function successorRoutesForWorkflow(
  modelSpec: ModelSpec,
  state: WorkflowState,
): WorkflowSuccessorRoute[] {
  if (!state.currentNodeId) {
    return [];
  }

  return modelSpec.dag.edges
    .filter((edge) => edge.source === state.currentNodeId)
    .map((edge) => {
      const node = modelSpec.dag.nodes.find((candidate) => candidate.id === edge.target);
      return node ? { edge, label: edge.routeLabel ?? node.label, node } : null;
    })
    .filter((route): route is WorkflowSuccessorRoute => Boolean(route));
}
