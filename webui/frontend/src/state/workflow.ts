import type {
  ActionKey,
  ActionResult,
  DagStatus,
  NodeRunKey,
  NodeRunResult,
  OutputRef,
  OutputRole,
  RestoredSessionRef,
  SessionId,
} from '../types';
import type {
  EnabledModelId,
  ModelDagEdge,
  ModelDagNode,
  ModelSpec,
  NodeInstanceId,
  OperationId,
} from '../models/types';

export type WorkflowNodeRun = {
  jsonResult: unknown;
  key: NodeRunKey;
  metadata: Record<string, unknown>;
  outputs: Record<OutputRole, OutputRef>;
};

export type WorkflowActionRun = {
  jsonResult: unknown;
  key: ActionKey;
  metadata: Record<string, unknown>;
  operationId: OperationId;
  outputs: Record<OutputRole, OutputRef>;
};

export type WorkflowHistoryEntry = {
  edgeId: string | null;
  nodeId: NodeInstanceId;
};

export type WorkflowState = {
  actionsBySourceRunKey: Record<NodeRunKey, WorkflowActionRun[]>;
  activeRunKeys: NodeRunKey[];
  currentNodeId: NodeInstanceId | null;
  nodeHistory: WorkflowHistoryEntry[];
  nodeRunsByNode: Record<NodeInstanceId, WorkflowNodeRun>;
  selectedModelId: EnabledModelId;
  sessionId: SessionId | null;
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

export function createInitialWorkflowState(): WorkflowState {
  return {
    actionsBySourceRunKey: {},
    activeRunKeys: [],
    currentNodeId: null,
    nodeHistory: [],
    nodeRunsByNode: {},
    selectedModelId: 'trellis2',
    sessionId: null,
  };
}

export function enterModelDag(state: WorkflowState, model: ModelSpec): WorkflowState {
  return {
    ...state,
    actionsBySourceRunKey: {},
    activeRunKeys: [],
    currentNodeId: model.dag.entryNodeId,
    nodeHistory: [{ edgeId: null, nodeId: model.dag.entryNodeId }],
    nodeRunsByNode: {},
    selectedModelId: model.id,
    sessionId: null,
  };
}

export function chooseWorkflowEdge(
  state: WorkflowState,
  model: ModelSpec,
  targetNodeId: NodeInstanceId,
): WorkflowState {
  if (!state.currentNodeId) {
    return state;
  }

  const edge = model.dag.edges.find(
    (candidate) => candidate.source === state.currentNodeId && candidate.target === targetNodeId,
  );

  if (!edge) {
    return state;
  }

  return {
    ...state,
    currentNodeId: targetNodeId,
    nodeHistory: [...state.nodeHistory, { edgeId: edge.id, nodeId: targetNodeId }],
  };
}

export function canGoBack(state: WorkflowState): boolean {
  return state.nodeHistory.length > 1;
}

export function goBackWorkflowNode(state: WorkflowState): WorkflowBackTransition {
  if (!canGoBack(state)) {
    return { resetNodeIds: [], state };
  }

  const nextHistory = state.nodeHistory.slice(0, -1);
  const resetNodeIds = state.nodeHistory.slice(nextHistory.length - 1).map((entry) => entry.nodeId);
  const retainedRunHistory = nextHistory.slice(0, -1);
  const activeNodeIds = new Set(retainedRunHistory.map((entry) => entry.nodeId));
  const nodeRunsByNode = Object.fromEntries(
    Object.entries(state.nodeRunsByNode).filter(([nodeId]) => activeNodeIds.has(nodeId)),
  );
  const activeRunKeys = retainedRunHistory
    .map((entry) => nodeRunsByNode[entry.nodeId]?.key)
    .filter((key): key is NodeRunKey => Boolean(key));
  const activeRunKeySet = new Set(activeRunKeys);

  return {
    resetNodeIds,
    state: {
      ...state,
      actionsBySourceRunKey: Object.fromEntries(
        Object.entries(state.actionsBySourceRunKey).filter(([sourceRunKey]) =>
          activeRunKeySet.has(sourceRunKey),
        ),
      ),
      activeRunKeys,
      currentNodeId: nextHistory[nextHistory.length - 1]?.nodeId ?? null,
      nodeHistory: nextHistory,
      nodeRunsByNode,
    },
  };
}

export function completeWorkflowNode(
  state: WorkflowState,
  nodeId: NodeInstanceId,
  result: NodeRunResult,
): WorkflowState {
  const nodeIndex = state.nodeHistory.findIndex((entry) => entry.nodeId === nodeId);
  const parentHistory = nodeIndex >= 0 ? state.nodeHistory.slice(0, nodeIndex) : state.nodeHistory;
  const activeRunKeys = [
    ...parentHistory
      .map((entry) => state.nodeRunsByNode[entry.nodeId]?.key)
      .filter((key): key is NodeRunKey => Boolean(key)),
    result.key,
  ];
  const activeNodeIds = new Set([...parentHistory.map((entry) => entry.nodeId), nodeId]);
  const activeRunKeySet = new Set(activeRunKeys);

  return {
    ...state,
    actionsBySourceRunKey: Object.fromEntries(
      Object.entries(state.actionsBySourceRunKey).filter(([sourceRunKey]) =>
        activeRunKeySet.has(sourceRunKey),
      ),
    ),
    activeRunKeys,
    nodeRunsByNode: {
      ...Object.fromEntries(
        Object.entries(state.nodeRunsByNode).filter(([candidateNodeId]) => activeNodeIds.has(candidateNodeId)),
      ),
      [nodeId]: {
        jsonResult: result.jsonResult,
        key: result.key,
        metadata: result.metadata,
        outputs: result.outputs,
      },
    },
    sessionId: result.sessionId,
  };
}

export function parentRunKeysForCurrentNode(state: WorkflowState): NodeRunKey[] {
  if (!state.currentNodeId) {
    return [];
  }

  const nodeIndex = state.nodeHistory.findIndex((entry) => entry.nodeId === state.currentNodeId);
  const parentHistory = nodeIndex >= 0 ? state.nodeHistory.slice(0, nodeIndex) : state.nodeHistory;

  return parentHistory
    .map((entry) => state.nodeRunsByNode[entry.nodeId]?.key)
    .filter((key): key is NodeRunKey => Boolean(key));
}

export function recordWorkflowAction(
  state: WorkflowState,
  sourceRunKey: NodeRunKey,
  operationId: OperationId,
  result: ActionResult,
): WorkflowState {
  const actionRun: WorkflowActionRun = {
    jsonResult: result.jsonResult,
    key: result.key,
    metadata: result.metadata,
    operationId,
    outputs: result.outputs,
  };

  return {
    ...state,
    actionsBySourceRunKey: {
      ...state.actionsBySourceRunKey,
      [sourceRunKey]: [
        ...(state.actionsBySourceRunKey[sourceRunKey] ?? []).filter((action) => action.key !== result.key),
        actionRun,
      ],
    },
  };
}

export function dagStatusForWorkflow(model: ModelSpec, state: WorkflowState): Record<NodeInstanceId, DagStatus> {
  return Object.fromEntries(
    model.dag.nodes.map((node) => {
      if (node.id === state.currentNodeId) {
        return [node.id, 'current'];
      }

      if (state.nodeRunsByNode[node.id]) {
        return [node.id, 'completed'];
      }

      return [node.id, 'inactive'];
    }),
  );
}

export function currentNodeCompleted(state: WorkflowState): boolean {
  return Boolean(state.currentNodeId && state.nodeRunsByNode[state.currentNodeId]);
}

export function successorRoutesForWorkflow(model: ModelSpec, state: WorkflowState): WorkflowSuccessorRoute[] {
  if (!state.currentNodeId) {
    return [];
  }

  return model.dag.edges
    .filter((edge) => edge.source === state.currentNodeId)
    .map((edge) => {
      const node = model.dag.nodes.find((candidate) => candidate.id === edge.target);

      return node
        ? {
            edge,
            label: edge.routeLabel ?? node.label,
            node,
          }
        : null;
    })
    .filter((route): route is WorkflowSuccessorRoute => Boolean(route));
}

export function chosenEdgeIdsForWorkflow(state: WorkflowState): string[] {
  return state.nodeHistory
    .map((entry) => entry.edgeId)
    .filter((edgeId): edgeId is string => Boolean(edgeId));
}

export function latestRunKeyForWorkflow(state: WorkflowState): NodeRunKey | null {
  return state.activeRunKeys[state.activeRunKeys.length - 1] ?? null;
}

export function workflowUrl(state: WorkflowState): string {
  if (!state.currentNodeId) {
    return '/';
  }

  const params = new URLSearchParams();
  const key = latestRunKeyForWorkflow(state);

  if (state.sessionId && key) {
    params.set('session', state.sessionId);
    params.set('key', key);
  }

  params.set('node', state.currentNodeId);
  return `/?${params.toString()}`;
}

export function restoreWorkflowSession(
  model: ModelSpec,
  restored: RestoredSessionRef,
  requestedNodeId?: NodeInstanceId | null,
): WorkflowState {
  const nodeRunsByKey = new Map(restored.nodeRuns.map((nodeRun) => [nodeRun.key, nodeRun]));
  const nodeRunsByNode: Record<NodeInstanceId, WorkflowNodeRun> = {};
  const nodeHistory: WorkflowHistoryEntry[] = [];

  for (const nodeRunKey of restored.activeRunKeys) {
    const nodeRun = nodeRunsByKey.get(nodeRunKey);
    const node = nodeRun
      ? model.dag.nodes.find((candidate) => candidate.operation === nodeRun.operationId)
      : undefined;

    if (node && nodeRun) {
      const previousNodeId = nodeHistory[nodeHistory.length - 1]?.nodeId ?? null;
      const edge = previousNodeId
        ? model.dag.edges.find((candidate) => candidate.source === previousNodeId && candidate.target === node.id)
        : null;

      nodeRunsByNode[node.id] = {
        jsonResult: nodeRun.jsonResult,
        key: nodeRun.key,
        metadata: nodeRun.metadata,
        outputs: nodeRun.outputs,
      };
      nodeHistory.push({ edgeId: edge?.id ?? null, nodeId: node.id });
    }
  }

  const lastCompletedNodeId = nodeHistory[nodeHistory.length - 1]?.nodeId ?? null;
  const requestedNode = requestedNodeId
    ? model.dag.nodes.find((candidate) => candidate.id === requestedNodeId)
    : undefined;

  if (requestedNode) {
    if (!lastCompletedNodeId && requestedNode.id === model.dag.entryNodeId) {
      nodeHistory.push({ edgeId: null, nodeId: requestedNode.id });
    } else if (lastCompletedNodeId && requestedNode.id !== lastCompletedNodeId) {
      const edge = model.dag.edges.find(
        (candidate) => candidate.source === lastCompletedNodeId && candidate.target === requestedNode.id,
      );

      if (edge) {
        nodeHistory.push({ edgeId: edge.id, nodeId: requestedNode.id });
      }
    }
  }

  const actionsBySourceRunKey = Object.fromEntries(
    Object.entries(restored.actions).map(([sourceRunKey, actions]) => [
      sourceRunKey,
      actions.map((action) => ({
        jsonResult: action.jsonResult,
        key: action.key,
        metadata: action.metadata,
        operationId: action.operationId,
        outputs: action.outputs,
      })),
    ]),
  );

  return {
    actionsBySourceRunKey,
    activeRunKeys: restored.activeRunKeys,
    currentNodeId: nodeHistory[nodeHistory.length - 1]?.nodeId ?? model.dag.entryNodeId,
    nodeHistory: nodeHistory.length > 0 ? nodeHistory : [{ edgeId: null, nodeId: model.dag.entryNodeId }],
    nodeRunsByNode,
    selectedModelId: model.id,
    sessionId: restored.sessionId,
  };
}
