import type { ActionKey, ActionKind, ArtifactKey, ArtifactRef, NodeRunKey, RequestId } from '../types';
import type { ModelId, NodeInstanceId, NodeKind, OperationId } from '../models/types';
import { postJson } from './client';

// BACKEND_PROTOCOL_PENDING: exact request params and metadata fields are finalized with each backend node/action.
// Successful artifactRefs must only point to artifacts that are already persisted and fetchable.
export type SubmitNodeRunRequest = {
  executionKind: 'node_run';
  inputArtifactKeys: ArtifactKey[];
  modelId: ModelId;
  nodeInstanceId: NodeInstanceId;
  nodeKind: NodeKind;
  operationId: OperationId;
  parentRunKeys: NodeRunKey[];
  params: Record<string, unknown>;
  requestId: RequestId;
  sessionId: string;
  type: 'execution.submit';
};

export type SubmitNodeRunResponse = {
  artifactRefs: Record<string, ArtifactRef>;
  key: NodeRunKey;
  metadata: Record<string, unknown>;
};

export type SubmitActionRequest = {
  actionKind: ActionKind;
  executionKind: 'action';
  operationId: OperationId;
  params: Record<string, unknown>;
  requestId: RequestId;
  sessionId: string;
  sourceArtifactKey?: ArtifactKey;
  sourceNodeRunKey?: NodeRunKey;
  type: 'execution.submit';
};

export type SubmitActionResponse<JsonResult = Record<string, unknown>> = {
  artifactRefs: Record<string, ArtifactRef>;
  jsonResult: JsonResult;
  key: ActionKey;
  metadata: Record<string, unknown>;
};

export function submitNodeRun(request: SubmitNodeRunRequest) {
  return postJson<SubmitNodeRunResponse>('/api/executions/submit', request);
}

export function submitAction<JsonResult = Record<string, unknown>>(request: SubmitActionRequest) {
  return postJson<SubmitActionResponse<JsonResult>>('/api/executions/submit', request);
}
