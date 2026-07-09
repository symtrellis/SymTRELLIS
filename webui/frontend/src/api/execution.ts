import type {
  ActionKey,
  ActionRecord,
  ActionResult,
  NodeRunKey,
  NodeRunResult,
  OutputRole,
  RequestId,
  RestoredSessionRef,
  SessionId,
  UploadKey,
} from '../types';
import type { ModelId, OperationId } from '../models/types';
import { postJson, type ApiResult } from './client';
import {
  actionOutputsWithUrls,
  nodeRunOutputsWithUrls,
  type BackendOutputRef,
} from './storage';

export type SubmitNodeRunRequest = {
  inputUploadKeys: UploadKey[];
  modelId: ModelId;
  operationId: OperationId;
  parentRunKeys: NodeRunKey[];
  params: Record<string, unknown>;
  requestId: RequestId;
  sessionId: SessionId | null;
};

export type SubmitActionRequest = {
  operationId: OperationId;
  params: Record<string, unknown>;
  requestId: RequestId;
  sessionId: SessionId;
  sourceNodeRunKey: NodeRunKey;
};

type BackendExecutionResponse = {
  cached: boolean;
  json_result: unknown;
  key: string;
  metadata: Record<string, unknown>;
  outputs: Record<OutputRole, BackendOutputRef>;
  session_id: string;
};

type BackendNodeRunRecord = {
  ancestor_run_keys: string[];
  input_upload_keys: string[];
  json_result: unknown;
  metadata: Record<string, unknown>;
  model_id: string;
  node_run_key: string;
  operation_id: string;
  operation_version: string;
  outputs: Record<OutputRole, BackendOutputRef>;
  params: Record<string, unknown>;
  parent_run_keys: string[];
};

type BackendActionRecord = {
  action_key: string;
  json_result: unknown;
  metadata: Record<string, unknown>;
  operation_id: string;
  operation_version: string;
  outputs: Record<OutputRole, BackendOutputRef>;
  params: Record<string, unknown>;
  source_node_run_key: string;
};

type BackendRestoreSessionResponse = {
  actions: Record<string, BackendActionRecord[]>;
  node_runs: BackendNodeRunRecord[];
  session: {
    active_run_keys: string[];
    model_id: string;
    session_id: string;
  };
};

export async function submitNodeRun(
  request: SubmitNodeRunRequest,
): Promise<ApiResult<NodeRunResult>> {
  const result = await postJson<BackendExecutionResponse>('/node-runs', {
    input_upload_keys: request.inputUploadKeys,
    model_id: request.modelId,
    operation_id: request.operationId,
    parent_run_keys: request.parentRunKeys,
    params: request.params,
    request_id: request.requestId,
    session_id: request.sessionId,
  });

  if (!result.ok) {
    return result;
  }

  const key = result.value.key as NodeRunKey;
  return {
    ok: true,
    value: {
      cached: result.value.cached,
      jsonResult: result.value.json_result,
      key,
      metadata: result.value.metadata,
      outputs: nodeRunOutputsWithUrls(key, result.value.outputs),
      sessionId: result.value.session_id as SessionId,
    },
  };
}

export async function restoreSession(
  sessionId: SessionId,
  key?: NodeRunKey,
): Promise<ApiResult<RestoredSessionRef>> {
  const params = key ? `?${new URLSearchParams({ key }).toString()}` : '';
  const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}${params}`);

  if (!response.ok) {
    return { message: `${response.status} ${response.statusText}`, ok: false };
  }

  const restored = (await response.json()) as BackendRestoreSessionResponse;
  return {
    ok: true,
    value: {
      actions: Object.fromEntries(
        Object.entries(restored.actions).map(([sourceKey, actions]) => [
          sourceKey as NodeRunKey,
          actions.map((action) => {
            const actionKey = action.action_key as ActionKey;
            return {
              jsonResult: action.json_result,
              key: actionKey,
              metadata: action.metadata,
              operationId: action.operation_id,
              operationVersion: action.operation_version,
              outputs: actionOutputsWithUrls(actionKey, action.outputs),
              params: action.params,
              sourceNodeRunKey: action.source_node_run_key as NodeRunKey,
            };
          }),
        ]),
      ) as Record<NodeRunKey, ActionRecord[]>,
      activeRunKeys: restored.session.active_run_keys as NodeRunKey[],
      modelId: restored.session.model_id,
      nodeRuns: restored.node_runs.map((run) => {
        const nodeRunKey = run.node_run_key as NodeRunKey;
        return {
          ancestorRunKeys: run.ancestor_run_keys as NodeRunKey[],
          inputUploadKeys: run.input_upload_keys as UploadKey[],
          jsonResult: run.json_result,
          key: nodeRunKey,
          metadata: run.metadata,
          modelId: run.model_id,
          operationId: run.operation_id,
          operationVersion: run.operation_version,
          outputs: nodeRunOutputsWithUrls(nodeRunKey, run.outputs),
          params: run.params,
          parentRunKeys: run.parent_run_keys as NodeRunKey[],
        };
      }),
      sessionId: restored.session.session_id as SessionId,
    },
  };
}

export async function submitAction<JsonResult = unknown>(
  request: SubmitActionRequest,
): Promise<ApiResult<ActionResult<JsonResult>>> {
  const result = await postJson<BackendExecutionResponse>('/actions', {
    operation_id: request.operationId,
    params: request.params,
    request_id: request.requestId,
    session_id: request.sessionId,
    source_node_run_key: request.sourceNodeRunKey,
  });

  if (!result.ok) {
    return result;
  }

  const key = result.value.key as ActionKey;
  return {
    ok: true,
    value: {
      cached: result.value.cached,
      jsonResult: result.value.json_result as JsonResult,
      key,
      metadata: result.value.metadata,
      outputs: actionOutputsWithUrls(key, result.value.outputs),
      sessionId: result.value.session_id as SessionId,
    },
  };
}
