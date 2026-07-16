import type {
  ActionKey,
  ActionRecord,
  ActionResult,
  ExecutionProgress,
  NodeRunKey,
  NodeRunResult,
  OutputRole,
  RestoredSessionRef,
  SessionId,
  SessionRevision,
  TaskId,
  UploadKey,
} from '../types';
import type { ModelId, OperationId } from '../models/types';
import { submitApi, type ApiResult } from './client';
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
  sessionId: SessionId | null;
  sessionRevision: SessionRevision;
};

export type SubmitActionRequest = {
  modelId: ModelId;
  operationId: OperationId;
  params: Record<string, unknown>;
  sessionId: SessionId;
  sessionRevision: SessionRevision;
  sourceNodeRunKey: NodeRunKey;
};

type BackendExecutionResponse = {
  cached: boolean;
  json_result: unknown;
  key: string;
  metadata: Record<string, unknown>;
  outputs: Record<OutputRole, BackendOutputRef>;
  session_id: string;
  session_revision: number;
};

type BackendPrepareResponse =
  | {
      result: BackendExecutionResponse;
      status: 'completed';
    }
  | {
      status: 'gpu_required';
      task_id: string;
    }
  | {
      status: 'session_expired';
    };

type BackendExecuteResponse =
  | BackendExecutionResponse
  | {
      status: 'session_expired';
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

type BackendRestoreSessionResponse =
  | {
      restored: {
        actions: Record<string, BackendActionRecord[]>;
        node_runs: BackendNodeRunRecord[];
        session: {
          active_run_keys: string[];
          model_id: string;
          revision: number;
          session_id: string;
        };
      };
      status: 'restored';
    }
  | {
      status: 'session_expired';
    }
  | {
      status: 'not_found';
    };

export type RestoreSessionResult =
  | {
      ok: true;
      value: RestoredSessionRef;
    }
  | {
      kind: 'backend_error' | 'not_found' | 'session_expired' | 'transport_error';
      message: string;
      ok: false;
    };

async function submitExecution(
  request: SubmitNodeRunRequest | SubmitActionRequest,
  executionKind: 'node_run' | 'action',
  onProgress?: (progress: ExecutionProgress) => void,
): Promise<ApiResult<BackendExecutionResponse>> {
  const nodeRunRequest = request as SubmitNodeRunRequest;
  const actionRequest = request as SubmitActionRequest;
  const payload = {
    execution_kind: executionKind,
    input_upload_keys: executionKind === 'node_run' ? nodeRunRequest.inputUploadKeys : [],
    model_id: request.modelId,
    operation_id: request.operationId,
    params: request.params,
    parent_run_keys: executionKind === 'node_run' ? nodeRunRequest.parentRunKeys : [],
    session_id: request.sessionId,
    session_revision: request.sessionRevision,
    source_node_run_key:
      executionKind === 'action' ? actionRequest.sourceNodeRunKey : null,
  };

  const prepare = await submitApi<BackendPrepareResponse>('/prepare_execution', {
    payload,
  });
  if (!prepare.ok) {
    return prepare;
  }

  if (prepare.value.status === 'session_expired') {
    return {
      kind: 'session_expired',
      message: 'Session expired',
      ok: false,
    };
  }

  if (prepare.value.status === 'completed') {
    return {
      ok: true,
      value: prepare.value.result,
    };
  }

  const taskId = prepare.value.task_id as TaskId;

  const execution = await submitApi<BackendExecuteResponse>(
    '/execute_execution',
    { task_id: taskId },
    onProgress,
    taskId,
  );
  if (!execution.ok) {
    return execution;
  }
  if ('status' in execution.value) {
    return {
      kind: 'session_expired',
      message: 'Session expired',
      ok: false,
    };
  }
  return {
    ok: true,
    value: execution.value,
  };
}

export async function submitNodeRun(
  request: SubmitNodeRunRequest,
  onProgress?: (progress: ExecutionProgress) => void,
): Promise<ApiResult<NodeRunResult>> {
  const result = await submitExecution(request, 'node_run', onProgress);
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
      sessionRevision: result.value.session_revision as SessionRevision,
    },
  };
}

export async function restoreSession(
  sessionId: SessionId,
  key?: NodeRunKey,
): Promise<RestoreSessionResult> {
  const result = await submitApi<BackendRestoreSessionResponse>('/restore_session', {
    key: key ?? null,
    session_id: sessionId,
  });
  if (!result.ok) {
    return result;
  }

  if (result.value.status === 'session_expired') {
    return {
      kind: 'session_expired',
      message: 'Session expired',
      ok: false,
    };
  }
  if (result.value.status === 'not_found') {
    return {
      kind: 'not_found',
      message: 'Session not found',
      ok: false,
    };
  }

  const restored = result.value.restored;
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
      sessionRevision: restored.session.revision as SessionRevision,
    },
  };
}

export async function submitAction<JsonResult = unknown>(
  request: SubmitActionRequest,
  onProgress?: (progress: ExecutionProgress) => void,
): Promise<ApiResult<ActionResult<JsonResult>>> {
  const result = await submitExecution(request, 'action', onProgress);
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
      sessionRevision: result.value.session_revision as SessionRevision,
    },
  };
}
