import type {
  ActionKey,
  NodeRunKey,
  OutputRef,
  OutputRole,
  UploadKey,
  UploadRef,
} from '../types';
import { postForm, type ApiResult } from './client';

export type BackendOutputRef = {
  filename: string;
  metadata: Record<string, unknown>;
};

type BackendUploadResponse = {
  content_hash: string;
  filename: string;
  mime_type: string;
  upload_key: string;
};

export async function uploadInputImage(
  file: Blob | File,
  filename: string,
): Promise<ApiResult<UploadRef>> {
  const formData = new FormData();
  formData.append('file', file, filename);

  const result = await postForm<BackendUploadResponse>('/uploads', formData);
  if (!result.ok) {
    return result;
  }

  return {
    ok: true,
    value: {
      contentHash: result.value.content_hash,
      filename: result.value.filename,
      mimeType: result.value.mime_type,
      uploadKey: result.value.upload_key as UploadKey,
    },
  };
}

export function nodeRunOutputUrl(nodeRunKey: NodeRunKey, role: OutputRole) {
  return `/node-runs/${encodeURIComponent(nodeRunKey)}/outputs/${encodeURIComponent(role)}`;
}

export function actionOutputUrl(actionKey: ActionKey, role: OutputRole) {
  return `/actions/${encodeURIComponent(actionKey)}/outputs/${encodeURIComponent(role)}`;
}

export function actionBundleUrl(actionKey: ActionKey) {
  return `/actions/${encodeURIComponent(actionKey)}/bundle`;
}

export function nodeRunOutputsWithUrls(
  nodeRunKey: NodeRunKey,
  outputs: Record<OutputRole, BackendOutputRef>,
) {
  return Object.fromEntries(
    Object.entries(outputs).map(([role, output]) => [
      role,
      {
        filename: output.filename,
        metadata: output.metadata,
        role,
        url: nodeRunOutputUrl(nodeRunKey, role),
      },
    ]),
  ) as Record<OutputRole, OutputRef>;
}

export function actionOutputsWithUrls(
  actionKey: ActionKey,
  outputs: Record<OutputRole, BackendOutputRef>,
) {
  return Object.fromEntries(
    Object.entries(outputs).map(([role, output]) => [
      role,
      {
        filename: output.filename,
        metadata: output.metadata,
        role,
        url: actionOutputUrl(actionKey, role),
      },
    ]),
  ) as Record<OutputRole, OutputRef>;
}

export function outputByRole(outputs: Record<OutputRole, OutputRef>, roles: OutputRole[]) {
  return roles.reduce<OutputRef | null>(
    (output, role) => output ?? outputs[role] ?? null,
    null,
  );
}
