import { Client } from '@gradio/client';
import type { ExecutionProgress } from '../types';

export type ApiResult<T> = { ok: true; value: T } | { message: string; ok: false };

let clientPromise: Promise<Client> | null = null;

export function gradioClient(): Promise<Client> {
  if (clientPromise === null) {
    clientPromise = Client.connect(window.location.origin, {
      events: ['data', 'status'],
    });
  }

  return clientPromise;
}

export async function submitApi<ResponseBody>(
  endpoint: string,
  payload: Record<string, unknown>,
  onProgress?: (progress: ExecutionProgress) => void,
): Promise<ApiResult<ResponseBody>> {
  const client = await gradioClient();
  const job = client.submit(endpoint, payload);

  for await (const message of job) {
    if (message.type === 'status') {
      if (message.stage === 'pending') {
        onProgress?.({ progress: 0, stage: 'queued' });
      }

      const progressUnit = message.progress_data?.at(-1);
      if (progressUnit) {
        onProgress?.({
          progress: progressUnit.progress ?? 0,
          stage: progressUnit.desc ?? message.stage,
        });
      }

      if (message.stage === 'error') {
        return {
          message: String(message.message ?? 'Gradio request failed'),
          ok: false,
        };
      }
    }

    if (message.type === 'data') {
      return {
        ok: true,
        value: message.data[0] as ResponseBody,
      };
    }
  }

  return {
    message: 'Gradio request completed without data',
    ok: false,
  };
}
