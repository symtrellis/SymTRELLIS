import { Client } from '@gradio/client';
import type { ExecutionProgress, TaskId } from '../types';

export type ApiError = {
  kind: 'backend_error' | 'session_expired' | 'transport_error';
  message: string;
  ok: false;
};

export type ApiResult<T> =
  | { ok: true; value: T }
  | ApiError;

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
  queueTaskId?: TaskId,
): Promise<ApiResult<ResponseBody>> {
  let heartbeatId: number | null = null;
  let queueFailure: ApiError | null = null;

  try {
    const client = await gradioClient();
    const job = client.submit(endpoint, payload);
    let progressStarted = false;

    if (queueTaskId) {
      const eventId = await job.wait_for_id();
      if (eventId === null) {
        await job.cancel();
        return {
          kind: 'backend_error',
          message: 'Gradio queue did not provide an event id',
          ok: false,
        };
      }

      const marked = await submitApi<{ ok: boolean }>('/mark_execution_queued', {
        event_id: eventId,
        task_id: queueTaskId,
      });
      if (!marked.ok) {
        await job.cancel();
        return marked;
      }

      heartbeatId = window.setInterval(() => {
        void submitApi<{ queued: boolean }>('/renew_execution_queue', {
          event_id: eventId,
          task_id: queueTaskId,
        }).then((renewed) => {
          if (renewed.ok && !renewed.value.queued && heartbeatId !== null) {
            window.clearInterval(heartbeatId);
            heartbeatId = null;
          } else if (!renewed.ok && queueFailure === null) {
            queueFailure = renewed;
            void job.cancel().then(
              () => undefined,
              () => {
                clientPromise = null;
              },
            );
          }
        });
      }, 60_000);
    }

    for await (const message of job) {
      if (message.type === 'status') {
        if (heartbeatId !== null && message.stage !== 'pending') {
          window.clearInterval(heartbeatId);
          heartbeatId = null;
        }

        const progressUnit = message.progress_data?.at(-1);
        if (progressUnit) {
          progressStarted = true;
          onProgress?.({
            progress: progressUnit.progress ?? 0,
            stage: progressUnit.desc ?? message.stage,
          });
        } else if (message.stage === 'pending' && !progressStarted) {
          onProgress?.({ progress: 0, stage: 'queued' });
        }

        if (message.stage === 'error') {
          if (queueFailure !== null) {
            return queueFailure;
          }
          return {
            kind: 'backend_error',
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

    if (queueFailure !== null) {
      return queueFailure;
    }
    return {
      kind: 'backend_error',
      message: 'Gradio request completed without data',
      ok: false,
    };
  } catch (error) {
    clientPromise = null;
    return {
      kind: 'transport_error',
      message: String(error),
      ok: false,
    };
  } finally {
    if (heartbeatId !== null) {
      window.clearInterval(heartbeatId);
    }
  }
}
