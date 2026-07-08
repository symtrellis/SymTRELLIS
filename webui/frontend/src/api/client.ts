export type ApiResult<T> = { ok: true; value: T } | { message: string; ok: false };

export async function postJson<ResponseBody>(
  path: string,
  body: unknown,
): Promise<ApiResult<ResponseBody>> {
  const response = await fetch(path, {
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });

  if (!response.ok) {
    return { message: `${response.status} ${response.statusText}`, ok: false };
  }

  return { ok: true, value: (await response.json()) as ResponseBody };
}

export async function postForm<ResponseBody>(
  path: string,
  formData: FormData,
): Promise<ApiResult<ResponseBody>> {
  const response = await fetch(path, {
    body: formData,
    method: 'POST',
  });

  if (!response.ok) {
    return { message: `${response.status} ${response.statusText}`, ok: false };
  }

  return { ok: true, value: (await response.json()) as ResponseBody };
}

// BACKEND_PROTOCOL_PENDING: streaming event names and payloads are finalized with backend execution wiring.
export function createJsonWebSocket(path: string) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${window.location.host}${path}`);

  return {
    sendJson(message: unknown) {
      socket.send(JSON.stringify(message));
    },
    socket,
  };
}
