// Share in-flight work and retain successful results for a bounded period.
export function createRequestCache<T>(ttlMs: number, now = Date.now) {
  const results = new Map<string, { value: T; expires: number }>();
  const pending = new Map<string, Promise<T>>();
  const peek = (key: string): T | undefined => {
    const item = results.get(key);
    if (item && item.expires > now()) return item.value;
    results.delete(key);
    return undefined;
  };
  return {
    peek,
    get(key: string, load: () => Promise<T>, refresh = false): Promise<T> {
      const running = pending.get(key);
      if (running) return running;
      const cached = peek(key);
      if (!refresh && cached !== undefined) return Promise.resolve(cached);
      const request = Promise.resolve().then(load).then(value => {
        results.set(key, { value, expires: now() + ttlMs });
        return value;
      }).finally(() => pending.delete(key));
      pending.set(key, request);
      return request;
    },
  };
}

export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
