/** Thin fetch wrapper over the research API. */

export const API_BASE: string =
  (import.meta.env?.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }

  /** The artifacts have not been exported yet. */
  get isNotExported(): boolean {
    return this.status === 503
  }

  /** Nothing to show yet (no results, or no such campaign). */
  get isMissing(): boolean {
    return this.status === 404
  }
}

/** How long a request may hang before it counts as unreachable.
 *
 * A backend that accepts the connection and then never answers - a wedged
 * worker left over from an earlier session, for instance - leaves `fetch`
 * pending forever, so the status chip sits on "Checking API" and no page ever
 * gets an error to show. A deadline turns that silence into the same honest
 * failure a refused connection already produces. */
export const REQUEST_TIMEOUT_MS = 20_000

function timeoutSignal(ms: number): AbortSignal | undefined {
  return typeof AbortSignal !== 'undefined' &&
    typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(ms)
    : undefined
}

export function buildQuery(params: Record<string, unknown> | undefined): string {
  if (!params) return ''
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ''
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, unknown>,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}${buildQuery(params)}`, {
      signal: timeoutSignal(REQUEST_TIMEOUT_MS),
    })
  } catch {
    throw new ApiError(
      0,
      `Could not reach the API at ${API_BASE}. Is the backend running?`,
    )
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* keep the default message */
    }
    throw new ApiError(response.status, detail)
  }

  return (await response.json()) as T
}
