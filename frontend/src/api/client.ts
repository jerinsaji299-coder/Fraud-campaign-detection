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
    response = await fetch(`${API_BASE}${path}${buildQuery(params)}`)
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
