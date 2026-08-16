import type { Attachment, Preset, RunSnapshot } from './types'

export type StartRequest = {
  prompt?: string
  preset?: string
  entities?: string[]
  constraints?: string[]
  /** Ids from `uploadAttachment`, not the files themselves. */
  attachments?: string[]
  mode: 'scripted' | 'live'
  /** null = let GOD choose, including choosing none. */
  domains: number | null
  execution: 'inprocess' | 'sandbox' | 'godbox'
}

async function json<T>(response: Response): Promise<T> {
  const body = await response.text()
  let parsed: unknown = null
  try {
    parsed = body ? JSON.parse(body) : null
  } catch {
    throw new Error(`${response.status}: ${body.slice(0, 200)}`)
  }
  if (!response.ok) {
    const message = (parsed as { error?: string })?.error ?? `request failed (${response.status})`
    throw new Error(message)
  }
  return parsed as T
}

export type PresetsResponse = {
  presets: Preset[]
  liveAvailable: boolean
  /** Why live is disabled, in the server's own words, for the tooltip. */
  liveReason: string | null
}

export async function fetchPresets(): Promise<PresetsResponse> {
  const response = await fetch('/api/presets')
  const body = await json<{
    presets: Preset[]
    live_available: boolean
    live_unavailable_reason: string | null
  }>(response)
  return {
    presets: body.presets,
    liveAvailable: body.live_available,
    liveReason: body.live_unavailable_reason,
  }
}

/** Upload one table and get back its id, profile, and derived sealed terms.
 *
 * The file goes up as the raw request body with its name in the query string.
 * The server has no web framework and therefore no multipart parser -- see the
 * route comment in `resolution/app.py`. `fetch` sends a `File` as a body
 * directly, so nothing has to be encoded on this side either.
 */
export async function uploadAttachment(file: File): Promise<Attachment> {
  const response = await fetch(`/api/uploads?name=${encodeURIComponent(file.name)}`, {
    method: 'POST',
    body: file,
  })
  const body = await json<{ attachment: Attachment }>(response)
  return body.attachment
}

export async function startRun(request: StartRequest): Promise<{ run_id: string; run: RunSnapshot }> {
  const response = await fetch('/api/runs', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request),
  })
  return json<{ run_id: string; run: RunSnapshot }>(response)
}

export async function fetchRun(runId: string): Promise<RunSnapshot> {
  const response = await fetch(`/api/runs/${runId}`)
  const body = await json<{ run: RunSnapshot }>(response)
  return body.run
}

export async function fetchRuns(): Promise<RunSnapshot[]> {
  const response = await fetch('/api/runs')
  const body = await json<{ runs: RunSnapshot[] }>(response)
  return body.runs
}

export async function cancelRun(runId: string): Promise<void> {
  await fetch(`/api/runs/${runId}/cancel`, { method: 'POST' })
}
