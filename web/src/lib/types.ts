export type UiEvent = {
  seq: number
  t: number
  node: string
  lane: string
  kind: string
  group: string
  message: string
  data: unknown
  phase: string | null
}

export type RunSnapshot = {
  run_id: string
  status: 'running' | 'done' | 'error' | 'cancelled'
  question: string
  mode: 'scripted' | 'live'
  execution: 'inprocess' | 'sandbox' | 'godbox'
  started_at: number
  finished_at: number | null
  error: string | null
  solution: Solution | null
  domains: { name: string; axes: string[]; language: string; tools: string[] }[]
}

export type Solution = {
  problem_id: string
  answer: string
  confidence: number
  domain_contributions: Record<string, string>
  conflicts: string[]
  gaps: string[]
}

export type Preset = {
  id: string
  label: string
  blurb: string
  /** How much a second opinion is worth on it — the axis the ✦ option responds to. */
  tier: 'easy' | 'medium' | 'hard'
  modes: string[]
  prompt: string
  entities: string[]
  constraints: string[]
}

export type ColumnProfile = {
  name: string
  type: 'number' | 'text' | 'boolean'
  nulls?: number
  unique?: number
  /** Categories joined by ` | `; one string so the projection stays small. */
  values?: string
  min?: number
  max?: number
  mean?: number
}

export type TableProfile = {
  format: string
  rows: number
  columns: ColumnProfile[]
  stats_from_first_rows?: number
  columns_omitted?: number
}

export type Attachment = {
  id: string
  name: string
  size: number
  profile: TableProfile
  /** Schema vocabulary sealed on the user's behalf, replacing the old field. */
  terms: string[]
}

export type NodeStatus = 'planned' | 'sealed' | 'running' | 'done' | 'failed'

export type RunNode = {
  id: string
  name: string
  kind: 'god' | 'demigod'
  status: NodeStatus
  axis?: string
  language?: string
  tools: string[]
  toolCalls: number
  confidence?: number
  conclusion?: string
  error?: string
  items: StreamItem[]
  firstSeen: number
  lastSeen: number
}

export type ToolState = 'call' | 'result' | 'deny' | 'error'

export type StreamItem =
  | { type: 'phase'; id: string; t: number; label: string; detail: string; phase: string }
  | { type: 'note'; id: string; t: number; text: string; tone: 'muted' | 'warn' | 'ok' }
  | {
      type: 'text'
      id: string
      t: number
      phase: string
      text: string
      open: boolean
      simulated: boolean
    }
  | {
      type: 'tool'
      id: string
      t: number
      tool: string
      state: ToolState
      args?: Record<string, unknown>
      result?: unknown
      error?: string
    }
  | {
      type: 'domain'
      id: string
      t: number
      name: string
      axis?: string
      language?: string
      tools: string[]
      stage: 'planned' | 'sealed' | 'rejected'
    }
  | {
      type: 'collect'
      id: string
      t: number
      name: string
      confidence?: number
      conclusion?: string
      toolCalls?: number
      ok: boolean
    }
  | {
      type: 'scope'
      id: string
      t: number
      axis: string
      language: string
      tools: string[]
      maxSteps?: number
      maxToolCalls?: number
    }
  | { type: 'artifact'; id: string; t: number; text: string; confidence?: number }
  | { type: 'reason'; id: string; t: number; text: string }

/** Every item in the run, in arrival order, tagged with the node that owns it. */
export type TimelineEntry = { node: string; item: StreamItem }

export type RunState = {
  timeline: TimelineEntry[]
  runId: string | null
  mode: 'scripted' | 'live'
  execution: 'inprocess' | 'sandbox' | 'godbox'
  status: 'idle' | 'running' | 'done' | 'error' | 'cancelled'
  question: string
  nodes: Record<string, RunNode>
  order: string[]
  solution: Solution | null
  error: string | null
  elapsed: number
  eventCount: number
  toolCalls: number
}
