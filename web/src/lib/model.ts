import type { RunNode, RunState, StreamItem, UiEvent } from './types'

/**
 * Events in, a renderable run tree out.
 *
 * The server sends a flat, append-only log. Everything hierarchical about the
 * UI -- the root, its children, which tool result belongs to which call -- is
 * derived here and nowhere else, so a reconnecting tab that replays the same
 * log lands in exactly the same state.
 *
 * The state is mutated in place, deliberately. A run emits a text delta every
 * few milliseconds, and cloning the tree per delta made the stream stutter well
 * before the model finished talking. The single writer is `applyEvent`, and the
 * component tree re-renders off a version counter (see `useRun`).
 */

export const GOD = 'god'

export function emptyRun(
  mode: 'scripted' | 'live',
  execution: RunState['execution'] = 'inprocess',
): RunState {
  return {
    timeline: [],
    runId: null,
    mode,
    execution,
    status: 'idle',
    question: '',
    nodes: {},
    order: [],
    solution: null,
    error: null,
    elapsed: 0,
    eventCount: 0,
    toolCalls: 0,
  }
}

export function startRun(
  runId: string,
  mode: 'scripted' | 'live',
  execution: RunState['execution'],
  question: string,
): RunState {
  const state = emptyRun(mode, execution)
  state.runId = runId
  state.status = 'running'
  state.question = question
  ensureNode(state, GOD, 0)
  return state
}

function ensureNode(state: RunState, id: string, t: number): RunNode {
  let node = state.nodes[id]
  if (!node) {
    node = {
      id,
      name: id === GOD ? 'GOD' : id.replace(/^demi:/, ''),
      kind: id === GOD ? 'god' : 'demigod',
      status: id === GOD ? 'running' : 'planned',
      tools: [],
      toolCalls: 0,
      items: [],
      firstSeen: t,
      lastSeen: t,
    }
    state.nodes[id] = node
    state.order.push(id)
  }
  return node
}

export function demigodNodes(state: RunState): RunNode[] {
  return state.order.filter((id) => id !== GOD).map((id) => state.nodes[id])
}

const PHASE_LABELS: Record<string, string> = {
  analyzing: 'Reading the problem',
  planning: 'Inventing representations',
  projecting: 'Projecting into each domain',
  spawning: 'Spawning demigods',
  collecting: 'Collecting artifacts',
  integrating: 'Mapping back to the native problem',
  done: 'Analysis complete',
}

export const TEXT_PHASE_LABELS: Record<string, string> = {
  invent: 'Inventing domains',
  critic: 'Orthogonality critic',
  integrate: 'Integration',
}

export function textPhaseLabel(phase: string): string {
  if (PHASE_LABELS[phase]) return PHASE_LABELS[phase]
  if (TEXT_PHASE_LABELS[phase]) return TEXT_PHASE_LABELS[phase]
  if (phase.startsWith('transform:')) return `Projection · ${phase.slice('transform:'.length)}`
  if (phase.startsWith('demigod:')) return 'Sealed reasoning'
  return phase || 'Model output'
}

/**
 * One item, appended twice: to its own node and to the merged timeline.
 *
 * The two views need different orders. A transcript is one node's history; the
 * main stream is everything in arrival order, which is the only way a tool call
 * inside a demigod shows up next to what GOD was doing at the time.
 */
function push(state: RunState, node: RunNode, item: StreamItem): StreamItem {
  node.items.push(item)
  state.timeline.push({ node: node.id, item })
  return item
}

function record(node: RunNode, ev: UiEvent): string {
  node.lastSeen = ev.t
  return `${node.id}-${ev.seq}`
}

function dict(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []
}

/** `confidence=0.86; the pool absorbs the extra inflow` -> both halves. */
function splitArtifact(message: string): { confidence?: number; text: string } {
  const match = /^confidence=([\d.]+);\s*/.exec(message)
  if (!match) return { text: message }
  return { confidence: Number(match[1]), text: message.slice(match[0].length) }
}

export function applyEvent(state: RunState, ev: UiEvent): void {
  state.eventCount += 1
  state.elapsed = Math.max(state.elapsed, ev.t)
  const node = ensureNode(state, ev.node, ev.t)
  const id = record(node, ev)
  const data = dict(ev.data)

  switch (ev.kind) {
    case 'text_open': {
      const phase = str(data?.phase) ?? ''
      push(state, node, {
        type: 'text',
        id,
        t: ev.t,
        phase,
        text: '',
        open: true,
        simulated: data?.simulated === true,
      })
      return
    }
    case 'text': {
      const block = lastOpenText(node)
      if (block) block.text += ev.message
      return
    }
    case 'text_close': {
      const block = lastOpenText(node)
      if (block) block.open = false
      return
    }
    case 'tool_call': {
      node.toolCalls += 1
      state.toolCalls += 1
      push(state, node, {
        type: 'tool',
        id,
        t: ev.t,
        tool: ev.message,
        state: 'call',
        args: dict(data?.arguments) ?? undefined,
      })
      return
    }
    case 'tool_result': {
      const pending = lastPendingTool(node, ev.message)
      if (pending) {
        pending.state = 'result'
        pending.result = data ? data.result : ev.data
      }
      return
    }
    case 'tool_deny':
    case 'tool_error': {
      const [tool, ...rest] = ev.message.split(': ')
      const pending = lastPendingTool(node, tool)
      const failure = ev.kind === 'tool_deny' ? 'deny' : 'error'
      if (pending) {
        pending.state = failure
        pending.error = rest.join(': ')
      } else {
        node.toolCalls += 1
        state.toolCalls += 1
        push(state, node, { type: 'tool', id, t: ev.t, tool, state: failure, error: rest.join(': ') })
      }
      return
    }
  }

  if (node.kind === 'god') applyGod(state, node, ev, id, data)
  else applyDemigod(state, node, ev, id, data)
}

function applyGod(
  state: RunState,
  node: RunNode,
  ev: UiEvent,
  id: string,
  data: Record<string, unknown> | null,
): void {
  switch (ev.kind) {
    case 'start':
      push(state, node, {
        type: 'phase',
        id,
        t: ev.t,
        phase: 'analyzing',
        label: PHASE_LABELS.analyzing,
        detail: ev.message.replace(/^problem=[^;]+;\s*/, ''),
      })
      return
    case 'runtime':
      // Where the demigods will run, stated once at the top of the stream. It
      // is the difference between a run that proves the orchestration and one
      // that proves the whole sandbox, image and lease path.
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'ok' })
      return
    case 'catalog':
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'muted' })
      return
    case 'plan': {
      if (Array.isArray(ev.data)) {
        for (const raw of ev.data as unknown[]) {
          const entry = dict(raw)
          const name = str(entry?.name)
          if (!name) continue
          const child = ensureNode(state, `demi:${name}`, ev.t)
          child.axis = str(entry?.axis)
          child.tools = strings(entry?.tools)
          push(state, node, {
            type: 'domain',
            id: `${id}-${name}`,
            t: ev.t,
            name,
            axis: child.axis,
            tools: child.tools,
            stage: 'planned',
          })
        }
        return
      }
      push(state, node, {
        type: 'phase',
        id,
        t: ev.t,
        phase: 'planning',
        label: PHASE_LABELS.planning,
        detail: ev.message,
      })
      return
    }
    case 'transform':
      return
    case 'sealed': {
      const name = str(data?.domain_name) ?? ev.message.replace(/ ready$/, '')
      const child = ensureNode(state, `demi:${name}`, ev.t)
      child.status = 'sealed'
      child.axis = str(data?.axis) ?? child.axis
      child.language = str(data?.language) ?? child.language
      const tools = strings(data?.tools)
      if (tools.length) child.tools = tools
      push(state, node, {
        type: 'domain',
        id,
        t: ev.t,
        name,
        axis: child.axis,
        language: child.language,
        tools: child.tools,
        stage: 'sealed',
      })
      return
    }
    case 'spawn':
      push(state, node, {
        type: 'phase',
        id,
        t: ev.t,
        phase: 'spawning',
        label: PHASE_LABELS.spawning,
        detail: ev.message,
      })
      return
    case 'collect': {
      const name = str(data?.domain_name) ?? ''
      const child = state.nodes[`demi:${name}`]
      const confidence = num(data?.confidence)
      const conclusion = str(data?.conclusion)
      if (child) {
        child.status = 'done'
        child.confidence = confidence
        child.conclusion = conclusion
        // TWO SOURCES, and only one exists per execution mode. In-process, the
        // registry emits a `tool_call` event per call and the counter above is
        // the live one. In a sandbox the calls happen inside the demigod's own
        // container and are recorded by the BROKER, so the only count that ever
        // reaches here is this one, at the end. Taking the max keeps whichever
        // exists without a mode check -- and without it a sandboxed run showed
        // "0 CALLS" forever, which reads as "it used no tools" rather than
        // "this view cannot see them".
        const reported = num(data?.tool_calls)
        if (reported !== undefined) {
          child.toolCalls = Math.max(child.toolCalls, reported)
        }
      }
      push(state, node, {
        type: 'collect',
        id,
        t: ev.t,
        name,
        confidence,
        conclusion,
        toolCalls: num(data?.tool_calls),
        ok: true,
      })
      return
    }
    case 'repair': {
      // GOD rewriting a planner leak instead of discarding the domain. Worth a
      // visible line rather than the muted default: it is an intervention that
      // changes what the demigod is told, and the alternative outcome -- a lost
      // domain and a gap in the final answer -- is one the reader should see
      // was avoided.
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'warn' })
      return
    }
    case 'reject':
    case 'failure': {
      const child = resolveDomain(state, ev.message)
      if (child) {
        child.status = 'failed'
        child.error = ev.message
      }
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'warn' })
      return
    }
    case 'integrate':
      push(state, node, {
        type: 'phase',
        id,
        t: ev.t,
        phase: 'integrating',
        label: PHASE_LABELS.integrating,
        detail: ev.message,
      })
      return
    case 'done':
      push(state, node, {
        type: 'phase',
        id,
        t: ev.t,
        phase: 'done',
        label: PHASE_LABELS.done,
        detail: ev.message,
      })
      return
    case 'run_end': {
      node.status = ev.message === 'done' ? 'done' : 'failed'
      state.status = (ev.message as RunState['status']) ?? 'done'
      const solution = dict(data?.solution)
      if (solution) state.solution = solution as unknown as RunState['solution']
      state.error = str(data?.error) ?? null
      return
    }
    default:
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'muted' })
  }
}

function applyDemigod(
  state: RunState,
  node: RunNode,
  ev: UiEvent,
  id: string,
  data: Record<string, unknown> | null,
): void {
  switch (ev.kind) {
    case 'start':
      node.status = 'running'
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'ok' })
      return
    case 'scope': {
      // "axis=topology; language=directed acyclic graph"
      const axis = /axis=([^;]+)/.exec(ev.message)?.[1]?.trim() ?? node.axis ?? ''
      const language = /language=(.+)$/.exec(ev.message)?.[1]?.trim() ?? node.language ?? ''
      node.axis = axis || node.axis
      node.language = language || node.language
      const tools = strings(data?.tools)
      if (tools.length) node.tools = tools
      push(state, node, {
        type: 'scope',
        id,
        t: ev.t,
        axis,
        language,
        tools: node.tools,
        maxSteps: num(data?.max_steps),
        maxToolCalls: num(data?.max_tool_calls),
      })
      return
    }
    case 'model':
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'muted' })
      return
    case 'reason':
      push(state, node, { type: 'reason', id, t: ev.t, text: ev.message })
      return
    case 'artifact': {
      const { confidence, text } = splitArtifact(ev.message)
      if (confidence !== undefined) node.confidence = confidence
      node.conclusion = text
      push(state, node, { type: 'artifact', id, t: ev.t, text, confidence })
      return
    }
    case 'done':
      node.status = 'done'
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'ok' })
      return
    case 'fail':
      node.status = 'failed'
      node.error = ev.message
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'warn' })
      return
    default:
      push(state, node, { type: 'note', id, t: ev.t, text: ev.message, tone: 'muted' })
  }
}

/**
 * Which demigod is this rejection about?
 *
 * GOD names the domain first in both cases but punctuates them differently:
 * `FAILURE` reads "catalytic_dag: the model refused", while `REJECT` reads
 * "catalytic_dag transform failed; continuing without it". Splitting on the
 * colon alone therefore matches one and silently misses the other, and a
 * missed match is not a cosmetic loss -- the node keeps its planned status and
 * the tree shows a domain as still queued long after GOD gave up on it.
 *
 * Matching against known node ids rather than parsing the sentence, so a
 * future rewording cannot quietly break it again.
 */
function resolveDomain(state: RunState, message: string): RunNode | null {
  const head = message.split(':')[0].trim()
  const direct = state.nodes[`demi:${head}`]
  if (direct) return direct
  const firstWord = message.trim().split(/\s+/)[0]
  return state.nodes[`demi:${firstWord}`] ?? null
}

function lastOpenText(node: RunNode): (StreamItem & { type: 'text' }) | null {
  for (let i = node.items.length - 1; i >= 0; i--) {
    const item = node.items[i]
    if (item.type === 'text' && item.open) return item
  }
  return null
}

function lastPendingTool(node: RunNode, tool: string): (StreamItem & { type: 'tool' }) | null {
  for (let i = node.items.length - 1; i >= 0; i--) {
    const item = node.items[i]
    if (item.type === 'tool' && item.tool === tool && item.state === 'call') return item
  }
  return null
}
