import { useEffect, useRef, useState } from 'react'
import { GOD } from '../lib/model'
import type { RunState, Solution } from '../lib/types'
import { Reflecting } from './Reflecting'
import { StreamRow } from './StreamItems'

// Long enough not to flicker between two events of a chatty phase, short
// enough to appear well before anyone starts wondering.
const QUIET_AFTER_S = 4

type Props = {
  state: RunState
  version: number
  onOpenNode: (nodeId: string) => void
}

/**
 * The run as it happens.
 *
 * Two lenses over the same timeline. `god` is the orchestration narrative on
 * its own; `all` interleaves every demigod's tool calls and findings at the
 * moment they occurred — which is the only view where you can see three sealed
 * agents working in parallel while GOD waits.
 */
export function StreamPane({ state, version, onOpenNode }: Props) {
  const [lens, setLens] = useState<'god' | 'all'>('all')
  const bodyRef = useRef<HTMLDivElement>(null)
  const pinnedRef = useRef(true)

  // Follow the stream, but stop following the moment the reader scrolls up --
  // yanking someone back to the bottom while they are reading a tool payload is
  // the fastest way to make a live view unusable.
  useEffect(() => {
    const body = bodyRef.current
    if (!body || !pinnedRef.current) return
    body.scrollTop = body.scrollHeight
  }, [version, lens])

  function onScroll() {
    const body = bodyRef.current
    if (!body) return
    pinnedRef.current = body.scrollHeight - body.scrollTop - body.clientHeight < 140
  }

  const entries =
    lens === 'god' ? state.timeline.filter((entry) => entry.node === GOD) : state.timeline

  // How long the stream has been silent. A GOD sandbox does not stream tokens,
  // so planning is a genuine 60-second gap between two phase lines -- and a
  // view that renders nothing at all during it is indistinguishable from a
  // hang. Past the threshold we say so, in the wordmark's own voice; the exact
  // silence is on the row's tooltip for anyone who wants the number.
  const last = entries[entries.length - 1]
  const quietFor = last ? state.elapsed - last.item.t : state.elapsed
  const waiting = state.status === 'running' && quietFor > QUIET_AFTER_S

  return (
    <section className="column stream-col">
      <header className="column-head">
        <span className="mono">Stream</span>
        <div className="seg" role="group" aria-label="Stream lens">
          <button type="button" aria-pressed={lens === 'all'} onClick={() => setLens('all')}>
            All lanes
          </button>
          <button type="button" aria-pressed={lens === 'god'} onClick={() => setLens('god')}>
            God only
          </button>
        </div>
        <span className="topbar-spacer" />
        <span className="mono muted">
          {state.eventCount} events · {state.toolCalls} tool calls
        </span>
        {state.status === 'running' && <span className="chip live">streaming</span>}
      </header>
      <div className="column-body" ref={bodyRef} onScroll={onScroll}>
        <div className="stream">
          {entries.map(({ node, item }) => (
            <StreamRow
              key={`${node}-${item.id}`}
              item={item}
              lane={node === GOD ? null : state.nodes[node]?.name ?? node}
              onLaneClick={() => onOpenNode(node)}
              onOpenNode={onOpenNode}
            />
          ))}
        </div>
        {waiting && (
          <div
            className="waiting"
            title={`still working · ${quietFor.toFixed(0)}s since the last event`}
          >
            <span className="caret" />
            <Reflecting />
          </div>
        )}
        {state.solution && <Answer solution={state.solution} />}
        {state.error && <p className="notice error">{state.error}</p>}
        {entries.length === 0 && <p className="empty">waiting for the first event…</p>}
      </div>
    </section>
  )
}

function Answer({ solution }: { solution: Solution }) {
  const contributions = Object.entries(solution.domain_contributions)
  return (
    <div className="answer">
      <h2>Integrated answer</h2>
      <div className="badge-line">
        <span className="chip">{Math.round(solution.confidence * 100)}% confidence</span>
        <span className="chip">{contributions.length} domains</span>
        {solution.conflicts.length > 0 && (
          <span className="chip">{solution.conflicts.length} conflicts</span>
        )}
      </div>
      <p>{solution.answer}</p>
      {contributions.length > 0 && (
        <div className="contribs">
          {contributions.map(([domain, text]) => (
            <div className="contrib" key={domain}>
              <span className="k">{domain}</span>
              <span>{text}</span>
            </div>
          ))}
        </div>
      )}
      {solution.gaps.length > 0 && (
        <div className="gaps">
          <strong>Gaps · </strong>
          {solution.gaps.join(' · ')}
        </div>
      )}
    </div>
  )
}
