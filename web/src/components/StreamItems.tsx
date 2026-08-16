import { useEffect, useRef, useState } from 'react'
import type { StreamItem } from '../lib/types'
import { textPhaseLabel } from '../lib/model'

export function fmtTime(t: number): string {
  return `${t.toFixed(1)}s`
}

function describe(value: unknown): string {
  if (value === null || value === undefined) return 'no payload'
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`
  if (typeof value === 'object') {
    const keys = Object.keys(value as object)
    return keys.length ? `fields: ${keys.join(', ')}` : 'empty object'
  }
  return String(value)
}

/** Model output, arriving live. Long blocks collapse so the stream stays readable. */
function TextBlock({ item }: { item: StreamItem & { type: 'text' } }) {
  const [expanded, setExpanded] = useState(false)
  const bodyRef = useRef<HTMLPreElement>(null)
  const long = item.text.length > 260

  useEffect(() => {
    const body = bodyRef.current
    if (!body) return
    // While it streams, ride the bottom. Once it closes, show the START of the
    // block -- a two-line teaser parked at the end of a finished payload reads
    // as a rendering bug rather than a summary.
    body.scrollTop = item.open ? body.scrollHeight : 0
  }, [item.text, item.open])

  const collapsed = long && !expanded && !item.open
  return (
    <div className={`text-block${item.open ? ' open' : ''}`}>
      <div className="text-head">
        <span className="mono muted">{textPhaseLabel(item.phase)}</span>
        {item.simulated && <span className="chip">replay</span>}
        {long && (
          <button type="button" onClick={() => setExpanded((v) => !v)}>
            {collapsed ? `expand · ${item.text.length} chars` : 'collapse'}
          </button>
        )}
      </div>
      <pre ref={bodyRef} className={`text-body${collapsed ? ' collapsed' : ''}`}>
        {item.text}
        {item.open && <span className="caret" />}
      </pre>
    </div>
  )
}

function ToolBadge({ item }: { item: StreamItem & { type: 'tool' } }) {
  const [open, setOpen] = useState(false)
  const args = item.args ? Object.keys(item.args) : []
  return (
    <div>
      <div className="badge-line">
        <button
          type="button"
          className={`tool-badge ${item.state}`}
          onClick={() => setOpen((v) => !v)}
          title="show the call payload"
        >
          <span className="arrow">{item.state === 'result' ? '✓' : '→'}</span>
          <span>{item.tool}</span>
          {args.length > 0 && <span className="tool-args">({args.join(', ')})</span>}
        </button>
        {item.state === 'call' && <span className="mono muted">running…</span>}
        {item.state === 'result' && <span className="mono muted">{describe(item.result)}</span>}
        {(item.state === 'deny' || item.state === 'error') && (
          <span className="mono" style={{ color: 'var(--amber)' }}>
            {item.state === 'deny' ? 'refused' : 'error'} · {item.error}
          </span>
        )}
      </div>
      {open && (
        <pre className="text-body" style={{ marginTop: 6 }}>
          {JSON.stringify({ arguments: item.args ?? {}, result: item.result }, null, 2)}
        </pre>
      )}
    </div>
  )
}

type Props = {
  item: StreamItem
  onOpenNode?: (nodeId: string) => void
}

export function ItemView({ item, onOpenNode }: Props) {
  switch (item.type) {
    case 'phase':
      return (
        <div>
          <div className="phase-title">{item.label}</div>
          {item.detail && <div className="phase-detail">{item.detail}</div>}
        </div>
      )
    case 'note':
      return <div className={`note-text ${item.tone}`}>{item.text}</div>
    case 'text':
      return <TextBlock item={item} />
    case 'tool':
      return <ToolBadge item={item} />
    case 'scope':
      return (
        <div className="badge-line">
          <span className="chip">axis · {item.axis}</span>
          {item.tools.map((tool) => (
            <span key={tool} className="chip">
              {tool}
            </span>
          ))}
          {item.maxToolCalls !== undefined && (
            <span className="mono muted">
              budget {item.maxSteps} steps / {item.maxToolCalls} calls
            </span>
          )}
        </div>
      )
    case 'domain':
      return (
        <button
          type="button"
          className="domain-card"
          onClick={() => onOpenNode?.(`demi:${item.name}`)}
        >
          <div className="head">
            <span className="chip solid">
              {item.stage === 'sealed' ? 'demigod sealed' : 'domain'}
            </span>
            <span className="name">{item.name}</span>
            {item.axis && <span className="chip">{item.axis}</span>}
          </div>
          {item.language && <div className="lang">{item.language}</div>}
          {item.tools.length > 0 && (
            <div className="badge-line">
              {item.tools.map((tool) => (
                <span key={tool} className="mono muted">
                  {tool}
                </span>
              ))}
            </div>
          )}
        </button>
      )
    case 'collect':
      return (
        <button
          type="button"
          className="collect-card"
          onClick={() => onOpenNode?.(`demi:${item.name}`)}
        >
          <div className="head badge-line">
            <span className="chip ok">artifact accepted</span>
            <span className="name mono">{item.name}</span>
            {item.confidence !== undefined && (
              <span className="chip">{Math.round(item.confidence * 100)}% confidence</span>
            )}
            {item.toolCalls !== undefined && <span className="chip">{item.toolCalls} tool calls</span>}
          </div>
          {item.conclusion && <div className="conclusion">{item.conclusion}</div>}
        </button>
      )
    case 'artifact':
      return (
        <div>
          <div className="badge-line" style={{ marginBottom: 4 }}>
            <span className="chip ok">finding</span>
            {item.confidence !== undefined && (
              <span className="chip">{Math.round(item.confidence * 100)}%</span>
            )}
          </div>
          <div className="conclusion">{item.text}</div>
        </div>
      )
    case 'reason':
      return (
        <div>
          <div className="mono muted" style={{ marginBottom: 4 }}>
            reasoning summary
          </div>
          <div className="conclusion">{item.text}</div>
        </div>
      )
  }
}

type RowProps = Props & {
  /** The demigod this row belongs to, or null for GOD's own lane. */
  lane?: string | null
  onLaneClick?: () => void
}

export function StreamRow({ item, lane, onLaneClick, onOpenNode }: RowProps) {
  const classes = ['stream-row']
  if (item.type === 'phase') classes.push('phase-row')
  if (lane) classes.push('lane-row')
  return (
    <div className={classes.join(' ')}>
      <div className="stream-time">{fmtTime(item.t)}</div>
      <div style={{ minWidth: 0 }}>
        {lane && (
          <button type="button" className="lane-tag" onClick={onLaneClick}>
            {lane}
          </button>
        )}
        <ItemView item={item} onOpenNode={onOpenNode} />
      </div>
    </div>
  )
}
