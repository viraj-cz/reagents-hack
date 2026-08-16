import { demigodNodes, GOD } from '../lib/model'
import type { RunNode, RunState } from '../lib/types'

type Props = {
  state: RunState
  selected: string | null
  onSelect: (nodeId: string) => void
}

const STATUS_LABEL: Record<RunNode['status'], string> = {
  planned: 'planned',
  sealed: 'sealed',
  running: 'reasoning',
  done: 'complete',
  failed: 'failed',
}

/**
 * The run as a tree: GOD at the root, one child per invented domain.
 *
 * A demigod appears the moment the planner names it and then changes state in
 * place — planned, sealed, reasoning, complete. Waiting until it produces
 * something would hide exactly the interval a user most wants to watch.
 */
export function TreeView({ state, selected, onSelect }: Props) {
  const god = state.nodes[GOD]
  const children = demigodNodes(state)

  return (
    <section className="column tree-col">
      <header className="column-head">
        <span className="mono">Run tree</span>
        <span className="topbar-spacer" />
        <span className="mono muted">{children.length} demigods</span>
      </header>
      <div className="column-body">
        <div className="tree">
          <button
            type="button"
            className="tree-card root"
            aria-pressed={selected === GOD}
            onClick={() => onSelect(GOD)}
          >
            <div className="row">
              <span className={`dot-status ${god?.status ?? 'running'}`} />
              <span className="name">GOD</span>
              <span className="topbar-spacer" />
              <span className="mono muted">{state.elapsed.toFixed(1)}s</span>
            </div>
            <div className="sub">{state.question}</div>
            <div className="tree-metrics">
              <span>{state.eventCount} events</span>
              <span>· {state.toolCalls} tool calls</span>
              <span>· {state.mode}</span>
              <span>· {state.execution}</span>
            </div>
          </button>

          {children.map((node, index) => (
            <div
              className={`tree-node${index === children.length - 1 ? ' last' : ''}`}
              key={node.id}
            >
              <div className="tree-rail" />
              <button
                type="button"
                className="tree-card"
                aria-pressed={selected === node.id}
                onClick={() => onSelect(node.id)}
              >
                <div className="row">
                  <span className={`dot-status ${node.status}`} />
                  <span className="name">{node.name}</span>
                  <span className="topbar-spacer" />
                  <span className="mono muted">{STATUS_LABEL[node.status]}</span>
                </div>
                {node.axis && (
                  <div className="tree-metrics">
                    <span>{node.axis}</span>
                    {node.tools.map((tool) => (
                      <span key={tool}>· {tool}</span>
                    ))}
                  </div>
                )}
                {node.conclusion && <div className="sub">{node.conclusion}</div>}
                {node.error && (
                  <div className="sub" style={{ color: 'var(--amber)' }}>
                    {node.error}
                  </div>
                )}
                <div className="bar">
                  <span style={{ width: `${progress(node)}%` }} />
                </div>
                <div className="tree-metrics">
                  <span>{node.toolCalls} calls</span>
                  {node.confidence !== undefined && (
                    <span>· {Math.round(node.confidence * 100)}% confidence</span>
                  )}
                </div>
              </button>
            </div>
          ))}

          {children.length === 0 && <p className="empty">no domains invented yet</p>}
        </div>
      </div>
    </section>
  )
}

function progress(node: RunNode): number {
  switch (node.status) {
    case 'planned':
      return 12
    case 'sealed':
      return 34
    case 'running':
      return Math.min(88, 40 + node.toolCalls * 12)
    default:
      return 100
  }
}
