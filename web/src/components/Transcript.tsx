import { useEffect } from 'react'
import type { RunNode } from '../lib/types'
import { StreamRow } from './StreamItems'

type Props = {
  node: RunNode | null
  onClose: () => void
  onOpenNode: (nodeId: string) => void
}

/** One node's whole transcript, in a drawer, because reading needs width. */
export function Transcript({ node, onClose, onOpenNode }: Props) {
  useEffect(() => {
    if (!node) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [node, onClose])

  if (!node) return null

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`${node.name} transcript`}>
        <header className="drawer-head">
          <div className="drawer-title">
            <span className={`dot-status ${node.status}`} />
            <h2>{node.name}</h2>
            <span className="chip">{node.kind === 'god' ? 'root' : 'demigod'}</span>
            <span className="topbar-spacer" />
            <button type="button" className="btn ghost small" onClick={onClose}>
              close
            </button>
          </div>
          <div className="kv">
            {node.axis && (
              <>
                <span className="k">axis</span>
                <span>{node.axis}</span>
              </>
            )}
            {node.language && (
              <>
                <span className="k">language</span>
                <span>{node.language}</span>
              </>
            )}
            {node.tools.length > 0 && (
              <>
                <span className="k">tools</span>
                <span className="badge-line">
                  {node.tools.map((tool) => (
                    <span className="chip" key={tool}>
                      {tool}
                    </span>
                  ))}
                </span>
              </>
            )}
            {node.confidence !== undefined && (
              <>
                <span className="k">confidence</span>
                <span>{Math.round(node.confidence * 100)}%</span>
              </>
            )}
            {node.error && (
              <>
                <span className="k">error</span>
                <span style={{ color: 'var(--amber)' }}>{node.error}</span>
              </>
            )}
          </div>
        </header>
        <div className="drawer-body">
          <div className="stream">
            {node.items.map((item) => (
              <StreamRow key={item.id} item={item} onOpenNode={onOpenNode} />
            ))}
          </div>
          {node.items.length === 0 && (
            <p className="empty">
              {node.status === 'planned'
                ? 'planned — this domain has not been sealed yet'
                : 'nothing recorded for this node'}
            </p>
          )}
        </div>
      </aside>
    </>
  )
}
