import { useEffect, useRef, useState } from 'react'
import type { StartRequest } from './lib/api'
import { useRun } from './lib/useRun'
import { Landing } from './components/Landing'
import { StreamPane } from './components/StreamPane'
import { TreeView } from './components/TreeView'
import { Transcript } from './components/Transcript'

function clearRunFromUrl() {
  const url = new URL(window.location.href)
  url.searchParams.delete('run')
  window.history.replaceState(null, '', url)
}

export default function App() {
  const { state, version, start, resume, stop, reset, connectionError } = useRun()
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<'stream' | 'tree'>('stream')

  // The run id lives in the URL, so a reload rejoins the run instead of losing
  // it. The events themselves are never lost -- the server keeps the whole log
  // and the stream replays it -- but a tab that cannot name the run it was
  // watching may as well have lost them.
  const resumedRef = useRef(false)
  useEffect(() => {
    if (resumedRef.current) return
    resumedRef.current = true
    const wanted = new URLSearchParams(window.location.search).get('run')
    if (!wanted) return
    resume(wanted).catch((exc: Error) => {
      setStartError(`could not resume ${wanted}: ${exc.message}`)
      clearRunFromUrl()
    })
  }, [resume])

  async function onStart(request: StartRequest, question: string) {
    setStarting(true)
    setStartError(null)
    try {
      const runId = await start(request, question)
      const url = new URL(window.location.href)
      url.searchParams.set('run', runId)
      window.history.replaceState(null, '', url)
    } catch (exc) {
      setStartError((exc as Error).message)
    } finally {
      setStarting(false)
    }
  }

  if (state.status === 'idle') {
    return <Landing busy={starting} error={startError} onStart={onStart} />
  }

  const running = state.status === 'running'

  return (
    <div className="shell">
      <header className="topbar">
        <div className="wordmark">
          <span className="re">re:</span>SOLUTION
        </div>
        <div className="topbar-meta">
          <span className="chip">{state.mode === 'scripted' ? 'replay' : 'live'}</span>
          <span className={`chip${state.execution === 'inprocess' ? '' : ' solid'}`}>
            {state.execution === 'godbox'
              ? 'sandbox'
              : state.execution === 'sandbox'
                ? 'demigod sandboxes'
                : 'in-process'}
          </span>
          <span className="chip">{state.runId}</span>
          <span className="chip">{state.elapsed.toFixed(1)}s</span>
          {running ? (
            <span className="chip live">running</span>
          ) : (
            <span className={`chip ${state.status === 'done' ? 'ok' : 'warn'}`}>
              {state.status}
            </span>
          )}
        </div>
        <span className="topbar-question">{state.question}</span>
        <div className="topbar-spacer" />
        <div className="seg tabs" role="group" aria-label="View">
          <button type="button" aria-pressed={view === 'stream'} onClick={() => setView('stream')}>
            Stream
          </button>
          <button type="button" aria-pressed={view === 'tree'} onClick={() => setView('tree')}>
            Tree
          </button>
        </div>
        {running ? (
          <button type="button" className="btn ghost small" onClick={() => void stop()}>
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="btn small"
            onClick={() => {
              setSelected(null)
              clearRunFromUrl()
              reset()
            }}
          >
            New run
          </button>
        )}
      </header>

      {connectionError && <p className="notice error">{connectionError}</p>}

      {/* Both columns always render; below 1000px the CSS shows whichever the
          tab selects, so switching views never unmounts a scrolled stream. */}
      <main className={`workspace view-${view}`}>
        <StreamPane state={state} version={version} onOpenNode={setSelected} />
        <TreeView state={state} selected={selected} onSelect={setSelected} />
      </main>

      <Transcript
        node={selected ? (state.nodes[selected] ?? null) : null}
        onClose={() => setSelected(null)}
        onOpenNode={setSelected}
      />
    </div>
  )
}
