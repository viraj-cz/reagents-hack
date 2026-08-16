import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelRun, fetchRun, startRun, type StartRequest } from './api'
import { applyEvent, emptyRun, startRun as freshRun } from './model'
import type { RunState, UiEvent } from './types'

/**
 * Owns the EventSource and the run state.
 *
 * Two things here are less obvious than they look:
 *
 * 1. Events are buffered and applied once per animation frame. A replayed run
 *    delivers a text delta every ~12ms and a live one can be faster; reducing
 *    per-message meant a React render per delta, which is how you get a stream
 *    that lags behind a model that has already stopped talking.
 * 2. The state object is mutated by `applyEvent` and never replaced, so a
 *    version counter -- not object identity -- is what tells React to paint.
 */
export function useRun() {
  const stateRef = useRef<RunState>(emptyRun('scripted'))
  const [version, setVersion] = useState(0)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const sourceRef = useRef<EventSource | null>(null)
  const bufferRef = useRef<UiEvent[]>([])
  const frameRef = useRef<number | null>(null)
  const startedAtRef = useRef<number | null>(null)

  const flush = useCallback(() => {
    frameRef.current = null
    const batch = bufferRef.current
    if (!batch.length) return
    bufferRef.current = []
    for (const event of batch) applyEvent(stateRef.current, event)
    setVersion((v) => v + 1)
  }, [])

  const schedule = useCallback(() => {
    if (frameRef.current !== null) return
    frameRef.current = requestAnimationFrame(flush)
  }, [flush])

  const close = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    frameRef.current = null
  }, [])

  const attach = useCallback(
    (runId: string) => {
      close()
      bufferRef.current = []
      const source = new EventSource(`/api/runs/${runId}/stream`)
      sourceRef.current = source
      source.onmessage = (message) => {
        try {
          bufferRef.current.push(JSON.parse(message.data) as UiEvent)
          schedule()
        } catch {
          /* a frame we cannot parse is not worth killing the stream over */
        }
      }
      // Flush BEFORE closing. `close()` cancels the pending animation frame,
      // and the last batch of a run -- the final answer and the run_end that
      // flips the status out of "running" -- is almost always still in it.
      source.addEventListener('end', () => {
        flush()
        close()
      })
      source.onerror = () => {
        // EventSource reconnects on its own; only a run that already finished
        // should surface as closed, and that arrives as the `end` event above.
        if (stateRef.current.status === 'running') {
          setConnectionError('stream interrupted — retrying')
        }
      }
    },
    [close, flush, schedule],
  )

  const start = useCallback(
    async (request: StartRequest, question: string) => {
      setConnectionError(null)
      const { run_id } = await startRun(request)
      startedAtRef.current = performance.now()
      stateRef.current = freshRun(run_id, request.mode, request.execution, question)
      setVersion((v) => v + 1)
      attach(run_id)
      return run_id
    },
    [attach],
  )

  /**
   * Pick up a run this tab did not start -- after a reload, or from another
   * window. Only the run's METADATA is fetched; the events arrive over the
   * stream, which replays the whole log from seq 1 before going live. Applying
   * the fetched events too would double every one of them.
   */
  const resume = useCallback(
    async (runId: string) => {
      setConnectionError(null)
      const run = await fetchRun(runId)
      stateRef.current = freshRun(
        run.run_id,
        run.mode,
        run.execution,
        run.question,
      )
      // The clock is the run's, not this tab's: a run resumed at second 90 is
      // ninety seconds old, however long this page has been open.
      startedAtRef.current = performance.now() - (run.started_at ? (Date.now() / 1000 - run.started_at) * 1000 : 0)
      if (run.status !== 'running') stateRef.current.status = run.status
      setVersion((v) => v + 1)
      attach(runId)
    },
    [attach],
  )

  const stop = useCallback(async () => {
    const runId = stateRef.current.runId
    if (runId) await cancelRun(runId)
  }, [])

  const reset = useCallback(() => {
    close()
    stateRef.current = emptyRun(stateRef.current.mode)
    setVersion((v) => v + 1)
  }, [close])

  useEffect(() => close, [close])

  // A running clock, independent of events: silence during a long model call
  // should still look like time passing rather than a frozen UI. The status it
  // watches lives in a ref, so the interval is unconditional and the check is
  // inside it -- an effect dependency would never see the change.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const state = stateRef.current
      if (state.status !== 'running' || !startedAtRef.current) return
      state.elapsed = (performance.now() - startedAtRef.current) / 1000
      setVersion((v) => v + 1)
    }, 100)
    return () => window.clearInterval(timer)
  }, [])

  return {
    state: stateRef.current,
    version,
    start,
    resume,
    stop,
    reset,
    connectionError,
  }
}
