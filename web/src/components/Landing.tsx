import { useEffect, useState } from 'react'
import { fetchPresets } from '../lib/api'
import type { StartRequest } from '../lib/api'
import type { Preset } from '../lib/types'
import { PixelCreation } from './PixelCreation'

type Props = {
  busy: boolean
  error: string | null
  onStart: (request: StartRequest, question: string) => void
}

export function Landing({ busy, error, onStart }: Props) {
  const [presets, setPresets] = useState<Preset[]>([])
  const [liveAvailable, setLiveAvailable] = useState(false)
  const [liveReason, setLiveReason] = useState<string | null>(null)
  const [mode, setMode] = useState<'scripted' | 'live'>('scripted')
  const [presetId, setPresetId] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [entities, setEntities] = useState('')
  const [domains, setDomains] = useState(3)
  const [execution, setExecution] = useState<'inprocess' | 'sandbox' | 'godbox'>(
    'godbox',
  )
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    fetchPresets()
      .then(({ presets: found, liveAvailable: live, liveReason: reason }) => {
        setPresets(found)
        setLiveAvailable(live)
        setLiveReason(reason)
        const first = found[0]
        if (first) applyPreset(first)
      })
      .catch((exc: Error) => setLoadError(exc.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function applyPreset(preset: Preset) {
    setPresetId(preset.id)
    setPrompt(preset.prompt)
    setEntities(preset.entities.join(', '))
  }

  const replay = mode === 'scripted'
  const usable = presets.filter((p) => p.modes.includes(mode))
  const selected = presets.find((p) => p.id === presetId) ?? null
  const unrecorded = replay && selected != null && !selected.modes.includes('scripted')
  const canRun = !busy && !unrecorded && (replay ? selected != null : prompt.trim().length > 0)

  function submit() {
    if (!canRun) return
    const pristine = selected != null && prompt.trim() === selected.prompt.trim()
    // A sandbox demigod runs the real agent against the real API; pairing it
    // with a replay would spend tokens re-enacting a recording, so the server
    // refuses that combination and the UI never offers it.
    const request: StartRequest = {
      mode,
      domains,
      execution: replay ? 'inprocess' : execution,
    }
    if (replay || pristine) {
      request.preset = selected!.id
    } else {
      request.prompt = prompt.trim()
      request.entities = entities
        .split(',')
        .map((e) => e.trim())
        .filter(Boolean)
    }
    onStart(request, (selected && pristine ? selected.prompt : prompt).trim())
  }

  return (
    <div className="landing">
      <div className="landing-inner">
        <header className="hero">
          <h1>
            <span className="re">re:</span>SOLUTION
          </h1>
          <p className="tagline">reasoning through translation</p>
          <p className="mono stamp">GOD · DEMI_GOD · ORTHOGONAL DOMAINS</p>
          <div className="hero-art">
            <PixelCreation scale={0.56} />
          </div>
          <p className="hero-blurb">
            One problem, projected into several invented representations. Each is sealed so no
            demigod can see the original vocabulary, solved in isolation, and mapped back.
          </p>
        </header>

        <div className="composer">
          <div className="composer-row">
            <div className="seg" role="group" aria-label="Execution mode">
              <button
                type="button"
                aria-pressed={mode === 'scripted'}
                onClick={() => setMode('scripted')}
              >
                Replay
              </button>
              <button
                type="button"
                aria-pressed={mode === 'live'}
                disabled={!liveAvailable}
                title={liveAvailable ? undefined : (liveReason ?? 'live runs are unavailable')}
                onClick={() => liveAvailable && setMode('live')}
              >
                Live
              </button>
            </div>
            <span className="mono muted">
              {replay ? 'recorded · no api key · no tokens' : 'anthropic inference · real cost'}
            </span>
            <div className="topbar-spacer" />
            <div className="composer-group">
              <span className="mono muted">domains</span>
              <div className="seg" role="group" aria-label="Number of domains">
                {[2, 3, 4].map((n) => (
                  <button
                    key={n}
                    type="button"
                    aria-pressed={domains === n}
                    disabled={replay}
                    title={replay ? 'the recording fixes the domain count at 3' : undefined}
                    onClick={() => setDomains(n)}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {!replay && (
            <div className="composer-row">
              <div className="seg" role="group" aria-label="Where the run executes">
                <button
                  type="button"
                  aria-pressed={execution === 'godbox'}
                  onClick={() => setExecution('godbox')}
                >
                  God sandbox
                </button>
                <button
                  type="button"
                  aria-pressed={execution === 'sandbox'}
                  onClick={() => setExecution('sandbox')}
                >
                  Demigod sandboxes
                </button>
                <button
                  type="button"
                  aria-pressed={execution === 'inprocess'}
                  onClick={() => setExecution('inprocess')}
                >
                  In-process
                </button>
              </div>
              <span className="mono muted">
                {execution === 'godbox'
                  ? 'god in its own sandbox, spawning demigod sandboxes'
                  : execution === 'sandbox'
                    ? 'god here · one sandbox per demigod · brokered tools'
                    : 'everything inside this server · no isolation'}
              </span>
            </div>
          )}

          <div>
            <label className="mono field-label" htmlFor="prompt">
              Problem statement
            </label>
            <div className="prompt-box">
              <textarea
                id="prompt"
                value={prompt}
                disabled={replay}
                placeholder="Describe the problem. GOD will invent orthogonal representations of it, seal each one, and reason about them separately."
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submit()
                }}
              />
            </div>
          </div>

          {!replay && (
            <div>
              <label className="mono field-label" htmlFor="entities">
                Native entities — the words no demigod may see
              </label>
              <div className="prompt-box" style={{ padding: '10px 14px' }}>
                <textarea
                  id="entities"
                  value={entities}
                  style={{ minHeight: 44 }}
                  placeholder="comma separated · leave empty to run without semantic sealing"
                  onChange={(event) => setEntities(event.target.value)}
                />
              </div>
            </div>
          )}

          {replay && (
            <p className="notice">
              Replay runs the recorded glycolysis problem against the scripted model: the same
              orchestration, the same tools, no inference. The prompt is fixed because the
              recording only covers this one problem.
            </p>
          )}

          <div className="presets">
            {usable.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className="preset"
                aria-pressed={preset.id === presetId}
                onClick={() => applyPreset(preset)}
              >
                <span className="mono muted">{preset.id}</span>
                <span className="name">{preset.label}</span>
                <span className="blurb">{preset.blurb}</span>
              </button>
            ))}
          </div>

          {(error ?? loadError) && <p className="notice error">{error ?? loadError}</p>}

          <div className="composer-row">
            <button type="button" className="btn primary" disabled={!canRun} onClick={submit}>
              {busy ? 'Starting…' : replay ? 'Run replay' : 'Run live'}
            </button>
            <span className="mono muted">or press cmd + enter</span>
          </div>
        </div>
      </div>
    </div>
  )
}
