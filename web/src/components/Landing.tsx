import { useEffect, useRef, useState } from 'react'
import { fetchPresets, uploadAttachment } from '../lib/api'
import type { StartRequest } from '../lib/api'
import type { Attachment, Preset } from '../lib/types'
import { PixelCreation } from './PixelCreation'

type Props = {
  busy: boolean
  error: string | null
  onStart: (request: StartRequest, question: string) => void
}

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

export function Landing({ busy, error, onStart }: Props) {
  const [presets, setPresets] = useState<Preset[]>([])
  const [liveAvailable, setLiveAvailable] = useState(false)
  const [liveReason, setLiveReason] = useState<string | null>(null)
  const [mode, setMode] = useState<'scripted' | 'live'>('scripted')
  const [presetId, setPresetId] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploading, setUploading] = useState(0)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  // `null` is the dynamic option: GOD reads the problem and decides how many
  // domains it is worth, down to none at all for something it can just answer.
  const [domains, setDomains] = useState<number | null>(3)
  const [execution, setExecution] = useState<'inprocess' | 'godbox'>('godbox')
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    fetchPresets()
      .then(({ presets: found, liveAvailable: live, liveReason: reason }) => {
        setPresets(found)
        setLiveAvailable(live)
        setLiveReason(reason)
      })
      .catch((exc: Error) => setLoadError(exc.message))
  }, [])

  // Keep the selection legal for the current mode. NOT just a nicety: only one
  // preset has a recording, and the list is ordered easy-to-hard rather than
  // recorded-first, so selecting `presets[0]` on load put an unrecorded problem
  // in the box while the page sat in replay mode -- prompt and preset card
  // disagreeing, and Run disabled with nothing saying why.
  useEffect(() => {
    if (presets.length === 0) return
    const current = presets.find((p) => p.id === presetId)
    if (current && current.modes.includes(mode)) return
    const fallback = presets.find((p) => p.modes.includes(mode))
    if (fallback) applyPreset(fallback)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, presets])

  function applyPreset(preset: Preset) {
    setPresetId(preset.id)
    setPrompt(preset.prompt)
  }

  async function addFiles(files: FileList | File[]) {
    // Only reachable in live mode: the dropzone renders under `!replay`, and a
    // replay would ignore the data anyway. The Replay button locks once
    // anything is attached, which is the other half of that rule.
    setUploadError(null)
    for (const file of Array.from(files)) {
      setUploading((n) => n + 1)
      try {
        const attachment = await uploadAttachment(file)
        // Replace by name rather than append: re-dropping an edited file is a
        // correction, not a second table, and mounting both under the same
        // shared/ name would make which one a demigod reads a coin flip.
        setAttachments((current) => [
          ...current.filter((a) => a.name !== attachment.name),
          attachment,
        ])
      } catch (exc) {
        setUploadError((exc as Error).message)
      } finally {
        setUploading((n) => n - 1)
      }
    }
  }

  const replay = mode === 'scripted'
  const usable = presets.filter((p) => p.modes.includes(mode))
  const selected = presets.find((p) => p.id === presetId) ?? null
  const unrecorded = replay && selected != null && !selected.modes.includes('scripted')
  const sealed = attachments.flatMap((a) => a.terms)
  const canRun =
    !busy &&
    !unrecorded &&
    uploading === 0 &&
    (replay ? selected != null : prompt.trim().length > 0)

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
    if (attachments.length) {
      // Ids, not bytes. The files are already on the server; this only says
      // which of them this run should see.
      request.attachments = attachments.map((a) => a.id)
    }
    if (replay || pristine) {
      request.preset = selected!.id
    } else {
      request.prompt = prompt.trim()
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
                disabled={attachments.length > 0}
                title={
                  attachments.length
                    ? 'a replay answers the recorded problem, so it would ignore your data'
                    : undefined
                }
                onClick={() => attachments.length === 0 && setMode('scripted')}
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
                <button
                  type="button"
                  className="icon"
                  aria-pressed={domains === null}
                  aria-label="Let GOD choose the number of domains"
                  disabled={replay}
                  title={
                    replay
                      ? 'the recording fixes the domain count at 3'
                      : 'let GOD decide — it spawns only what the problem needs, and nothing at all for one it can just answer'
                  }
                  onClick={() => setDomains(null)}
                >
                  ✦
                </button>
              </div>
            </div>
          </div>

          {!replay && (
            <div className="composer-row execution-row">
              <span className="mono muted">
                {execution === 'godbox'
                  ? 'god in its own sandbox, spawning demigod sandboxes'
                  : 'everything inside this server · no isolation'}
              </span>
              <div className="seg" role="group" aria-label="Where the run executes">
                <button
                  type="button"
                  aria-pressed={execution === 'godbox'}
                  onClick={() => setExecution('godbox')}
                >
                  Sandbox
                </button>
                <button
                  type="button"
                  aria-pressed={execution === 'inprocess'}
                  onClick={() => setExecution('inprocess')}
                >
                  In-process
                </button>
              </div>
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
              <label className="mono field-label" htmlFor="attach">
                Data — CSV, TSV or parquet
              </label>
              <div
                className={`dropzone${dragging ? ' over' : ''}`}
                onDragOver={(event) => {
                  event.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragging(false)
                  if (event.dataTransfer.files.length) void addFiles(event.dataTransfer.files)
                }}
              >
                <input
                  id="attach"
                  ref={fileInput}
                  type="file"
                  multiple
                  accept=".csv,.tsv,.parquet"
                  onChange={(event) => {
                    if (event.target.files?.length) void addFiles(event.target.files)
                    // Clear it, or re-picking the same file fires no change event.
                    event.target.value = ''
                  }}
                />
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => fileInput.current?.click()}
                >
                  Choose files
                </button>
                <span className="mono muted">
                  {uploading > 0
                    ? `profiling ${uploading} file${uploading > 1 ? 's' : ''}…`
                    : 'or drop them here · the table is mounted read-only under shared/'}
                </span>
              </div>

              {attachments.length > 0 && (
                <ul className="attachments">
                  {attachments.map((file) => (
                    <li key={file.id}>
                      <div className="attachment-head">
                        <span className="name">{file.name}</span>
                        <span className="mono muted">
                          {file.profile.rows.toLocaleString()} rows ·{' '}
                          {file.profile.columns.length} cols · {humanSize(file.size)}
                        </span>
                        <button
                          type="button"
                          className="btn ghost small"
                          aria-label={`Remove ${file.name}`}
                          onClick={() =>
                            setAttachments((current) =>
                              current.filter((a) => a.id !== file.id),
                            )
                          }
                        >
                          Remove
                        </button>
                      </div>
                      <div className="mono columns">
                        {file.profile.columns.map((column) => (
                          <span key={column.name} className={`col ${column.type}`}>
                            {column.name}
                          </span>
                        ))}
                        {file.profile.columns_omitted ? (
                          <span className="col muted">
                            +{file.profile.columns_omitted} more
                          </span>
                        ) : null}
                      </div>
                    </li>
                  ))}
                </ul>
              )}

              {sealed.length > 0 && (
                <p className="notice seal">
                  <strong>{sealed.length} terms</strong> taken from your schema are hidden
                  from every demigod: <span className="mono">{sealed.slice(0, 12).join(', ')}</span>
                  {sealed.length > 12 ? `, +${sealed.length - 12} more` : ''}. The projected
                  summary of each table is sealed — but the file itself is mounted with its
                  real headers, so a demigod that opens it sees them.
                </p>
              )}
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
                <span className="preset-head">
                  <span className="mono muted">{preset.id}</span>
                  <span className={`tier ${preset.tier}`}>{preset.tier}</span>
                </span>
                <span className="name">{preset.label}</span>
                <span className="blurb">{preset.blurb}</span>
              </button>
            ))}
          </div>

          {(error ?? loadError ?? uploadError) && (
            <p className="notice error">{error ?? loadError ?? uploadError}</p>
          )}

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
