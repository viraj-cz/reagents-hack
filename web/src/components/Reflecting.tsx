import { useEffect, useState } from 'react'

/**
 * The twelve ways of thinking, as the wordmark writes them -- everything after
 * the "re:" of re:SOLUTION. A run that has gone quiet is not stalled, it is
 * doing one of these, so the indicator names the act rather than counting
 * seconds at a reader who cannot do anything with the number.
 */
const WORDS = [
  'FLECT',
  'ASON',
  'CONSIDER',
  'CALL',
  'COLLECT',
  'MINISCE',
  'CKON',
  'THINK',
  'VIEW',
  'ASSESS',
  'VISIT',
  'VOLVE',
]

// One scramble every three seconds: hold the word, churn, arrive on the next.
const CYCLE_MS = 3000
const SCRAMBLE_MS = 620
const HOLD_MS = CYCLE_MS - SCRAMBLE_MS
// Slow enough that the churn reads as letters rather than as grey noise.
const TICK_MS = 45

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

function glyph() {
  return GLYPHS[Math.floor(Math.random() * GLYPHS.length)]
}

/**
 * One frame of the churn between two words, `p` running 0 -> 1.
 *
 * Width interpolates so VIEW -> CONSIDER grows into place instead of snapping,
 * and letters lock left to right, which reads as a word arriving rather than as
 * text being swapped out. At p = 1 the width is exactly the target's and every
 * position has locked, so the last frame is the word itself.
 */
function frame(from: string, to: string, p: number) {
  const width = Math.round(from.length + (to.length - from.length) * p)
  let out = ''
  for (let i = 0; i < width; i++) {
    out += i < to.length && p >= (i + 1) / (to.length + 1) ? to[i] : glyph()
  }
  return out
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return reduced
}

/**
 * The waiting indicator: re:FLECT, re:ASON, re:CONSIDER...
 *
 * Mounted only while the stream is quiet, so the timers stop the moment an
 * event lands. Under prefers-reduced-motion the words still turn over on the
 * same three-second beat, they just cut rather than churn -- a text field
 * changing twenty-two times a second is exactly what that setting is for.
 */
export function Reflecting() {
  const reduced = usePrefersReducedMotion()
  const [index, setIndex] = useState(0)
  const [text, setText] = useState(WORDS[0])

  useEffect(() => {
    const next = (index + 1) % WORDS.length
    let churn: ReturnType<typeof setInterval> | undefined

    const hold = setTimeout(
      () => {
        if (reduced) {
          setText(WORDS[next])
          setIndex(next)
          return
        }
        const from = WORDS[index]
        const started = performance.now()
        churn = setInterval(() => {
          const p = (performance.now() - started) / SCRAMBLE_MS
          if (p >= 1) {
            clearInterval(churn)
            setText(WORDS[next])
            setIndex(next)
          } else {
            setText(frame(from, WORDS[next], p))
          }
        }, TICK_MS)
      },
      reduced ? CYCLE_MS : HOLD_MS,
    )

    return () => {
      clearTimeout(hold)
      if (churn) clearInterval(churn)
    }
  }, [index, reduced])

  return (
    <span className="mono reflecting" aria-label="still working">
      <span className="re" aria-hidden="true">
        re:
      </span>
      <span aria-hidden="true">{text}</span>
    </span>
  )
}
