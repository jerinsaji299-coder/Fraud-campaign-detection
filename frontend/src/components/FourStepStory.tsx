import { institutionColor } from '../theme'
import { useThemeMode } from '../lib/themeContext'

/**
 * The whole research question in four pictures: a campaign exists, it gets
 * split across banks, each bank sees only a fragment, collaboration puts the
 * fragments back together.
 *
 * These are illustrations of the *mechanism*, not plots of data.
 */

const NODES = [
  { x: 20, y: 20 },
  { x: 60, y: 14 },
  { x: 92, y: 40 },
  { x: 74, y: 74 },
  { x: 32, y: 70 },
]

const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 0],
]

/** Which institution owns each account in the "split" steps. */
const OWNER = [0, 1, 1, 2, 0]

function Diagram({
  step,
  mode,
}: {
  step: 1 | 2 | 3 | 4
  mode: 'light' | 'dark'
}) {
  const neutral = mode === 'dark' ? '#c3c2b7' : '#52514e'

  const edgeStyle = (edge: [number, number]) => {
    if (step === 3) {
      // one bank's point of view: it only holds the edges touching its accounts
      const seen = OWNER[edge[0]] === 0 || OWNER[edge[1]] === 0
      return { stroke: seen ? neutral : neutral, opacity: seen ? 1 : 0.15 }
    }
    if (step === 4) return { stroke: neutral, opacity: 1 }
    return { stroke: neutral, opacity: step === 1 ? 1 : 0.55 }
  }

  const nodeFill = (index: number) => {
    if (step === 1) return neutral
    const color = institutionColor(OWNER[index], mode)
    if (step === 3 && OWNER[index] !== 0) return mode === 'dark' ? '#383835' : '#e1e0d9'
    return color
  }

  return (
    <svg viewBox="0 0 112 90" className="h-24 w-full" role="img" aria-hidden="true">
      {step === 4 && (
        <rect
          x="2"
          y="2"
          width="108"
          height="86"
          rx="8"
          fill="none"
          stroke={institutionColor(5, mode)}
          strokeWidth="2"
          strokeDasharray="4 3"
        />
      )}
      {EDGES.map((edge, i) => {
        const style = edgeStyle(edge)
        return (
          <line
            key={i}
            x1={NODES[edge[0]].x}
            y1={NODES[edge[0]].y}
            x2={NODES[edge[1]].x}
            y2={NODES[edge[1]].y}
            stroke={style.stroke}
            strokeOpacity={style.opacity}
            strokeWidth="2"
          />
        )
      })}
      {NODES.map((node, i) => (
        <circle
          key={i}
          cx={node.x}
          cy={node.y}
          r="7"
          fill={nodeFill(i)}
          stroke={mode === 'dark' ? '#1a1a19' : '#fcfcfb'}
          strokeWidth="2"
        />
      ))}
    </svg>
  )
}

const STEPS = [
  {
    step: 1 as const,
    title: 'A laundering campaign happens',
    body: 'Money moves through a ring of accounts. Seen whole, the shape is obvious.',
  },
  {
    step: 2 as const,
    title: 'Its accounts sit at different banks',
    body: 'In reality those accounts belong to separate institutions. Colour shows which bank holds which account.',
  },
  {
    step: 3 as const,
    title: 'Each bank sees only a fragment',
    body: 'One bank sees just the transactions touching its own accounts. The ring is no longer visible from inside any single bank.',
  },
  {
    step: 4 as const,
    title: 'Collaboration can recover the whole',
    body: 'If banks share what they have learned, the shape can reappear — the question is how much that helps as the fragments get smaller.',
  },
]

export function FourStepStory() {
  const { mode } = useThemeMode()

  return (
    <ol className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {STEPS.map(({ step, title, body }) => (
        <li key={step} className="rounded-xl border border-grid bg-surface p-4">
          <div className="mb-2 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-grid text-xs font-semibold text-ink">
              {step}
            </span>
            <h3 className="text-sm font-semibold text-ink">{title}</h3>
          </div>
          <Diagram step={step} mode={mode} />
          <p className="mt-2 text-sm leading-relaxed text-ink2">{body}</p>
        </li>
      ))}
    </ol>
  )
}
