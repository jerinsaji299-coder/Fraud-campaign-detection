import { useMemo, useState } from 'react'
import { RotateCcw } from 'lucide-react'

import { binOf, visibilityOf, type Assignment, type Edge } from '../lib/visibility'
import { BIN_LABELS, binColor, institutionColor, institutionLabel } from '../theme'
import { useThemeMode } from '../lib/themeContext'
import type { BinKey } from '../theme'

/**
 * A hands-on scenario the user builds themselves. Because the backend has
 * never seen this arrangement, the visibility number here is computed in the
 * browser — using `visibilityOf`, which mirrors the backend definition
 * exactly. It is the only research metric the frontend ever computes.
 */

const ZONES = [0, 1, 2]

interface Scenario {
  key: string
  label: string
  description: string
  accounts: string[]
  edges: Edge[]
  initial: Assignment
}

const SCENARIOS: Scenario[] = [
  {
    key: 'fan-out',
    label: 'Fan-out (hub shape)',
    description:
      'One account pays out to four others. Try as hard as you like: the hub’s own bank always sees every transaction, so visibility never drops below 100%.',
    accounts: ['Hub', 'A', 'B', 'C', 'D'],
    edges: [
      { src: 'Hub', dst: 'A' },
      { src: 'Hub', dst: 'B' },
      { src: 'Hub', dst: 'C' },
      { src: 'Hub', dst: 'D' },
    ],
    initial: { Hub: 0, A: 0, B: 0, C: 0, D: 0 },
  },
  {
    key: 'cycle',
    label: 'Cycle (fragmentable shape)',
    description:
      'Money moves around a ring of six accounts. Spread them across the three banks and watch visibility fall — this is the shape the study can actually hide.',
    accounts: ['A', 'B', 'C', 'D', 'E', 'F'],
    edges: [
      { src: 'A', dst: 'B' },
      { src: 'B', dst: 'C' },
      { src: 'C', dst: 'D' },
      { src: 'D', dst: 'E' },
      { src: 'E', dst: 'F' },
      { src: 'F', dst: 'A' },
    ],
    initial: { A: 0, B: 0, C: 0, D: 0, E: 0, F: 0 },
  },
]

export function VisibilityToy() {
  const { mode } = useThemeMode()
  const [scenarioKey, setScenarioKey] = useState(SCENARIOS[1].key)
  const scenario = SCENARIOS.find((s) => s.key === scenarioKey) ?? SCENARIOS[1]
  const [assignment, setAssignment] = useState<Assignment>(scenario.initial)
  const [dragging, setDragging] = useState<string | null>(null)

  const achieved = useMemo(
    () => visibilityOf(scenario.edges, assignment, ZONES.length),
    [scenario.edges, assignment],
  )
  const bin = binOf(achieved)

  const move = (account: string, zone: number) =>
    setAssignment((current) => ({ ...current, [account]: zone }))

  const reset = (next: Scenario) => {
    setScenarioKey(next.key)
    setAssignment(next.initial)
  }

  return (
    <div className="rounded-xl border border-grid bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">Try it yourself</h2>
          <p className="mt-1 max-w-xl text-sm leading-relaxed text-ink2">
            {scenario.description}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
            Shape
            <select
              value={scenarioKey}
              onChange={(event) => {
                const next = SCENARIOS.find((s) => s.key === event.target.value)
                if (next) reset(next)
              }}
              className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
            >
              {SCENARIOS.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => reset(scenario)}
            className="mt-5 inline-flex items-center gap-1.5 rounded-lg border border-grid px-2.5 py-1.5 text-sm text-ink2 hover:text-ink"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Reset
          </button>
        </div>
      </div>

      {/* live readout */}
      <div className="mt-4 flex flex-wrap items-center gap-4 rounded-lg border border-axis p-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-muted">
            Visibility
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className="text-3xl font-semibold tabular-nums text-ink"
              aria-live="polite"
              aria-label="Visibility of your scenario"
            >
              {Math.round(achieved * 100)}%
            </span>
            <span className="flex items-center gap-1.5 text-xs text-ink2">
              <span
                aria-hidden="true"
                className="h-2.5 w-2.5 rounded-sm"
                style={{ background: binColor(bin, mode) }}
              />
              bin {BIN_LABELS[bin as BinKey]}
            </span>
          </div>
        </div>
        <p className="max-w-md text-xs leading-snug text-ink2">
          The best-placed bank sees{' '}
          {Math.round(achieved * scenario.edges.length)} of {scenario.edges.length}{' '}
          transactions. A bank sees a transaction when it holds either end of it &mdash;
          which is why moving one account changes two transactions at once.
        </p>
      </div>

      {/* bank zones */}
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {ZONES.map((zone) => {
          const held = scenario.accounts.filter((account) => assignment[account] === zone)
          return (
            <div
              key={zone}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault()
                if (dragging) move(dragging, zone)
                setDragging(null)
              }}
              className="min-h-28 rounded-lg border-2 border-dashed p-3"
              style={{ borderColor: institutionColor(zone, mode) }}
            >
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-ink">
                <span
                  aria-hidden="true"
                  className="h-2.5 w-2.5 rounded-sm"
                  style={{ background: institutionColor(zone, mode) }}
                />
                Bank {institutionLabel(zone)}
              </div>
              <ul className="flex flex-wrap gap-2">
                {held.map((account) => (
                  <li key={account}>
                    <button
                      type="button"
                      draggable
                      onDragStart={() => setDragging(account)}
                      onDragEnd={() => setDragging(null)}
                      onClick={() => move(account, (zone + 1) % ZONES.length)}
                      aria-label={`Account ${account}, currently at bank ${institutionLabel(
                        zone,
                      )}. Activate to move it to the next bank.`}
                      className="cursor-grab rounded-md border border-grid bg-plane px-2 py-1 text-sm font-medium text-ink active:cursor-grabbing"
                    >
                      {account}
                    </button>
                  </li>
                ))}
                {held.length === 0 && (
                  <li className="text-xs text-muted">Drop an account here</li>
                )}
              </ul>
            </div>
          )
        })}
      </div>

      <p className="mt-3 text-xs text-muted">
        Drag an account between banks, or click it to send it to the next bank
        (keyboard-friendly).
      </p>
    </div>
  )
}
