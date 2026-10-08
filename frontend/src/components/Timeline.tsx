import { useEffect } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'

import type { CampaignTransaction } from '../api/types'

interface Props {
  transactions: CampaignTransaction[]
  value: number
  onChange: (value: number) => void
  playing: boolean
  onPlayingChange: (playing: boolean) => void
  /** True completion time; equals the cutoff when the campaign is censored. */
  deadline: string
  cutoff: string
}

const STEP_MS = 900

function toTime(value: string): number {
  return new Date(value).getTime()
}

function formatStamp(value: string): string {
  return value.slice(0, 16).replace('T', ' ')
}

/**
 * Reveals a campaign's transactions in time order. The slider steps through
 * transactions (not clock time) so every step shows something new; the strip
 * underneath places each transaction on a real time axis, with the deadline
 * and the cutoff marked.
 */
export function Timeline({
  transactions,
  value,
  onChange,
  playing,
  onPlayingChange,
  deadline,
  cutoff,
}: Props) {
  const last = transactions.length - 1

  // Advancing from `value` on a timeout (rather than a self-ticking interval)
  // keeps the parent as the single source of truth for where the replay is.
  useEffect(() => {
    if (!playing) return
    if (value >= last) {
      onPlayingChange(false)
      return
    }
    const handle = window.setTimeout(() => onChange(value + 1), STEP_MS)
    return () => window.clearTimeout(handle)
  }, [playing, value, last, onChange, onPlayingChange])

  if (transactions.length === 0) return null

  const start = toTime(transactions[0].timestamp)
  const end = Math.max(toTime(transactions[last].timestamp), toTime(deadline))
  const span = Math.max(end - start, 1)
  const position = (stamp: string) => ((toTime(stamp) - start) / span) * 100

  const current = transactions[Math.min(value, last)]
  const cutoffPosition = position(cutoff)

  return (
    <div className="rounded-xl border border-grid bg-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-ink">Time replay</h3>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              if (value >= last) onChange(0)
              onPlayingChange(!playing)
            }}
            className="inline-flex items-center gap-1.5 rounded-lg border border-grid px-2.5 py-1.5 text-sm font-medium text-ink hover:bg-grid/60"
          >
            {playing ? (
              <Pause className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => {
              onPlayingChange(false)
              onChange(last)
            }}
            className="inline-flex items-center gap-1.5 rounded-lg border border-grid px-2.5 py-1.5 text-sm text-ink2 hover:text-ink"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Show all
          </button>
        </div>
      </div>

      <label className="mt-3 block">
        <span className="sr-only">Transactions revealed</span>
        <input
          type="range"
          min={0}
          max={last}
          value={Math.min(value, last)}
          onChange={(event) => {
            onPlayingChange(false)
            onChange(Number(event.target.value))
          }}
          className="w-full accent-[#2a78d6]"
          aria-valuetext={`Transaction ${value + 1} of ${transactions.length}, ${formatStamp(
            current.timestamp,
          )}`}
        />
      </label>

      <div className="mt-1 flex items-baseline justify-between text-xs">
        <span className="text-ink2">
          Showing <strong className="text-ink">{Math.min(value, last) + 1}</strong> of{' '}
          {transactions.length} transactions
        </span>
        <span className="tabular-nums text-muted">{formatStamp(current.timestamp)}</span>
      </div>

      {/* time axis: one tick per transaction, plus deadline and cutoff */}
      <div className="relative mt-3 h-8">
        <div className="absolute inset-x-0 top-3 h-px bg-axis" />
        {transactions.map((txn, index) => (
          <span
            key={txn.txn_index}
            className="absolute top-1.5 h-2 w-0.5 rounded"
            style={{
              left: `${position(txn.timestamp)}%`,
              background: index <= value ? 'var(--color-ink)' : 'var(--color-axis)',
            }}
            aria-hidden="true"
          />
        ))}
        <span
          className="absolute top-0 flex flex-col items-center"
          style={{ left: `${position(deadline)}%` }}
        >
          <span className="h-5 w-0.5 bg-ink" aria-hidden="true" />
          <span className="whitespace-nowrap text-[10px] text-ink2">deadline</span>
        </span>
        {cutoffPosition >= 0 && cutoffPosition <= 100 && (
          <span
            className="absolute top-0 flex flex-col items-center"
            style={{ left: `${cutoffPosition}%` }}
          >
            <span
              className="h-5 w-0.5 bg-warning"
              aria-hidden="true"
              style={{ borderLeft: '1px dashed' }}
            />
            <span className="whitespace-nowrap text-[10px] text-ink2">cutoff</span>
          </span>
        )}
      </div>
    </div>
  )
}
