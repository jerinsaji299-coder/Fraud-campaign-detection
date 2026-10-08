import type { ReactNode } from 'react'

import { institutionColor, institutionLabel } from '../theme'
import { useThemeMode } from '../lib/themeContext'

interface StatCardProps {
  label: ReactNode
  value: ReactNode
  hint?: ReactNode
  emphasis?: boolean
}

export function StatCard({ label, value, hint, emphasis }: StatCardProps) {
  return (
    <div
      className={`rounded-xl border bg-surface p-4 ${
        emphasis ? 'border-axis' : 'border-grid'
      }`}
    >
      <div className="text-xs font-medium uppercase tracking-wide text-muted">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
      {hint && <div className="mt-1 text-xs leading-snug text-ink2">{hint}</div>}
    </div>
  )
}

interface ChartCardProps {
  title: ReactNode
  howToRead?: ReactNode
  children: ReactNode
  actions?: ReactNode
}

export function ChartCard({ title, howToRead, children, actions }: ChartCardProps) {
  return (
    <figure className="m-0 rounded-xl border border-grid bg-surface p-4">
      <figcaption className="mb-3 flex items-start justify-between gap-3">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        {actions}
      </figcaption>
      {children}
      {howToRead && (
        <p className="mt-3 border-t border-grid pt-2 text-xs leading-relaxed text-muted">
          <span className="font-semibold">How to read this: </span>
          {howToRead}
        </p>
      )}
    </figure>
  )
}

/** Institution identity: always the label, with color as a secondary cue. */
export function InstitutionBadge({
  institution,
  className = '',
}: {
  institution: number
  className?: string
}) {
  const { mode } = useThemeMode()
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border border-grid px-1.5 py-0.5 text-xs font-medium text-ink ${className}`}
    >
      <span
        aria-hidden="true"
        className="h-2.5 w-2.5 rounded-sm"
        style={{ background: institutionColor(institution, mode) }}
      />
      <span>{institutionLabel(institution)}</span>
    </span>
  )
}

export function Pill({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'neutral' | 'good' | 'warning'
}) {
  const tones = {
    neutral: 'border-grid text-ink2',
    good: 'border-good/40 text-good',
    warning: 'border-warning/50 text-ink2',
  }
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  )
}
