import type { ReactNode } from 'react'
import { AlertTriangle, Inbox, Loader2, ServerCog } from 'lucide-react'

import { ApiError } from '../api/client'

export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-md bg-grid ${className}`}
      aria-hidden="true"
    />
  )
}

export function LoadingBlock({ label = 'Loading' }: { label?: string }) {
  return (
    <div
      className="flex items-center gap-2 p-6 text-sm text-muted"
      role="status"
      aria-live="polite"
    >
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      {label}...
    </div>
  )
}

interface EmptyStateProps {
  title: string
  description?: ReactNode
  icon?: ReactNode
  action?: ReactNode
}

/** Shown when there is genuinely nothing to display yet. Never a stand-in
 * for invented data. */
export function EmptyState({ title, description, icon, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-axis bg-surface px-6 py-12 text-center">
      <div className="text-muted" aria-hidden="true">
        {icon ?? <Inbox className="h-8 w-8" />}
      </div>
      <h3 className="text-base font-semibold text-ink">{title}</h3>
      {description && (
        <p className="max-w-md text-sm leading-relaxed text-ink2">{description}</p>
      )}
      {action}
    </div>
  )
}

/** Turns an API failure into something a human can act on. */
export function ErrorState({ error }: { error: unknown }) {
  const apiError = error instanceof ApiError ? error : undefined

  if (apiError?.isNotExported) {
    return (
      <EmptyState
        icon={<ServerCog className="h-8 w-8" />}
        title="The data has not been exported yet"
        description={
          <>
            {apiError.detail} Until then, this page has nothing real to show — and
            it will not invent anything.
          </>
        }
      />
    )
  }

  const message =
    apiError?.detail ??
    (error instanceof Error ? error.message : 'Something went wrong.')

  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-xl border border-critical/40 bg-surface p-4 text-sm"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-critical" aria-hidden="true" />
      <div>
        <p className="font-semibold text-ink">Could not load this view</p>
        <p className="mt-1 text-ink2">{message}</p>
      </div>
    </div>
  )
}
