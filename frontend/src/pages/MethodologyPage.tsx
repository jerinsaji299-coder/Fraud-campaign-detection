import { useMethodology } from '../api/hooks'
import { ErrorState, LoadingBlock } from '../components/states'
import type { MethodologyEntry } from '../api/types'

/** Renders a definition's machine-readable values as a compact list. */
function ValueList({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value)
  if (entries.length === 0) return null

  return (
    <dl className="mt-3 grid gap-x-4 gap-y-1.5 border-t border-grid pt-3 sm:grid-cols-2">
      {entries.map(([key, raw]) => (
        <div key={key} className="flex flex-wrap items-baseline gap-x-2">
          <dt className="text-xs font-medium uppercase tracking-wide text-muted">
            {key.replace(/_/g, ' ')}
          </dt>
          <dd className="font-mono text-xs tabular-nums text-ink">
            {Array.isArray(raw)
              ? raw.join(', ')
              : typeof raw === 'object' && raw !== null
                ? JSON.stringify(raw)
                : String(raw)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function EntryCard({ entry }: { entry: MethodologyEntry }) {
  return (
    <article className="rounded-xl border border-grid bg-surface p-4">
      <h2 className="text-sm font-semibold text-ink">{entry.title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-ink2">{entry.definition}</p>
      <div className="mt-3 rounded-lg border-l-2 border-axis bg-plane p-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">
          Why it is defined this way
        </p>
        <p className="mt-1 text-sm leading-relaxed text-ink2">{entry.reason}</p>
      </div>
      <ValueList value={entry.value} />
    </article>
  )
}

export function MethodologyPage() {
  const { data, isLoading, error } = useMethodology()

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Methodology</h1>
        <p className="mt-1 text-sm leading-relaxed text-ink2">
          The frozen definitions this study runs on, each with the reason it was
          chosen. Every value on this page is read from the same constants module the
          research pipeline uses, so what you see here is what the code actually did
          &mdash; the two cannot drift apart.
        </p>
      </header>

      {isLoading && <LoadingBlock label="Loading methodology" />}
      {error && <ErrorState error={error} />}

      {data && (
        <>
          <p className="text-xs text-muted">{data.n_entries} definitions</p>
          <div className="space-y-4">
            {data.entries.map((entry) => (
              <EntryCard key={entry.key} entry={entry} />
            ))}
          </div>
          <p className="border-t border-grid pt-4 text-xs leading-relaxed text-muted">
            This is a research prototype on a synthetic dataset. It is not a
            production detection system, and none of these definitions should be read
            as a recommendation for real-world compliance practice.
          </p>
        </>
      )}
    </div>
  )
}
