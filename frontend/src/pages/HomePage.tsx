import { Link } from 'react-router-dom'
import { ArrowRight, CheckCircle2, Circle } from 'lucide-react'

import { useHealth, useSummary } from '../api/hooks'
import { FourStepStory } from '../components/FourStepStory'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { StatCard } from '../components/cards'
import { ErrorState, Skeleton } from '../components/states'

const TOUR = [
  { to: '/dataset', label: 'See the dataset' },
  { to: '/campaigns', label: 'Browse campaigns' },
  { to: '/visibility', label: 'Try the Visibility Lab' },
]

/** What exists today vs what later phases will add. Driven by /api/health,
 * so it reports the real state of the artifacts rather than a hardcoded list. */
function PipelineStatus() {
  const { data, isLoading, isError } = useHealth()

  const stages = [
    {
      label: 'Dataset processed and campaigns extracted',
      done: Boolean(data?.artifacts['campaigns.parquet']),
    },
    {
      label: 'Visibility scenarios generated',
      done: Boolean(data?.artifacts['visibility/seed_0.parquet']),
    },
    {
      label: 'Experiment results (Phase 5)',
      done: Boolean(data?.results_available),
    },
  ]

  return (
    <div className="rounded-xl border border-grid bg-surface p-4">
      <h2 className="text-sm font-semibold text-ink">Pipeline status</h2>
      {isLoading && <Skeleton className="mt-3 h-20 w-full" />}
      {isError && (
        <p className="mt-2 text-sm text-ink2">
          The API is not reachable, so the pipeline status is unknown.
        </p>
      )}
      {data && (
        <ul className="mt-3 space-y-2">
          {stages.map((stage) => (
            <li key={stage.label} className="flex items-start gap-2 text-sm">
              {stage.done ? (
                <CheckCircle2
                  className="mt-0.5 h-4 w-4 shrink-0 text-good"
                  aria-hidden="true"
                />
              ) : (
                <Circle className="mt-0.5 h-4 w-4 shrink-0 text-muted" aria-hidden="true" />
              )}
              <span className={stage.done ? 'text-ink' : 'text-muted'}>
                {stage.label}
                <span className="sr-only">{stage.done ? ' (done)' : ' (not yet)'}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function KeyStats() {
  const { data, isLoading, error } = useSummary()

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    )
  }
  if (error) return <ErrorState error={error} />
  if (!data) return null

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <StatCard
        label="Transactions"
        value={data.rows.total.toLocaleString()}
        hint={`${data.rows.pre_cutoff.toLocaleString()} usable before the cutoff`}
      />
      <StatCard
        label="Campaigns"
        value={data.campaigns.total.toLocaleString()}
        hint={`${data.campaigns.eval_ok} large enough to evaluate`}
      />
      <StatCard
        label="Institutions"
        value={data.institutions.k}
        hint="Simulated banks sharing the dataset"
      />
      <StatCard
        label="Laundering transactions"
        value={data.laundering.rows.toLocaleString()}
        hint={`${data.laundering.unassigned.toLocaleString()} belong to no known campaign`}
      />
    </div>
  )
}

export function HomePage() {
  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <header>
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">
          Start here
        </p>
        <h1 className="mt-1 text-2xl font-semibold leading-tight text-ink sm:text-3xl">
          When banks each see only a piece of a fraud campaign, how much does
          working together actually help?
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-relaxed text-ink2">
          Money laundering is deliberately spread across institutions, so no single
          bank ever sees the whole scheme. This project measures how the benefit of
          collaboration changes as each bank&rsquo;s{' '}
          <GlossaryTerm term="visibility">visibility</GlossaryTerm> of a{' '}
          <GlossaryTerm term="campaign">campaign</GlossaryTerm> shrinks &mdash; and
          whether that depends on the campaign&rsquo;s shape, since{' '}
          <GlossaryTerm term="hub pattern">hub-shaped</GlossaryTerm> schemes cannot
          be hidden the same way{' '}
          <GlossaryTerm term="fragmentable">fragmentable</GlossaryTerm> ones can.
        </p>
      </header>

      <section aria-labelledby="story">
        <h2 id="story" className="sr-only">
          How the problem works
        </h2>
        <FourStepStory />
      </section>

      <section aria-labelledby="stats" className="space-y-3">
        <h2 id="stats" className="text-sm font-semibold text-ink">
          The data behind it
        </h2>
        <KeyStats />
      </section>

      <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
        <section aria-labelledby="tour" className="rounded-xl border border-grid bg-surface p-4">
          <h2 id="tour" className="text-sm font-semibold text-ink">
            Take the tour
          </h2>
          <p className="mt-1 text-sm text-ink2">
            Three stops, about two minutes.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {TOUR.map((stop, index) => (
              <Link
                key={stop.to}
                to={stop.to}
                className="inline-flex items-center gap-2 rounded-lg border border-grid px-3 py-2 text-sm font-medium text-ink hover:bg-grid/60"
              >
                <span className="text-muted">{index + 1}</span>
                {stop.label}
                <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
              </Link>
            ))}
          </div>
        </section>

        <PipelineStatus />
      </div>
    </div>
  )
}
