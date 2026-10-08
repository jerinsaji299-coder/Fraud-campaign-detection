import { BarChart3, LineChart as LineChartIcon, TrendingUp } from 'lucide-react'

import { useResultsSummary } from '../api/hooks'
import { ConditionLegend } from '../components/ConditionLegend'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { EmptyState, LoadingBlock } from '../components/states'
import { ApiError } from '../api/client'

/**
 * Until Phase 5 produces real results, this page shows labelled *outlines* of
 * the figures that are planned. The outlines contain no numbers, invented or
 * otherwise - they exist so the panel can see what the study will report.
 */

interface PlannedFigure {
  title: string
  icon: typeof BarChart3
  xLabel: string
  yLabel: string
  howToRead: React.ReactNode
}

const PLANNED: PlannedFigure[] = [
  {
    title: 'Campaign recall vs visibility bin, per condition',
    icon: BarChart3,
    xLabel: 'Achieved visibility bin (100 → low)',
    yLabel: 'Share of campaigns discovered',
    howToRead: (
      <>
        The headline figure. Four lines, one per condition, with confidence
        intervals. If collaboration helps most when banks see least, the gap between
        the isolated line and the federated lines should widen towards the right.
      </>
    ),
  },
  {
    title: 'Median lead time vs visibility bin',
    icon: LineChartIcon,
    xLabel: 'Achieved visibility bin (100 → low)',
    yLabel: 'Hours before true completion',
    howToRead: (
      <>
        Discovering a campaign at all is not the same as discovering it early. This
        asks how much <GlossaryTerm term="lead time">lead time</GlossaryTerm> each
        condition buys, and whether that advantage survives as fragments shrink.
      </>
    ),
  },
  {
    title: 'Knowledge advantage and evidence advantage',
    icon: TrendingUp,
    xLabel: 'Achieved visibility bin (100 → low)',
    yLabel: 'Improvement over the isolated baseline',
    howToRead: (
      <>
        Two derived curves: sharing model weights (
        <GlossaryTerm term="fedavg">FedAvg</GlossaryTerm>) minus isolated is the
        knowledge advantage; adding{' '}
        <GlossaryTerm term="embedding exchange">embedding exchange</GlossaryTerm> on
        top of that is the evidence advantage. Separating the two is this
        project&rsquo;s main contribution.
      </>
    ),
  },
  {
    title: 'The same comparison on the control group',
    icon: BarChart3,
    xLabel: 'Achieved visibility bin (100 → low)',
    yLabel: 'Share of campaigns discovered',
    howToRead: (
      <>
        <GlossaryTerm term="hub pattern">Hub</GlossaryTerm> and{' '}
        <GlossaryTerm term="unfragmentable">unfragmentable</GlossaryTerm> campaigns
        stay fully visible however the accounts are split, so any apparent effect
        here would be a sign that something other than visibility is driving the
        result.
      </>
    ),
  },
  {
    title: 'False alarms per condition',
    icon: BarChart3,
    xLabel: 'Condition',
    yLabel: 'False alarms',
    howToRead: (
      <>
        A detector that flags everything would score well on recall. This figure is
        the honesty check that stops that from looking like success.
      </>
    ),
  },
]

function FigureOutline({ figure }: { figure: PlannedFigure }) {
  const Icon = figure.icon
  return (
    <figure className="m-0 rounded-xl border border-grid bg-surface p-4">
      <figcaption className="mb-3 flex items-start gap-2">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted" aria-hidden="true" />
        <h3 className="text-sm font-semibold text-ink">{figure.title}</h3>
      </figcaption>

      {/* axis frame only: deliberately empty of data */}
      <div className="relative h-40 rounded-lg border border-dashed border-axis">
        <span className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 whitespace-nowrap text-[10px] text-muted">
          {figure.yLabel}
        </span>
        <span className="absolute inset-x-0 bottom-1 text-center text-[10px] text-muted">
          {figure.xLabel}
        </span>
        <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-muted">
          awaiting Phase 5
        </span>
      </div>

      <p className="mt-3 border-t border-grid pt-2 text-xs leading-relaxed text-muted">
        <span className="font-semibold">How to read this: </span>
        {figure.howToRead}
      </p>
    </figure>
  )
}

export function ResultsPage() {
  const { isLoading, error, data } = useResultsSummary()

  const notYet = error instanceof ApiError && error.isMissing

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Results</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink2">
          What the experiments will report, and how to read each figure. The
          experiments themselves are a later phase, so there are no numbers on this
          page yet &mdash; and none will be invented to fill the space.
        </p>
      </header>

      {isLoading && <LoadingBlock label="Checking for results" />}

      {notYet && (
        <EmptyState
          title="Experiment results are not available yet"
          description="They will appear here after Phase 5 runs the four conditions across every visibility bin and seed. Until then, the planned figures are outlined below."
        />
      )}

      {error && !notYet && (
        <EmptyState
          title="Could not check for results"
          description="The API is not reachable, so it is unknown whether results exist. The planned figures are outlined below."
        />
      )}

      {data && data.total > 0 && (
        <EmptyState
          title="Results exist but are not charted yet"
          description={`The API reports ${data.total} result rows. The charts that render them are built in a later phase, alongside the experiments that produce them.`}
        />
      )}

      <section aria-labelledby="conditions" className="rounded-xl border border-grid bg-surface p-4">
        <h2 id="conditions" className="text-sm font-semibold text-ink">
          The four conditions being compared
        </h2>
        <p className="mt-1 text-xs text-ink2">
          Each keeps a fixed colour and line style across every figure.
        </p>
        <div className="mt-3">
          <ConditionLegend />
        </div>
      </section>

      <section aria-labelledby="planned" className="space-y-4">
        <h2 id="planned" className="text-sm font-semibold text-ink">
          Planned figures
        </h2>
        <div className="grid gap-4 lg:grid-cols-2">
          {PLANNED.map((figure) => (
            <FigureOutline key={figure.title} figure={figure} />
          ))}
        </div>
      </section>

      <p className="text-xs leading-relaxed text-muted">
        Each figure will be filterable by topology group, by test subset (all, fully
        observed, or no shared accounts) and by metric, and downloadable as a PNG,
        once there is something real to show.
      </p>
    </div>
  )
}
