import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useCampaigns, useSummary } from '../api/hooks'
import { ChartCard, StatCard } from '../components/cards'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { ErrorState, LoadingBlock, Skeleton } from '../components/states'
import { CHROME, GROUP_LABELS, SPLIT_LABELS, institutionColor } from '../theme'
import { useThemeMode } from '../lib/themeContext'
import type { Summary } from '../api/types'

const CUTOFF_DAY = '2022-09-11'

function axisProps(mode: 'light' | 'dark') {
  return {
    stroke: CHROME[mode].axis,
    tick: { fill: CHROME[mode].muted, fontSize: 11 },
    tickLine: false,
  }
}

/** Recharts types the formatter value as unknown-ish, so coerce once here. */
function countFormatter(label: string) {
  return (value: unknown): [string, string] => [Number(value).toLocaleString(), label]
}

function tooltipStyle(mode: 'light' | 'dark') {
  return {
    contentStyle: {
      background: CHROME[mode].surface,
      border: `1px solid ${CHROME[mode].grid}`,
      borderRadius: 8,
      fontSize: 12,
      color: CHROME[mode].ink,
    },
    labelStyle: { color: CHROME[mode].ink },
    cursor: { fill: CHROME[mode].grid, fillOpacity: 0.4 },
  }
}

function TransactionsPerDay({ summary }: { summary: Summary }) {
  const { mode } = useThemeMode()
  const data = Object.entries(summary.transactions_per_day).map(([day, count]) => ({
    day,
    count,
    excluded: day >= CUTOFF_DAY,
  }))
  const excludedDays = data.filter((d) => d.excluded)

  return (
    <ChartCard
      title="Transactions per day"
      howToRead={
        <>
          Each bar is one day of activity. Everything from the dashed{' '}
          <GlossaryTerm term="cutoff">cutoff</GlossaryTerm> line onward (shaded) is
          excluded from the study &mdash; that tail is tiny but overwhelmingly
          laundering, so a model trained on it could cheat.
        </>
      }
    >
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
            <CartesianGrid stroke={CHROME[mode].grid} vertical={false} />
            <XAxis dataKey="day" {...axisProps(mode)} tickFormatter={(d: string) => d.slice(5)} />
            <YAxis
              {...axisProps(mode)}
              tickFormatter={(v: number) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`)}
            />
            <Tooltip
              {...tooltipStyle(mode)}
              formatter={countFormatter('Transactions')}
            />
            {excludedDays.length > 0 && (
              <ReferenceArea
                x1={excludedDays[0].day}
                x2={excludedDays[excludedDays.length - 1].day}
                fill={CHROME[mode].muted}
                fillOpacity={0.12}
              />
            )}
            <ReferenceLine
              x={CUTOFF_DAY}
              stroke={CHROME[mode].ink}
              strokeDasharray="4 3"
              label={{
                value: 'Cutoff',
                position: 'insideTopRight',
                fill: CHROME[mode].ink,
                fontSize: 11,
              }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={28}>
              {data.map((entry) => (
                <Cell
                  key={entry.day}
                  fill={institutionColor(0, mode)}
                  fillOpacity={entry.excluded ? 0.3 : 1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  )
}

function CampaignsPerType({ summary }: { summary: Summary }) {
  const { mode } = useThemeMode()
  const hubTypes = new Set(['FAN-IN', 'FAN-OUT', 'GATHER-SCATTER'])
  const data = Object.entries(summary.by_base_type)
    .map(([type, count]) => ({ type, count, hub: hubTypes.has(type) }))
    .sort((a, b) => b.count - a.count)

  return (
    <ChartCard
      title="Campaigns by pattern type"
      howToRead={
        <>
          Pattern shapes from the dataset&rsquo;s ground truth.{' '}
          <GlossaryTerm term="hub pattern">Hub shapes</GlossaryTerm> (marked
          &ldquo;hub&rdquo;) always stay fully visible to one bank, so they act as a
          control group rather than a treatment group.
        </>
      }
    >
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 36, bottom: 4, left: 8 }}
          >
            <CartesianGrid stroke={CHROME[mode].grid} horizontal={false} />
            <XAxis type="number" {...axisProps(mode)} />
            <YAxis
              type="category"
              dataKey="type"
              width={110}
              {...axisProps(mode)}
              tickFormatter={(t: string) => (hubTypes.has(t) ? `${t} (hub)` : t)}
            />
            <Tooltip
              {...tooltipStyle(mode)}
              formatter={countFormatter('Campaigns')}
            />
            <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={20}>
              {data.map((entry) => (
                <Cell
                  key={entry.type}
                  fill={institutionColor(entry.hub ? 3 : 0, mode)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  )
}

/** Histogram of campaign durations. This is a display binning of values the
 * API already returned, not a computed research metric. */
function DurationDistribution() {
  const { mode } = useThemeMode()
  const { data, isLoading, error } = useCampaigns({ page_size: 500 })

  if (isLoading) return <Skeleton className="h-80" />
  if (error) return <ErrorState error={error} />
  if (!data) return null

  const edges = [0, 12, 24, 48, 72, 96, 120, 168, Number.POSITIVE_INFINITY]
  const labels = ['0-12h', '12-24h', '1-2d', '2-3d', '3-4d', '4-5d', '5-7d', '7d+']
  const counts = labels.map(() => 0)
  for (const campaign of data.items) {
    const index = edges.findIndex(
      (edge, i) => campaign.duration_h >= edge && campaign.duration_h < edges[i + 1],
    )
    if (index >= 0) counts[index] += 1
  }
  const chartData = labels.map((label, i) => ({ label, count: counts[i] }))

  return (
    <ChartCard
      title="How long campaigns run"
      howToRead="Most campaigns stretch over days, not minutes. That is what makes early discovery meaningful: there is a window in which a scheme is still unfolding."
    >
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
            <CartesianGrid stroke={CHROME[mode].grid} vertical={false} />
            <XAxis dataKey="label" {...axisProps(mode)} />
            <YAxis {...axisProps(mode)} />
            <Tooltip
              {...tooltipStyle(mode)}
              formatter={countFormatter('Campaigns')}
            />
            <Bar
              dataKey="count"
              fill={institutionColor(0, mode)}
              radius={[4, 4, 0, 0]}
              maxBarSize={48}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  )
}

function SplitBreakdown({ summary }: { summary: Summary }) {
  const order = ['train', 'val', 'test', 'stress']
  return (
    <div className="rounded-xl border border-grid bg-surface p-4">
      <h3 className="text-sm font-semibold text-ink">Splits</h3>
      <p className="mt-1 text-xs text-ink2">
        Campaigns are split by the day they start.{' '}
        <GlossaryTerm term="censored">Censored</GlossaryTerm> campaigns had not
        finished by the cutoff.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">
            Campaign counts per split, with evaluable, censored and shared-account
            counts
          </caption>
          <thead>
            <tr className="border-b border-grid text-left text-xs uppercase tracking-wide text-muted">
              <th scope="col" className="py-2 pr-3 font-medium">Split</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">All</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Evaluable</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Censored</th>
              <th scope="col" className="py-2 text-right font-medium">Shared acct.</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {order.map((split) => {
              const row = summary.by_split[split]
              if (!row) return null
              return (
                <tr key={split} className="border-b border-grid/60 last:border-0">
                  <th scope="row" className="py-2 pr-3 text-left font-medium text-ink">
                    {SPLIT_LABELS[split] ?? split}
                  </th>
                  <td className="py-2 pr-3 text-right text-ink2">{row.campaigns}</td>
                  <td className="py-2 pr-3 text-right text-ink">{row.eval_ok}</td>
                  <td className="py-2 pr-3 text-right text-ink2">{row.crosses_cutoff}</td>
                  <td className="py-2 text-right text-ink2">{row.shares_train_account}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function GroupBreakdown({ summary }: { summary: Summary }) {
  const { mode } = useThemeMode()
  const total = Object.values(summary.by_group_eval_ok).reduce((a, b) => a + b, 0)

  return (
    <div className="rounded-xl border border-grid bg-surface p-4">
      <h3 className="text-sm font-semibold text-ink">
        Which campaigns can actually be hidden
      </h3>
      <p className="mt-1 text-xs text-ink2">
        Of the {total} evaluable campaigns, only the{' '}
        <GlossaryTerm term="fragmentable">fragmentable</GlossaryTerm> ones can have
        their visibility reduced. The rest form the control group.
      </p>
      <ul className="mt-3 space-y-2">
        {Object.entries(summary.by_group_eval_ok).map(([group, count]) => {
          const share = total ? Math.round((count / total) * 100) : 0
          const isTreatment = group === 'fragmentable'
          return (
            <li key={group}>
              <div className="flex items-baseline justify-between text-sm">
                <span className="font-medium text-ink">
                  {GROUP_LABELS[group] ?? group}
                  {!isTreatment && (
                    <span className="ml-1.5 text-xs font-normal text-muted">
                      (control)
                    </span>
                  )}
                </span>
                <span className="tabular-nums text-ink2">
                  {count} &middot; {share}%
                </span>
              </div>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-grid">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${share}%`,
                    background: institutionColor(isTreatment ? 0 : 3, mode),
                  }}
                />
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function CutoffExplainer({ summary }: { summary: Summary }) {
  const excludedShare = Math.round(summary.cutoff.laundering_share_excluded * 100)
  const overallShare = summary.cutoff.laundering_share_overall * 100

  return (
    <div className="rounded-xl border border-axis bg-surface p-4">
      <h3 className="text-sm font-semibold text-ink">
        Why we excluded everything after Sept 10
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-ink2">
        The file runs to Sept 18, but activity collapses after Sept 10. Only{' '}
        <strong className="text-ink">
          {summary.cutoff.rows_excluded.toLocaleString()}
        </strong>{' '}
        transactions remain in that tail &mdash; and{' '}
        <strong className="text-ink">{excludedShare}%</strong> of them are
        laundering, against{' '}
        <strong className="text-ink">{overallShare.toFixed(2)}%</strong> across the
        dataset as a whole. A model trained on that tail could score well by
        learning &ldquo;late in the file means laundering&rdquo; instead of learning
        anything about behaviour, so the study stops at the{' '}
        <GlossaryTerm term="cutoff">cutoff</GlossaryTerm>. The excluded data is
        still used for one thing only: knowing when a campaign truly finished.
      </p>
    </div>
  )
}

export function DatasetPage() {
  const { data, isLoading, error } = useSummary()

  if (isLoading) return <LoadingBlock label="Loading dataset summary" />
  if (error) return <ErrorState error={error} />
  if (!data) return null

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-ink">The dataset</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink2">
          IBM&rsquo;s synthetic anti-money-laundering transaction set (HI-Small),
          which ships with a ground-truth file naming every transaction that belongs
          to a laundering scheme. Covering{' '}
          {data.date_range.min.slice(0, 10)} to {data.date_range.max.slice(0, 10)}.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Transactions" value={data.rows.total.toLocaleString()} />
        <StatCard label="Banks" value={data.banks.total.toLocaleString()} />
        <StatCard
          label="Campaigns"
          value={data.campaigns.total}
          hint={`${data.campaigns.pattern_transactions.toLocaleString()} transactions`}
        />
        <StatCard
          label="Evaluable campaigns"
          value={data.campaigns.eval_ok}
          hint={`${data.campaigns.reassignable} can be fragmented`}
        />
      </div>

      <TransactionsPerDay summary={data} />

      <div className="grid gap-4 lg:grid-cols-2">
        <CampaignsPerType summary={data} />
        <DurationDistribution />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <SplitBreakdown summary={data} />
        <GroupBreakdown summary={data} />
      </div>

      <CutoffExplainer summary={data} />
    </div>
  )
}
