import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useVisibilityDistribution } from '../api/hooks'
import { ChartCard, StatCard } from '../components/cards'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { VisibilityToy } from '../components/VisibilityToy'
import { ErrorState, LoadingBlock } from '../components/states'
import {
  BIN_LABELS,
  BIN_ORDER,
  CHROME,
  GROUP_LABELS,
  binColor,
  institutionColor,
} from '../theme'
import { useThemeMode } from '../lib/themeContext'
import type { TargetDistribution } from '../api/types'

const SEEDS = [0, 1, 2, 3, 4]

function axisProps(mode: 'light' | 'dark') {
  return {
    stroke: CHROME[mode].axis,
    tick: { fill: CHROME[mode].muted, fontSize: 11 },
    tickLine: false,
  }
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

/** Achieved visibility, histogrammed in 10-point buckets, for one target. */
function AchievedHistogram({ entry }: { entry: TargetDistribution }) {
  const { mode } = useThemeMode()
  const treated = entry.campaigns.filter((campaign) => campaign.reassigned)

  const buckets = Array.from({ length: 10 }, (_, i) => ({
    label: `${i * 10}-${i * 10 + 10}%`,
    mid: i * 10 + 5,
    count: 0,
  }))
  for (const campaign of treated) {
    const index = Math.min(9, Math.floor(campaign.achieved_visibility * 10))
    buckets[index].count += 1
  }

  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={buckets} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={CHROME[mode].grid} vertical={false} />
          <XAxis
            dataKey="label"
            {...axisProps(mode)}
            interval={1}
            tickFormatter={(label: string) => label.split('-')[0]}
          />
          <YAxis {...axisProps(mode)} allowDecimals={false} />
          <Tooltip
            {...tooltipStyle(mode)}
            formatter={(value: unknown) => [String(value), 'Campaigns']}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={32}>
            {buckets.map((bucket) => (
              <Cell
                key={bucket.label}
                fill={binColor(
                  bucket.mid >= 90
                    ? '100'
                    : bucket.mid >= 65
                      ? '75'
                      : bucket.mid >= 45
                        ? '50'
                        : 'low',
                  mode,
                )}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function BinsByGroup({ entry }: { entry: TargetDistribution }) {
  const { mode } = useThemeMode()
  const groups = Object.keys(entry.bin_counts_by_group).sort()
  const data = BIN_ORDER.map((bin) => {
    const row: Record<string, string | number> = { bin: BIN_LABELS[bin].split(' ')[0] }
    for (const group of groups) {
      row[group] = entry.bin_counts_by_group[group]?.[bin] ?? 0
    }
    return row
  })

  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke={CHROME[mode].grid} vertical={false} />
          <XAxis dataKey="bin" {...axisProps(mode)} />
          <YAxis {...axisProps(mode)} allowDecimals={false} />
          <Tooltip {...tooltipStyle(mode)} />
          {groups.map((group, index) => (
            <Bar
              key={group}
              dataKey={group}
              name={GROUP_LABELS[group] ?? group}
              fill={institutionColor(index === 0 ? 0 : index === 1 ? 3 : 6, mode)}
              radius={[4, 4, 0, 0]}
              maxBarSize={28}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function VisibilityLabPage() {
  const [seed, setSeed] = useState(0)
  const [targetIndex, setTargetIndex] = useState(3)
  const { data, isLoading, error } = useVisibilityDistribution(seed)

  const entry = data?.per_target[targetIndex]
  const treatedCount = entry?.campaigns.filter((c) => c.reassigned).length ?? 0

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Visibility Lab</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink2">
          <GlossaryTerm term="visibility">Visibility</GlossaryTerm> is the share of a
          campaign that the single best-placed bank can see. It is the dial this whole
          study turns. Build a scenario yourself below, then see what the real data
          does.
        </p>
      </header>

      <VisibilityToy />

      <section aria-labelledby="real" className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="real" className="text-sm font-semibold text-ink">
              What actually happens in the dataset
            </h2>
            <p className="mt-1 max-w-2xl text-xs leading-snug text-ink2">
              Every number below comes from the research pipeline, not from this page.
            </p>
          </div>
          <div className="flex items-end gap-3">
            <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
              Seed
              <select
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
                className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
              >
                {SEEDS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
              Target
              <select
                value={targetIndex}
                onChange={(event) => setTargetIndex(Number(event.target.value))}
                className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
              >
                {(data?.targets ?? [1, 0.75, 0.5, 0.25]).map((target, index) => (
                  <option key={target} value={index}>
                    {Math.round(target * 100)}%
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {isLoading && <LoadingBlock label="Loading visibility distributions" />}
        {error && <ErrorState error={error} />}

        {data && entry && (
          <>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard
                label="Target"
                value={`${Math.round(entry.target * 100)}%`}
                hint="What the search aimed for"
              />
              <StatCard
                label="Campaigns fragmented"
                value={treatedCount}
                hint="Only these have their accounts moved"
              />
              <StatCard
                label="Landed in bin 100"
                value={entry.bin_counts['100'] ?? 0}
                hint="Still fully visible to one bank"
              />
              <StatCard
                label="Unfragmentable"
                value={data.unfragmentable_campaigns}
                hint="Could never be pushed below 90%"
              />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <ChartCard
                title={`Achieved visibility at target ${Math.round(entry.target * 100)}%`}
                howToRead={
                  <>
                    Each bar counts the fragmented campaigns whose achieved visibility
                    landed in that range. The search often misses its target &mdash;
                    which is exactly why the study analyses results by achieved bin
                    rather than by the target it asked for.
                  </>
                }
              >
                <AchievedHistogram entry={entry} />
              </ChartCard>

              <ChartCard
                title="Which bin each group lands in"
                howToRead={
                  <>
                    <GlossaryTerm term="hub pattern">Hub</GlossaryTerm> and{' '}
                    <GlossaryTerm term="unfragmentable">unfragmentable</GlossaryTerm>{' '}
                    campaigns sit in the 100 bin whatever the target, because their
                    shape leaves one bank seeing everything. That contrast is the
                    control the study leans on.
                  </>
                }
              >
                <BinsByGroup entry={entry} />
              </ChartCard>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
