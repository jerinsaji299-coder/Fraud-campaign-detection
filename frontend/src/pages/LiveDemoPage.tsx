/**
 * The live demo panel, for a faculty presentation.
 *
 * Input -> output, computed on the spot. The viewer picks a detection window
 * (directly, or via the demo campaign which selects its own windows) and an
 * institution, then flips one toggle:
 *
 *   own      - that institution's own model, alone (stage 4.1)
 *   averaged - all K models applied to that same view, scores averaged (4.2)
 *
 * Every number on this page is computed by the backend from `fraudcamp`'s own
 * functions on each request. This component formats; it does not compute a
 * research metric, and when the bundle or the aggregate artifact is missing it
 * shows an empty state rather than placeholder numbers.
 */

import { useMemo, useState } from 'react'

import { ChartCard, InstitutionBadge, Pill, StatCard } from '../components/cards'
import { EmptyState, ErrorState, LoadingBlock } from '../components/states'
import { useDemoAggregate, useDemoManifest, useDemoRun } from '../api/hooks'
import { ApiError } from '../api/client'
import type { DemoCluster, DemoRun } from '../api/types'

const MODE_LABEL: Record<string, string> = {
  own: 'This institution alone',
  averaged: 'All 8 models, scores averaged',
}

const KIND_LABEL: Record<string, string> = {
  hit: 'matches a campaign',
  other_split_hit: 'matches a campaign from another split',
  ambiguous: 'touches unassigned laundering',
  false_alarm: 'false alarm',
}

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(0)}%`
}

function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

function LatencyNote({ run }: { run: DemoRun }) {
  const total = run.latency_ms?.total
  const slow = typeof total === 'number' && total > 5000
  return (
    <p className={`text-xs ${slow ? 'text-warning' : 'text-muted'}`}>
      Recomputed in <strong>{num(total, 0)} ms</strong>
      {typeof run.latency_ms?.predict === 'number' && (
        <>
          {' '}
          — featurise {num(run.latency_ms.featurise, 0)} ms
          {run.latency_ms.featurise_was_cached ? ' (cached)' : ''}, predict{' '}
          {num(run.latency_ms.predict, 0)} ms, clusters{' '}
          {num(run.latency_ms.clusters, 0)} ms
        </>
      )}
      {slow && ' — above the 5 s target for a live toggle.'}
    </p>
  )
}

function ClusterRow({ cluster }: { cluster: DemoCluster }) {
  const match = cluster.vs_demo_campaign
  return (
    <tr className="border-t border-grid">
      <td className="py-1 pr-3">
        {cluster.matches_demo_campaign ? (
          <Pill tone="good">demo campaign</Pill>
        ) : (
          <span className="text-muted">
            {KIND_LABEL[cluster.kind] ?? cluster.kind}
          </span>
        )}
      </td>
      <td className="py-1 pr-3 text-right tabular-nums">{cluster.n_accounts}</td>
      <td className="py-1 pr-3 text-right tabular-nums">{pct(match?.purity)}</td>
      <td className="py-1 pr-3 text-right tabular-nums">
        {match ? `${match.coverage_accounts}/${match.n_campaign_accounts}` : '—'}
      </td>
      <td className="py-1 pr-3">
        <span className="flex flex-wrap gap-1">
          {cluster.institutions
            .filter((i) => i >= 0)
            .map((i) => (
              <InstitutionBadge key={i} institution={i} />
            ))}
        </span>
      </td>
    </tr>
  )
}

function Output({ run }: { run: DemoRun }) {
  if (run.empty_reason) {
    return (
      <EmptyState
        title="Nothing to score"
        description={run.empty_reason}
      />
    )
  }
  const lead = run.lead_time
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <StatCard
          label="Flagged edges"
          value={run.n_flagged_edges.toLocaleString()}
          hint={`of ${run.n_edges_in_view.toLocaleString()} in this view`}
        />
        <StatCard
          label="Alerts raised"
          value={run.alert_count.toLocaleString()}
          hint="clusters an analyst would see"
        />
        <StatCard
          label="Demo campaign"
          value={run.demo_campaign_detected ? 'found' : 'missed'}
          emphasis={run.demo_campaign_detected}
          hint={`campaign ${run.demo_campaign_id ?? '—'}`}
        />
        <StatCard
          label="Best purity"
          value={pct(run.best_purity_vs_demo_campaign)}
          hint="of a cluster, against the demo campaign"
        />
        <StatCard
          label="Best coverage"
          value={pct(run.best_coverage_vs_demo_campaign)}
          hint="of the demo campaign's accounts"
        />
      </div>

      <div className="rounded-lg border border-grid p-3">
        <h4 className="mb-1 text-sm font-semibold">Lead time</h4>
        {lead?.detected ? (
          <p className="text-sm">
            Detected <strong>{num(lead.hours_before_deadline, 1)} h</strong>{' '}
            before the deadline
            {lead.hours_before_end !== null && (
              <> and {num(lead.hours_before_end, 1)} h before the campaign ends</>
            )}
            .{' '}
            {lead.beat_the_deadline === false && (
              <span className="text-critical">
                After the deadline, so it does not count as a detection.
              </span>
            )}
          </p>
        ) : (
          <p className="text-sm text-muted">
            Not detected at this detection time, so there is no lead time.
          </p>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">Clusters extracted at this setting</caption>
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-muted">
              <th className="pb-1 pr-3">Cluster</th>
              <th className="pb-1 pr-3 text-right">Accounts</th>
              <th className="pb-1 pr-3 text-right">Purity</th>
              <th className="pb-1 pr-3 text-right">Coverage</th>
              <th className="pb-1 pr-3">Institutions</th>
            </tr>
          </thead>
          <tbody>
            {run.clusters.length === 0 && (
              <tr>
                <td colSpan={5} className="py-2 text-muted">
                  No cluster passed the threshold: no alert at all here.
                </td>
              </tr>
            )}
            {run.clusters.map((cluster, index) => (
              <ClusterRow key={index} cluster={cluster} />
            ))}
          </tbody>
        </table>
      </div>
      <LatencyNote run={run} />
    </div>
  )
}

export function LiveDemoPage() {
  const manifest = useDemoManifest()
  const aggregate = useDemoAggregate()
  const [windowIndex, setWindowIndex] = useState(0)
  const [institution, setInstitution] = useState<number | null>(null)
  const [mode, setMode] = useState<'own' | 'averaged'>('own')
  const [point, setPoint] = useState('total_10')

  const detectionTime = manifest.data?.detection_times?.[windowIndex] ?? ''
  const chosenInstitution = useMemo(() => {
    if (institution !== null) return institution
    const holding = manifest.data?.demo_campaign?.institutions_holding ?? []
    const withModel = manifest.data?.institutions_with_a_model ?? []
    return holding.find((i) => withModel.includes(i)) ?? withModel[0] ?? 0
  }, [institution, manifest.data])

  const run = useDemoRun({
    detection_time: detectionTime,
    institution: chosenInstitution,
    mode,
    operating_point: point,
    enabled: Boolean(detectionTime) && manifest.isSuccess,
  })

  if (manifest.isLoading) return <LoadingBlock label="Loading the demo bundle" />
  if (manifest.error) {
    const error = manifest.error
    if (error instanceof ApiError && error.isMissing) {
      return (
        <EmptyState
          title="No live demo bundle installed"
          description={error.detail}
        />
      )
    }
    return <ErrorState error={error} />
  }

  const m = manifest.data!
  const windowMeta = m.windows?.[windowIndex]

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold">Live detection panel</h1>
        <p className="max-w-3xl text-sm text-ink2">
          One detection window, scored on the spot. The toggle switches between
          an institution scoring with <strong>its own model</strong> and the
          same view scored by <strong>all {m.k_institutions} models with their
          scores averaged</strong>. Clusters, matches, purity, coverage, lead
          time and the alert count are recomputed on every flip.
        </p>
        {m.development_only && (
          <p className="rounded-md border border-warning/50 p-2 text-xs text-ink2">
            {m.development_only}
          </p>
        )}
      </header>

      <ChartCard
        title="Chosen to illustrate the mechanism — not a representative result"
        howToRead="The campaign below was selected by a fixed rule, stated here in full. It shows what the mechanism does on one campaign. How often it holds is the aggregate table at the bottom of this page, and only that table speaks to typicality."
      >
        <div className="space-y-2 text-sm">
          <p>
            Campaign <strong>{m.demo_campaign.campaign_id}</strong>:{' '}
            {m.demo_campaign.accounts.length} accounts held across{' '}
            {m.demo_campaign.institutions_holding.length} institutions
            {m.demo_campaign.achieved_visibility !== null && (
              <> — best single-institution visibility {pct(m.demo_campaign.achieved_visibility)}</>
            )}
            .
          </p>
          <ol className="list-decimal space-y-1 pl-5 text-xs text-ink2">
            {m.selection_rule.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className="text-xs text-ink2">
            Chosen at rank {m.selection_audit.rank_chosen ?? '—'} of{' '}
            {m.selection_audit.n_candidates ?? '—'} candidates,{' '}
            {m.selection_audit.n_skipped} skipped as unreachable by the
            centralized oracle.
          </p>
          {m.selection_disclaimer && (
            <p className="text-xs font-semibold">{m.selection_disclaimer}</p>
          )}
        </div>
      </ChartCard>

      <ChartCard
        title="Input"
        howToRead="Pick a window and an institution, then flip the toggle. Everything below is recomputed by the backend on each change."
      >
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-sm">
            <span className="block text-xs font-semibold uppercase tracking-wide text-muted">
              Detection window
            </span>
            <select
              className="mt-1 rounded-md border border-grid bg-surface p-1"
              value={windowIndex}
              onChange={(event) => setWindowIndex(Number(event.target.value))}
            >
              {m.detection_times.map((t, index) => (
                <option key={t} value={index}>
                  {t}
                  {m.windows?.[index]
                    ? ` — ${m.windows[index].n_campaign_edges} campaign edge(s)`
                    : ''}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm">
            <span className="block text-xs font-semibold uppercase tracking-wide text-muted">
              Institution
            </span>
            <select
              className="mt-1 rounded-md border border-grid bg-surface p-1"
              value={chosenInstitution}
              onChange={(event) => setInstitution(Number(event.target.value))}
            >
              {m.institutions_with_a_model.map((i) => (
                <option key={i} value={i}>
                  I{i}
                  {m.demo_campaign.institutions_holding.includes(i)
                    ? ' — holds campaign accounts'
                    : ''}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm">
            <span className="block text-xs font-semibold uppercase tracking-wide text-muted">
              Operating point
            </span>
            <select
              className="mt-1 rounded-md border border-grid bg-surface p-1"
              value={point}
              onChange={(event) => setPoint(event.target.value)}
            >
              {(m.primary_points.length ? m.primary_points : ['total_10']).map(
                (p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ),
              )}
            </select>
          </label>

          <fieldset className="text-sm">
            <legend className="block text-xs font-semibold uppercase tracking-wide text-muted">
              Models
            </legend>
            <div
              className="mt-1 inline-flex rounded-md border border-grid"
              role="group"
              aria-label="Scoring mode"
            >
              {(['own', 'averaged'] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={mode === value}
                  onClick={() => setMode(value)}
                  className={`px-3 py-1 text-sm first:rounded-l-md last:rounded-r-md ${
                    mode === value
                      ? 'bg-ink text-surface'
                      : 'bg-surface text-ink2'
                  }`}
                >
                  {MODE_LABEL[value]}
                </button>
              ))}
            </div>
          </fieldset>
        </div>
        {m.suggested_settings?.length > 0 && (
          <div className="mt-3 rounded-md border border-grid p-2">
            <p className="text-xs font-semibold">
              Settings where the toggle changes the answer
            </p>
            <ul className="mt-1 space-y-0.5 text-xs text-ink2">
              {m.suggested_settings.map((hint) => (
                <li key={`${hint.detection_time}-${hint.institution}`}>
                  <button
                    type="button"
                    className="underline"
                    onClick={() => {
                      const index = m.detection_times.indexOf(hint.detection_time)
                      if (index >= 0) setWindowIndex(index)
                      setInstitution(hint.institution)
                      setMode('own')
                    }}
                  >
                    {hint.detection_time}, I{hint.institution}
                  </button>
                  {' — own misses it, averaging finds it'}
                  {hint.hours_before_deadline !== null &&
                    ` ${hint.hours_before_deadline} h before the deadline`}
                </li>
              ))}
            </ul>
            {Object.keys(m.setting_outcomes ?? {}).length > 0 && (
              <p className="mt-1 text-xs text-muted">
                Over all {Object.values(m.setting_outcomes).reduce((a, b) => a + b, 0)}{' '}
                window-and-institution settings offered here: both modes find
                it in {m.setting_outcomes.both ?? 0}, averaging only in{' '}
                {m.setting_outcomes.averaged_only ?? 0}, own model only in{' '}
                {m.setting_outcomes.own_only ?? 0}, neither in{' '}
                {m.setting_outcomes.neither ?? 0}. Averaging is not a
                guaranteed improvement, and this counts every case rather than
                the favourable ones.
              </p>
            )}
          </div>
        )}
        {windowMeta && (
          <p className="mt-2 text-xs text-muted">
            This window: {windowMeta.n_edges.toLocaleString()} transactions,{' '}
            {windowMeta.n_accounts.toLocaleString()} accounts,{' '}
            {windowMeta.n_laundering_edges} laundering edge(s), of which{' '}
            {windowMeta.n_campaign_edges} belong to the demo campaign.
          </p>
        )}
      </ChartCard>

      <ChartCard
        title={`Output — ${MODE_LABEL[mode]}`}
        howToRead="Purity is the share of a cluster's accounts that belong to the demo campaign; coverage is how many of the campaign's accounts the cluster contains. Both come from the frozen matching rule, so they are the same quantities the research results use."
      >
        {run.isLoading && <LoadingBlock label="Scoring" />}
        {run.error && <ErrorState error={run.error} />}
        {run.data && <Output run={run.data} />}
      </ChartCard>

      <ChartCard
        title="And how often does it hold?"
        howToRead="The panel above shows one campaign. This is where the aggregate from the full evaluation goes: each arm's recall relative to its own ceiling, with the number of campaigns behind it. Until that run finishes there is nothing here, and a development run is never shown in its place."
      >
        {aggregate.isLoading && <LoadingBlock label="Loading aggregates" />}
        {aggregate.error && (
          <EmptyState
            title="No aggregate figures yet"
            description={
              aggregate.error instanceof ApiError
                ? aggregate.error.detail
                : 'The comparison artifact is not available.'
            }
          />
        )}
        {aggregate.data && !aggregate.data.available && (
          <EmptyState
            title="No aggregate figures yet"
            description={
              aggregate.data.reason ??
              'The comparison artifact has no rows to show.'
            }
          />
        )}
        {aggregate.data?.available && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">
                Aggregate comparison across conditions
              </caption>
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-muted">
                  <th className="pb-1 pr-3">Contrast</th>
                  <th className="pb-1 pr-3">Unit</th>
                  <th className="pb-1 pr-3">Target</th>
                  <th className="pb-1 pr-3">Point</th>
                  <th className="pb-1 pr-3 text-right">Pairs</th>
                  <th className="pb-1 pr-3 text-right">Mean difference</th>
                  <th className="pb-1 pr-3">Below seed noise?</th>
                </tr>
              </thead>
              <tbody>
                {aggregate.data.rows.slice(0, 40).map((row, index) => (
                  <tr
                    key={index}
                    className="border-t border-grid"
                  >
                    <td className="py-1 pr-3">{String(row.contrast ?? '')}</td>
                    <td className="py-1 pr-3">{String(row.unit ?? '')}</td>
                    <td className="py-1 pr-3 tabular-nums">
                      {String(row.visibility_target ?? '')}
                    </td>
                    <td className="py-1 pr-3">
                      {String(row.operating_point ?? '')}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {String(row.n_pairs ?? '')}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {typeof row.mean_difference === 'number'
                        ? row.mean_difference.toFixed(3)
                        : String(row.mean_difference ?? '')}
                    </td>
                    <td className="py-1 pr-3">
                      {row.below_seed_noise_floor === true ? (
                        <Pill tone="warning">inside noise</Pill>
                      ) : (
                        ''
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted">
              A difference flagged <em>inside noise</em> is smaller than the
              model-seed spread Phase 3 measured at centralized (~0.04 relative
              recall, about 2 campaigns of 56). Phase 4 holds the model seed
              fixed, so it cannot distinguish such a difference from noise.
            </p>
          </div>
        )}
      </ChartCard>
    </div>
  )
}
