/**
 * The live demo panel, for a faculty presentation.
 *
 * Input -> output, computed on the spot. The viewer picks a detection window
 * and an institution, and the panel scores that one view **both ways at
 * once**:
 *
 *   own      - that institution's own model, alone (stage 4.1)
 *   averaged - all K models applied to that same view, scores averaged (4.2)
 *
 * Both are computed and shown side by side, with the rows that differ marked,
 * so the difference is visible without anyone flipping a control or
 * remembering a previous number.
 *
 * The panel opens on the first setting where the two modes **differ**, chosen
 * from the bundle's own `suggested_settings` rather than from a hard-coded
 * timestamp. If the bundle offers no such setting the panel says so rather
 * than opening on a window where nothing happens.
 *
 * Every number on this page is computed by the backend from `fraudcamp`'s own
 * functions on each request. This component formats; it does not compute a
 * research metric, and when the bundle or the aggregate artifact is missing it
 * shows an empty state rather than placeholder numbers.
 */

import { useState, type ReactNode } from 'react'

import { ChartCard, InstitutionBadge, Pill } from '../components/cards'
import { EmptyState, ErrorState, LoadingBlock } from '../components/states'
import { useDemoAggregate, useDemoManifest, useDemoRun } from '../api/hooks'
import { ApiError } from '../api/client'
import type { DemoCluster, DemoManifest, DemoRun } from '../api/types'

const MODE_LABEL: Record<string, string> = {
  own: 'This institution alone',
  averaged: 'All 8 models, scores averaged',
}

/** How many non-matching clusters to show before the disclosure. */
const OTHER_CLUSTERS_SHOWN = 3

const KIND_LABEL: Record<string, string> = {
  hit: 'matches a campaign',
  other_split_hit: 'matches a campaign from another split',
  ambiguous: 'touches unassigned laundering',
  false_alarm: 'false alarm',
}

/**
 * A disclosure for apparatus: how a number was derived, never what it says.
 *
 * Qualifications - the development-check warning, "not a representative
 * result", the disclaimer, the four-way outcome count - are never put in one
 * of these. Only the derivation is.
 */
function Details({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="mt-3 border-t border-grid pt-2">
      <summary className="cursor-pointer text-xs font-semibold text-muted">
        {summary}
      </summary>
      <div className="mt-2 text-xs leading-relaxed text-muted">{children}</div>
    </details>
  )
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
    <tr
      className={`border-t border-grid ${
        cluster.matches_demo_campaign ? 'bg-grid/40' : ''
      }`}
    >
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

function ClusterHead() {
  return (
    <thead>
      <tr className="text-left text-xs uppercase tracking-wide text-muted">
        <th className="pb-1 pr-3">Cluster</th>
        <th className="pb-1 pr-3 text-right">Accounts</th>
        <th className="pb-1 pr-3 text-right">Purity</th>
        <th className="pb-1 pr-3 text-right">Coverage</th>
        <th className="pb-1 pr-3">Institutions</th>
      </tr>
    </thead>
  )
}

/**
 * The clusters, matched ones first.
 *
 * Ordering and truncation only - every cluster the backend returned is still
 * reachable behind the disclosure, and nothing is recomputed here. Twelve
 * rows of 0% is noise on a projector, and the row that matters is the one
 * that matched the demo campaign.
 */
function ClusterTable({ run, label }: { run: DemoRun; label: string }) {
  const matched = run.clusters.filter((c) => c.matches_demo_campaign)
  const others = run.clusters.filter((c) => !c.matches_demo_campaign)
  const shown = [...matched, ...others.slice(0, OTHER_CLUSTERS_SHOWN)]
  const hidden = others.slice(OTHER_CLUSTERS_SHOWN)

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">{`Clusters extracted at this setting — ${label}`}</caption>
        <ClusterHead />
        <tbody>
          {run.clusters.length === 0 && (
            <tr>
              <td colSpan={5} className="py-2 text-muted">
                No cluster passed the threshold: no alert at all here.
              </td>
            </tr>
          )}
          {shown.map((cluster, index) => (
            <ClusterRow key={index} cluster={cluster} />
          ))}
        </tbody>
      </table>
      {hidden.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs font-semibold text-muted">
            Show all {run.clusters.length} clusters ({hidden.length} more)
          </summary>
          <table className="mt-2 w-full text-sm">
            <caption className="sr-only">
              {`The remaining clusters — ${label}`}
            </caption>
            <ClusterHead />
            <tbody>
              {hidden.map((cluster, index) => (
                <ClusterRow key={index} cluster={cluster} />
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  )
}

function leadLabel(run: DemoRun | undefined): string {
  const lead = run?.lead_time
  if (!run) return '—'
  if (!lead?.detected || lead.hours_before_deadline === null) return 'not detected'
  return `${num(lead.hours_before_deadline, 1)} h before the deadline`
}

interface Line {
  label: string
  own: string
  averaged: string
  /** Rendered in the colour of a hit when the mode found the campaign. */
  emphasise?: boolean
}

/** The two modes on one grid, with the rows that differ marked. */
function Comparison({ own, averaged }: { own?: DemoRun; averaged?: DemoRun }) {
  const lines: Line[] = [
    {
      label: 'Flagged edges',
      own: own ? own.n_flagged_edges.toLocaleString() : '—',
      averaged: averaged ? averaged.n_flagged_edges.toLocaleString() : '—',
    },
    {
      label: 'Alerts raised',
      own: own ? own.alert_count.toLocaleString() : '—',
      averaged: averaged ? averaged.alert_count.toLocaleString() : '—',
    },
    {
      label: 'Demo campaign',
      own: own ? (own.demo_campaign_detected ? 'found' : 'missed') : '—',
      averaged: averaged
        ? averaged.demo_campaign_detected
          ? 'found'
          : 'missed'
        : '—',
      emphasise: true,
    },
    {
      label: 'Best purity',
      own: pct(own?.best_purity_vs_demo_campaign),
      averaged: pct(averaged?.best_purity_vs_demo_campaign),
    },
    {
      label: 'Best coverage',
      own: pct(own?.best_coverage_vs_demo_campaign),
      averaged: pct(averaged?.best_coverage_vs_demo_campaign),
    },
    {
      label: 'Lead time',
      own: leadLabel(own),
      averaged: leadLabel(averaged),
      emphasise: true,
    },
  ]

  return (
    <table className="w-full text-sm">
      <caption className="sr-only">
        The same view scored both ways, with the differing rows marked
      </caption>
      <thead>
        <tr className="text-left text-xs uppercase tracking-wide text-muted">
          <th className="pb-2 pr-3"> </th>
          <th className="pb-2 pr-3">{MODE_LABEL.own}</th>
          <th className="pb-2 pr-3">{MODE_LABEL.averaged}</th>
        </tr>
      </thead>
      <tbody>
        {lines.map((line) => {
          const differs = line.own !== line.averaged
          const cell = (value: string) =>
            `py-2 pr-3 text-lg font-semibold tabular-nums ${
              line.emphasise && value === 'found'
                ? 'text-good'
                : differs
                  ? 'text-ink'
                  : 'text-ink2'
            }`
          return (
            <tr
              key={line.label}
              className={`border-t border-grid ${
                differs ? 'border-l-2 border-l-axis bg-grid/30' : ''
              }`}
            >
              <th
                scope="row"
                className="py-2 pl-2 pr-3 text-left align-middle text-xs font-medium uppercase tracking-wide text-muted"
              >
                {line.label}
                {differs && (
                  <span className="ml-2 normal-case">
                    <Pill>differs</Pill>
                  </span>
                )}
              </th>
              <td className={cell(line.own)}>{line.own}</td>
              <td className={cell(line.averaged)}>{line.averaged}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/**
 * Said when the viewer has moved off a listed setting and the two modes
 * agree. Which way they agree is read from the runs on screen, not assumed
 * from the list, so the note cannot contradict the table under it.
 */
function AgreementNote({
  outcomes,
  detected,
  onBack,
  canGoBack,
}: {
  outcomes: Record<string, number> | undefined
  detected: boolean
  onBack: () => void
  canGoBack: boolean
}) {
  const counts = outcomes ?? {}
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  const same = detected ? counts.both : counts.neither
  return (
    <p className="mb-3 rounded-md border border-grid bg-plane p-2 text-xs text-ink2">
      <strong>
        At this setting the two modes agree on the outcome —{' '}
        {detected ? 'both find' : 'neither finds'} the campaign.
      </strong>{' '}
      {total > 0 && typeof same === 'number' && (
        <>
          That is the common case: {detected ? 'both do' : 'neither does'} in{' '}
          <strong>
            {same} of {total}
          </strong>{' '}
          settings offered here.{' '}
        </>
      )}
      {canGoBack && (
        <button type="button" className="underline" onClick={onBack}>
          Back to the setting where they differ
        </button>
      )}
    </p>
  )
}

/** One mode's own detail: the window it saw, its clusters, its latency. */
function ModeColumn({
  mode,
  query,
}: {
  mode: 'own' | 'averaged'
  query: ReturnType<typeof useDemoRun>
}) {
  return (
    <div className="rounded-lg border border-grid p-3">
      <h4 className="mb-2 text-sm font-semibold text-ink">{MODE_LABEL[mode]}</h4>
      {query.isLoading && <LoadingBlock label="Scoring" />}
      {query.error && <ErrorState error={query.error} />}
      {query.data && query.data.empty_reason && (
        <EmptyState title="Nothing to score" description={query.data.empty_reason} />
      )}
      {query.data && !query.data.empty_reason && (
        <>
          <p className="mb-2 text-xs text-muted">
            {query.data.n_flagged_edges.toLocaleString()} of{' '}
            {query.data.n_edges_in_view.toLocaleString()} edges in this view
            flagged
            {typeof query.data.n_models_used === 'number' &&
              ` · ${query.data.n_models_used} model(s)`}
          </p>
          <ClusterTable run={query.data} label={MODE_LABEL[mode]} />
          <div className="mt-2">
            <LatencyNote run={query.data} />
          </div>
        </>
      )}
    </div>
  )
}

/**
 * The setting the panel opens on: the first one the bundle lists where the
 * two modes disagree. Picked by that rule, never by a hard-coded timestamp,
 * so a rebuilt bundle moves the default with it. `null` when the bundle
 * offers no such setting - which the page then states outright.
 */
function openingSetting(
  manifest: DemoManifest | undefined,
): { windowIndex: number; institution: number } | null {
  const settings = manifest?.suggested_settings ?? []
  const times = manifest?.detection_times ?? []
  for (const setting of settings) {
    const index = times.indexOf(setting.detection_time)
    if (index >= 0) {
      return { windowIndex: index, institution: setting.institution }
    }
  }
  return null
}

/** Is this (window, institution) one of the settings the bundle lists? */
function isListedSetting(
  manifest: DemoManifest | undefined,
  detectionTime: string,
  institution: number,
): boolean {
  return (manifest?.suggested_settings ?? []).some(
    (setting) =>
      setting.detection_time === detectionTime &&
      setting.institution === institution,
  )
}

/** Whose view to score when neither the viewer nor the bundle has said. */
function fallbackInstitution(manifest: DemoManifest | undefined): number {
  const holding = manifest?.demo_campaign?.institutions_holding ?? []
  const withModel = manifest?.institutions_with_a_model ?? []
  return holding.find((i) => withModel.includes(i)) ?? withModel[0] ?? 0
}

export function LiveDemoPage() {
  const manifest = useDemoManifest()
  const aggregate = useDemoAggregate()
  /** `null` means "not chosen by the viewer yet", so the opening setting wins. */
  const [windowIndex, setWindowIndex] = useState<number | null>(null)
  const [institution, setInstitution] = useState<number | null>(null)
  const [point, setPoint] = useState('total_10')

  const opening = openingSetting(manifest.data)
  const effectiveWindow = windowIndex ?? opening?.windowIndex ?? 0
  const detectionTime = manifest.data?.detection_times?.[effectiveWindow] ?? ''
  const chosenInstitution =
    institution ?? opening?.institution ?? fallbackInstitution(manifest.data)

  const enabled = Boolean(detectionTime) && manifest.isSuccess
  const ownRun = useDemoRun({
    detection_time: detectionTime,
    institution: chosenInstitution,
    mode: 'own',
    operating_point: point,
    enabled,
  })
  const averagedRun = useDemoRun({
    detection_time: detectionTime,
    institution: chosenInstitution,
    mode: 'averaged',
    operating_point: point,
    enabled,
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
  const windowMeta = m.windows?.[effectiveWindow]

  // Shown only when both runs are in and they agree, so the note is never
  // at odds with the `differs` markers in the table beneath it.
  const bothIn = Boolean(ownRun.data && averagedRun.data)
  const detected = Boolean(ownRun.data?.demo_campaign_detected)
  const theyAgree =
    bothIn && detected === Boolean(averagedRun.data?.demo_campaign_detected)
  const offTheList =
    !isListedSetting(m, detectionTime, chosenInstitution) && theyAgree

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold">Live detection panel</h1>
        <p className="max-w-3xl text-sm text-ink2">
          One detection window, scored on the spot — <strong>both ways at
          once</strong>. The left column is an institution scoring with its{' '}
          <strong>own model</strong>; the right is the same view scored by{' '}
          <strong>all {m.k_institutions} models with their scores averaged</strong>.
          Clusters, matches, purity, coverage, lead time and the alert count are
          recomputed on every change, and the rows that differ are marked.
        </p>
        {m.development_only && (
          <p className="rounded-md border border-warning/50 p-2 text-xs text-ink2">
            {m.development_only}
          </p>
        )}
      </header>

      <ChartCard title="Chosen to illustrate the mechanism — not a representative result">
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
          {m.selection_disclaimer && (
            <p className="text-xs font-semibold">{m.selection_disclaimer}</p>
          )}
          <Details summary="How this campaign was selected, and from what">
            <p className="mb-1">
              It was selected by a fixed rule, stated here in full. It shows
              what the mechanism does on one campaign; how often it holds is
              the aggregate at the bottom of this page, and only that speaks
              to typicality.
            </p>
            <ol className="list-decimal space-y-1 pl-5">
              {m.selection_rule.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
            <p className="mt-2">
              Chosen at rank {m.selection_audit.rank_chosen ?? '—'} of{' '}
              {m.selection_audit.n_candidates ?? '—'} candidates,{' '}
              {m.selection_audit.n_skipped} skipped as unreachable by the
              centralized oracle.
            </p>
          </Details>
        </div>
      </ChartCard>

      <ChartCard title="Input">
        {!opening && (
          <p className="mb-3 rounded-md border border-warning/50 p-2 text-xs text-ink2">
            <strong>No setting in this bundle shows the two modes
            disagreeing.</strong> The panel has opened on the first window
            instead, so the two columns below may well be identical — that is
            the bundle, not a failure of the page. Rebuild it with{' '}
            <code>scripts/make_demo_bundle.py</code> to look for one.
          </p>
        )}
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-sm">
            <span className="block text-xs font-semibold uppercase tracking-wide text-muted">
              Detection window
            </span>
            <select
              className="mt-1 rounded-md border border-grid bg-surface p-1"
              value={effectiveWindow}
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
        </div>
        {m.suggested_settings?.length > 0 && (
          <div className="mt-3 rounded-md border border-grid p-2">
            <p className="text-xs font-semibold">
              Settings where the two modes differ
              <span className="ml-2 font-normal text-muted">
                (the panel opens on the first)
              </span>
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
        <Details summary="How to read this">
          Pick a window and an institution. Both scoring modes are recomputed
          by the backend on every change and shown together below, so the
          comparison never depends on remembering a previous number.
        </Details>
      </ChartCard>

      <ChartCard title="The same view, scored both ways">
        {(ownRun.isLoading || averagedRun.isLoading) && (
          <LoadingBlock label="Scoring both ways" />
        )}
        {ownRun.error && <ErrorState error={ownRun.error} />}
        {averagedRun.error && <ErrorState error={averagedRun.error} />}
        {offTheList && (
          <AgreementNote
            outcomes={m.setting_outcomes}
            detected={detected}
            canGoBack={opening !== null}
            onBack={() => {
              if (!opening) return
              setWindowIndex(opening.windowIndex)
              setInstitution(opening.institution)
            }}
          />
        )}
        {(ownRun.data || averagedRun.data) && (
          <Comparison own={ownRun.data} averaged={averagedRun.data} />
        )}
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <ModeColumn mode="own" query={ownRun} />
          <ModeColumn mode="averaged" query={averagedRun} />
        </div>
        <Details summary="How to read this">
          Purity is the share of a cluster's accounts that belong to the demo
          campaign; coverage is how many of the campaign's accounts the cluster
          contains. Both come from the frozen matching rule, so they are the
          same quantities the research results use. A row is marked{' '}
          <em>differs</em> when the two modes do not agree on it.
        </Details>
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
