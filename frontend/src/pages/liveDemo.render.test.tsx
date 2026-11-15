/**
 * The live panel's two obligations to a viewer.
 *
 * 1. With no bundle installed it shows an **empty state**, never placeholder
 *    numbers. A live panel that invents an output is worse than no panel.
 * 2. With a bundle it states, unmissably, that the campaign was **chosen to
 *    illustrate the mechanism** and is not a representative result - and it
 *    still shows the aggregate section, because the panel must not ship
 *    without it.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LiveDemoPage } from './LiveDemoPage'
import { ThemeProvider } from '../lib/ThemeProvider'

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  })
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MemoryRouter>
          <LiveDemoPage />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  )
}

const MANIFEST = {
  visibility_target: 1.0,
  seed: 0,
  lookback_h: 48,
  split: 'test',
  k_institutions: 8,
  institutions_with_a_model: [0, 1, 2],
  demo_campaign: {
    campaign_id: 42,
    accounts: ['a', 'b', 'c', 'd'],
    institutions_holding: [0, 1, 2],
    achieved_visibility: 0.5,
  },
  selection_rule: ['candidates: test-split campaigns', 'rank by institutions'],
  selection_disclaimer:
    'Chosen to ILLUSTRATE THE MECHANISM, by the rule above. It is not a representative result and must never be described as one.',
  selection_audit: { rank_chosen: 0, n_candidates: 7, n_skipped: 0 },
  windows: [
    {
      detection_time: '2022-09-07 00:00:00',
      n_edges: 100,
      n_accounts: 40,
      n_laundering_edges: 5,
      n_campaign_edges: 3,
    },
  ],
  detection_times: ['2022-09-07 00:00:00'],
  primary_points: ['f1', 'total_10'],
  taus: { total_10: { '0': 0.5 } },
  match_rule: { min_account_fraction: 0.5, min_accounts: 3 },
  models_mb: 1.2,
  modes: ['own', 'averaged'],
  development_only: '1% sample: a development check, not a result.',
  generated: '2026-11-15T00:00:00',
}

const RUN = {
  detection_time: '2022-09-07 00:00:00',
  institution: 0,
  mode: 'own',
  tau: 0.5,
  n_edges_in_window: 100,
  n_edges_in_view: 40,
  n_flagged_edges: 6,
  flagged_edges: [],
  clusters: [
    {
      accounts: ['a', 'b', 'c'],
      n_accounts: 3,
      kind: 'hit',
      hit_campaigns: [42],
      matches_demo_campaign: true,
      vs_demo_campaign: {
        n_cluster_accounts: 3,
        n_campaign_accounts: 4,
        overlap: 3,
        purity: 1,
        coverage_accounts: 3,
        coverage_fraction: 0.75,
        min_fraction: 0.5,
        min_accounts: 3,
        matches: true,
      },
      institutions: [0, 1],
    },
  ],
  n_clusters: 1,
  alert_count: 1,
  demo_campaign_id: 42,
  demo_campaign_detected: true,
  best_purity_vs_demo_campaign: 1,
  best_coverage_vs_demo_campaign: 0.75,
  lead_time: {
    detected: true,
    deadline: '2022-09-08 00:00:00',
    campaign_end: '2022-09-09 00:00:00',
    hours_before_deadline: 24,
    hours_before_end: 48,
    beat_the_deadline: true,
  },
  latency_ms: { featurise: 120, predict: 30, clusters: 5, total: 160 },
}

function stub(routes: Record<string, { status: number; body: unknown }>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const match = Object.keys(routes).find((key) => url.includes(key))
      const hit = match ? routes[match] : { status: 404, body: { detail: 'nope' } }
      return new Response(JSON.stringify(hit.body), {
        status: hit.status,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('LiveDemoPage', () => {
  it('shows an empty state, not invented numbers, when no bundle is installed', async () => {
    stub({
      '/api/demo/manifest': {
        status: 404,
        body: { detail: 'The live demo bundle is not installed.' },
      },
    })
    renderPage()
    expect(
      await screen.findByText(/No live demo bundle installed/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/bundle is not installed/i),
    ).toBeInTheDocument()
    // nothing that looks like a result
    expect(screen.queryByText(/Flagged edges/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Alerts raised/i)).not.toBeInTheDocument()
  })

  it('states the campaign was chosen to illustrate the mechanism', async () => {
    stub({
      '/api/demo/manifest': { status: 200, body: MANIFEST },
      '/api/demo/run': { status: 200, body: RUN },
      '/api/demo/aggregate': {
        status: 200,
        body: { available: false, reason: 'no comparison artifact yet', rows: [] },
      },
    })
    renderPage()
    // the card's own heading says it, so it cannot be scrolled past
    expect(
      await screen.findByRole('heading', {
        name: /chosen to illustrate the mechanism .* not a representative result/i,
      }),
    ).toBeInTheDocument()
    // and the disclaimer repeats it in the body
    expect(
      screen.getByText(/must never be described as one/i),
    ).toBeInTheDocument()
    // the selection rule itself is on the page, not just a claim about it
    expect(screen.getByText(/rank by institutions/i)).toBeInTheDocument()
  })

  it('ships the aggregate section even when the aggregate is missing', async () => {
    stub({
      '/api/demo/manifest': { status: 200, body: MANIFEST },
      '/api/demo/run': { status: 200, body: RUN },
      '/api/demo/aggregate': {
        status: 200,
        body: { available: false, reason: 'no comparison artifact yet', rows: [] },
      },
    })
    renderPage()
    expect(
      await screen.findByText(/And how often does it hold\?/i),
    ).toBeInTheDocument()
    expect(screen.getByText(/No aggregate figures yet/i)).toBeInTheDocument()
    expect(screen.getByText(/no comparison artifact yet/i)).toBeInTheDocument()
  })

  it('shows the live output and its latency', async () => {
    stub({
      '/api/demo/manifest': { status: 200, body: MANIFEST },
      '/api/demo/run': { status: 200, body: RUN },
      '/api/demo/aggregate': {
        status: 200,
        body: { available: false, reason: 'none', rows: [] },
      },
    })
    renderPage()
    expect(await screen.findByText(/Alerts raised/i)).toBeInTheDocument()
    expect(screen.getByText(/Recomputed in/i)).toBeInTheDocument()
    expect(screen.getByText(/160 ms/i)).toBeInTheDocument()
    // purity and coverage are shown, from the frozen rule's own numbers
    const table = screen.getByRole('table', {
      name: /Clusters extracted at this setting/i,
    })
    expect(within(table).getByText('100%')).toBeInTheDocument()
    expect(within(table).getByText('3/4')).toBeInTheDocument()
  })
})
