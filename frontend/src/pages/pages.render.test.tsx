/**
 * Render smoke tests: mount each real page against a stubbed API and check
 * the content a reader actually needs is on screen. These stand in for
 * clicking through the app in a browser.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ReactElement } from 'react'

import { ThemeProvider } from '../lib/ThemeProvider'
import { HomePage } from './HomePage'
import { DatasetPage } from './DatasetPage'
import { CampaignExplorerPage } from './CampaignExplorerPage'

const SUMMARY = {
  generated_at: '2026-10-06T00:00:00',
  source: { trans_csv: 'x', patterns_txt: 'y' },
  rows: { total: 5078345, pre_cutoff: 5077237, post_cutoff: 1108 },
  date_range: { min: '2022-09-01T00:00:00', max: '2022-09-18T16:18:00' },
  banks: { total: 30470 },
  laundering: { rows: 5177, in_campaign: 3209, unassigned: 1968 },
  campaigns: {
    total: 370,
    eval_ok: 258,
    pattern_transactions: 3209,
    reassignable: 152,
  },
  by_base_type: { CYCLE: 54, 'FAN-OUT': 48, BIPARTITE: 49 },
  by_split: {
    train: { campaigns: 179, eval_ok: 100, crosses_cutoff: 6, shares_train_account: 0 },
    val: { campaigns: 48, eval_ok: 26, crosses_cutoff: 18, shares_train_account: 0 },
    test: { campaigns: 113, eval_ok: 56, crosses_cutoff: 2, shares_train_account: 7 },
    stress: { campaigns: 30, eval_ok: 76, crosses_cutoff: 74, shares_train_account: 0 },
  },
  by_group: { fragmentable: 264, hub: 106 },
  by_group_eval_ok: { fragmentable: 152, hub: 106 },
  cutoff: {
    timestamp: '2022-09-11T00:00:00',
    rows_excluded: 1108,
    laundering_rows_excluded: 655,
    laundering_share_excluded: 0.591,
    laundering_share_overall: 0.00102,
  },
  transactions_per_day: {
    '2022-09-01': 1114921,
    '2022-09-10': 208325,
    '2022-09-11': 396,
    '2022-09-12': 281,
  },
  campaigns_starting_per_day: { '2022-09-01': 38 },
  institutions: { k: 8 },
  visibility: { targets: [1, 0.75, 0.5, 0.25], seeds: [0, 1, 2, 3, 4], bins: ['100', '75', '50', 'low'] },
}

const CAMPAIGN = {
  campaign_id: 0,
  base_type: 'CYCLE',
  n_txn: 10,
  n_accounts: 10,
  n_banks: 9,
  start: '2022-09-01T00:03:00',
  end: '2022-09-04T15:51:00',
  duration_h: 87.8,
  crosses_cutoff: false,
  deadline: '2022-09-04T15:51:00',
  eval_ok: true,
  split: 'train',
  group: 'fragmentable',
  shares_train_account: false,
  reassignable: true,
}

const CAMPAIGNS = {
  items: [CAMPAIGN, { ...CAMPAIGN, campaign_id: 1, base_type: 'FAN-OUT', group: 'hub' }],
  total: 2,
  page: 1,
  page_size: 25,
  pages: 1,
}

const HEALTH = {
  status: 'ok',
  artifacts_dir: '/artifacts',
  artifacts: { 'campaigns.parquet': true, 'visibility/seed_0.parquet': true },
  core_artifacts_available: true,
  results_available: false,
}

function stubApi() {
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      let body: unknown = {}
      if (url.includes('/api/health')) body = HEALTH
      else if (url.includes('/api/summary')) body = SUMMARY
      else if (url.includes('/api/campaigns')) body = CAMPAIGNS
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(body),
      } as Response)
    }),
  )
}

function renderPage(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter>
          <Routes>
            <Route path="/" element={ui} />
          </Routes>
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  stubApi()
  // Recharts measures its container; jsdom reports zero, so give it a size
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    value: 800,
  })
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    value: 400,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('HomePage', () => {
  it('leads with the research question in plain language', async () => {
    renderPage(<HomePage />)
    expect(
      screen.getByRole('heading', { name: /how much does\s+working together actually help/i }),
    ).toBeInTheDocument()
  })

  it('shows real key statistics from the API', async () => {
    renderPage(<HomePage />)
    expect(await screen.findByText('5,078,345')).toBeInTheDocument()
    expect(await screen.findByText('370')).toBeInTheDocument()
    expect(await screen.findByText('258 large enough to evaluate')).toBeInTheDocument()
  })

  it('reports pipeline status from the artifacts that exist', async () => {
    renderPage(<HomePage />)
    expect(
      await screen.findByText('Dataset processed and campaigns extracted'),
    ).toBeInTheDocument()
    // results do not exist yet, so that stage is explicitly not marked done
    const results = await screen.findByText(/Experiment results \(Phase 5\)/)
    expect(results).toHaveTextContent('(not yet)')
  })
})

describe('DatasetPage', () => {
  it('explains the cutoff with the real numbers', async () => {
    renderPage(<DatasetPage />)
    await waitFor(() =>
      expect(
        screen.getByText(/Why we excluded everything after Sept 10/),
      ).toBeInTheDocument(),
    )
    expect(screen.getByText('1,108')).toBeInTheDocument()
    expect(screen.getByText('59%')).toBeInTheDocument()
  })

  it('breaks campaigns down by split, including censored counts', async () => {
    renderPage(<DatasetPage />)
    expect(await screen.findByText('Splits')).toBeInTheDocument()
    expect(screen.getByText('Test (Sept 6-7)')).toBeInTheDocument()
  })

  it('separates the fragmentable treatment group from the control group', async () => {
    renderPage(<DatasetPage />)
    expect(
      await screen.findByText('Which campaigns can actually be hidden'),
    ).toBeInTheDocument()
    expect(screen.getByText('(control)')).toBeInTheDocument()
  })
})

describe('CampaignExplorerPage', () => {
  it('lists campaigns with a result count', async () => {
    renderPage(<CampaignExplorerPage />)
    expect(await screen.findByText('2')).toBeInTheDocument()
    expect(await screen.findByText(/campaigns in the dataset/)).toBeInTheDocument()
  })

  it('renders a sortable table with flag pills', async () => {
    renderPage(<CampaignExplorerPage />)
    const header = await screen.findByRole('button', { name: /Txns/ })
    expect(header).toBeInTheDocument()

    // scope to the table: "Hub" is also a Group filter option
    const table = within(screen.getByRole('table'))
    expect(table.getAllByText('evaluable')).toHaveLength(2)
    expect(table.getByText('Hub')).toBeInTheDocument()
    expect(table.getByText('Fragmentable')).toBeInTheDocument()
  })

  it('exposes every row as a keyboard-reachable link', async () => {
    renderPage(<CampaignExplorerPage />)
    const rows = await screen.findAllByRole('link', { name: /Open campaign/ })
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveAttribute('tabindex', '0')
  })
})
