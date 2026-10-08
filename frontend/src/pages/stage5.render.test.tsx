/**
 * Render tests for the Stage 5 pages. Cytoscape needs a real canvas, so the
 * campaign graph is stubbed out here; what these tests cover is the
 * surrounding behaviour — controls, the bank's-eye readout, empty states and
 * the methodology cards.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ReactElement } from 'react'

import { ThemeProvider } from '../lib/ThemeProvider'

vi.mock('../components/CampaignGraph', () => ({
  CampaignGraph: () => <div data-testid="campaign-graph" />,
}))

const { CampaignDetailPage } = await import('./CampaignDetailPage')
const { VisibilityLabPage } = await import('./VisibilityLabPage')
const { ResultsPage } = await import('./ResultsPage')
const { MethodologyPage } = await import('./MethodologyPage')

const CAMPAIGN_DETAIL = {
  campaign: {
    campaign_id: 7,
    base_type: 'CYCLE',
    n_txn: 3,
    n_accounts: 3,
    n_banks: 3,
    start: '2022-09-01T00:00:00',
    end: '2022-09-02T00:00:00',
    duration_h: 24,
    crosses_cutoff: false,
    deadline: '2022-09-02T00:00:00',
    eval_ok: true,
    split: 'train',
    group: 'fragmentable',
    shares_train_account: false,
    reassignable: true,
  },
  transactions: [
    {
      txn_index: 0,
      campaign_id: 7,
      timestamp: '2022-09-01T00:00:00',
      from_bank: 1,
      to_bank: 2,
      src: '1_A',
      dst: '2_B',
      amount_paid: 100,
      payment_currency: 'US Dollar',
      amount_received: 100,
      receiving_currency: 'US Dollar',
      payment_format: 'ACH',
      is_laundering: 1,
      src_natural_institution: 0,
      dst_natural_institution: 1,
    },
    {
      txn_index: 1,
      campaign_id: 7,
      timestamp: '2022-09-01T12:00:00',
      from_bank: 2,
      to_bank: 3,
      src: '2_B',
      dst: '3_C',
      amount_paid: 200,
      payment_currency: 'Euro',
      amount_received: 200,
      receiving_currency: 'Euro',
      payment_format: 'ACH',
      is_laundering: 1,
      src_natural_institution: 1,
      dst_natural_institution: 2,
    },
    {
      txn_index: 2,
      campaign_id: 7,
      timestamp: '2022-09-02T00:00:00',
      from_bank: 3,
      to_bank: 1,
      src: '3_C',
      dst: '1_A',
      amount_paid: 300,
      payment_currency: 'US Dollar',
      amount_received: 300,
      receiving_currency: 'US Dollar',
      payment_format: 'ACH',
      is_laundering: 1,
      src_natural_institution: 2,
      dst_natural_institution: 0,
    },
  ],
  accounts: [
    { account: '1_A', natural_institution: 0 },
    { account: '2_B', natural_institution: 1 },
    { account: '3_C', natural_institution: 2 },
  ],
}

const CAMPAIGN_VISIBILITY = {
  campaign_id: 7,
  seed: 0,
  target: 0.25,
  achieved_visibility: 0.6666666666666666,
  bin: '75',
  campaign_reassigned: true,
  n_transactions: 3,
  accounts: [
    { account: '1_A', institution: 4, reassigned: true },
    { account: '2_B', institution: 4, reassigned: true },
    { account: '3_C', institution: 6, reassigned: true },
  ],
  institutions: [
    { institution: 4, visible_txn_indices: [0, 1, 2], n_visible: 3 },
    { institution: 6, visible_txn_indices: [1, 2], n_visible: 2 },
  ],
}

const DISTRIBUTION = {
  seed: 0,
  targets: [1, 0.75, 0.5, 0.25],
  bins: ['100', '75', '50', 'low'],
  per_target: [1, 0.75, 0.5, 0.25].map((target) => ({
    target,
    campaigns: [
      {
        campaign_id: 0,
        achieved_visibility: target,
        bin: target >= 0.9 ? '100' : target >= 0.65 ? '75' : '50',
        group: 'fragmentable',
        reassigned: true,
      },
      {
        campaign_id: 1,
        achieved_visibility: 1,
        bin: '100',
        group: 'hub',
        reassigned: false,
      },
    ],
    bin_counts: {
      '100': target >= 0.9 ? 2 : 1,
      '75': target === 0.75 ? 1 : 0,
      '50': target < 0.65 ? 1 : 0,
      low: 0,
    },
    bin_counts_by_group: {
      fragmentable: {
        '100': target >= 0.9 ? 1 : 0,
        '75': target === 0.75 ? 1 : 0,
        '50': target < 0.65 ? 1 : 0,
        low: 0,
      },
      hub: { '100': 1, '75': 0, '50': 0, low: 0 },
    },
  })),
  unfragmentable_campaigns: 2,
}

const METHODOLOGY = {
  n_entries: 2,
  entries: [
    {
      key: 'cutoff',
      title: 'Cutoff',
      definition: 'Model input is every transaction before 2022-09-11T00:00:00.',
      value: { cutoff: '2022-09-11T00:00:00' },
      reason: 'Only 1,108 rows remain after the cutoff and 59.1% of them are laundering.',
    },
    {
      key: 'institutions',
      title: 'Institutions',
      definition: 'Banks are grouped into K = 8 institutions.',
      value: { k: 8 },
      reason: 'Every transaction is visible to both endpoint banks.',
    },
  ],
}

const SUMMARY = {
  cutoff: { timestamp: '2022-09-11T00:00:00' },
}

function stubApi(options: { resultsExist?: boolean } = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      const respond = (body: unknown, status = 200) =>
        Promise.resolve({
          ok: status < 400,
          status,
          json: () => Promise.resolve(body),
        } as Response)

      if (url.includes('/api/results/')) {
        if (options.resultsExist) return respond({ items: [{}], total: 1 })
        return respond(
          { detail: 'Experiment results are not available yet.' },
          404,
        )
      }
      if (url.includes('/visibility/distribution')) return respond(DISTRIBUTION)
      if (url.includes('/visibility')) return respond(CAMPAIGN_VISIBILITY)
      if (url.includes('/api/methodology')) return respond(METHODOLOGY)
      if (url.includes('/api/summary')) return respond(SUMMARY)
      if (url.includes('/api/campaigns/7')) return respond(CAMPAIGN_DETAIL)
      if (url.includes('/api/health')) {
        return respond({
          status: 'ok',
          artifacts_dir: '/a',
          artifacts: {},
          core_artifacts_available: true,
          results_available: false,
        })
      }
      return respond({})
    }),
  )
}

function renderAt(ui: ReactElement, path = '/') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/campaigns/:campaignId" element={ui} />
            <Route path="/" element={ui} />
          </Routes>
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  stubApi()
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

describe('CampaignDetailPage', () => {
  it('shows the campaign, its flags and its transactions', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    expect(await screen.findByRole('heading', { name: 'Campaign 7' })).toBeInTheDocument()
    expect(screen.getByTestId('campaign-graph')).toBeInTheDocument()
    expect(screen.getByText(/of 3 transactions/)).toBeInTheDocument()
  })

  it('starts on the natural bank split, with the seed selector disabled', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    await screen.findByRole('heading', { name: 'Campaign 7' })

    expect(screen.getByLabelText('View')).toHaveValue('natural')
    expect(screen.getByLabelText('Seed')).toBeDisabled()
    expect(
      screen.getByText(/Accounts sit with the institution that owns their real bank/),
    ).toBeInTheDocument()
  })

  it('shows achieved visibility and bin once a target is chosen', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    await screen.findByRole('heading', { name: 'Campaign 7' })

    await userEvent.selectOptions(screen.getByLabelText('View'), '0.25')

    expect(await screen.findByText('67%')).toBeInTheDocument()
    expect(screen.getByText(/bin 75%/)).toBeInTheDocument()
    expect(screen.getByLabelText('Seed')).toBeEnabled()
  })

  it('reports how much one bank can see when a bank is selected', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    await screen.findByRole('heading', { name: 'Campaign 7' })

    expect(screen.getByText(/Showing everything, as no single bank ever could/)).toBeInTheDocument()

    // natural split puts each account at a different institution, so a single
    // bank sees only the two transactions touching its own account
    const tabs = screen.getAllByRole('tab')
    await userEvent.click(tabs[1])

    await waitFor(() =>
      expect(screen.getByText(/This bank sees/)).toHaveTextContent(
        'This bank sees 2 of 3 transactions.',
      ),
    )
  })

  it('explains each metadata flag in plain language', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    await screen.findByRole('heading', { name: 'Campaign 7' })

    const sidebar = within(screen.getByRole('complementary'))
    expect(sidebar.getByText('Evaluated')).toBeInTheDocument()
    expect(
      sidebar.getByText(/At least 3 transactions and 3 accounts/),
    ).toBeInTheDocument()
    expect(
      sidebar.getByText(/Its accounts can be spread across banks/),
    ).toBeInTheDocument()
  })

  it('shows an empty state for detection rather than inventing results', async () => {
    renderAt(<CampaignDetailPage />, '/campaigns/7')
    expect(
      await screen.findByText('No detection results for this campaign yet'),
    ).toBeInTheDocument()
  })
})

describe('VisibilityLabPage', () => {
  it('computes visibility live for the user-built cycle', async () => {
    renderAt(<VisibilityLabPage />)
    await screen.findByRole('heading', { name: 'Visibility Lab' })

    // all six accounts start in one bank, so that bank sees everything
    const readout = screen.getByLabelText('Visibility of your scenario')
    expect(readout).toHaveTextContent('100%')

    // Moving a single account changes nothing: the bank holding the other
    // five still touches every transaction in the ring. This is the point the
    // toy is meant to teach.
    await userEvent.click(screen.getByRole('button', { name: /^Account A,/ }))
    await waitFor(() => expect(readout).toHaveTextContent('100%'))

    // Splitting the ring into three contiguous pairs (A,B | C,D | E,F) is
    // what actually works: every bank then sees exactly half of it.
    await userEvent.click(screen.getByRole('button', { name: /^Account A,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account A,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account B,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account B,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account B,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account C,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account D,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account E,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account E,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account F,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account F,/ }))

    await waitFor(() => expect(readout).toHaveTextContent('50%'))
  })

  it('shows the hub shape can never be fragmented', async () => {
    renderAt(<VisibilityLabPage />)
    await screen.findByRole('heading', { name: 'Visibility Lab' })

    await userEvent.selectOptions(screen.getByLabelText('Shape'), 'fan-out')
    const readout = screen.getByLabelText('Visibility of your scenario')
    expect(readout).toHaveTextContent('100%')

    // move two leaves away from the hub: the hub's bank still sees every edge
    await userEvent.click(screen.getByRole('button', { name: /^Account A,/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Account B,/ }))
    await waitFor(() => expect(readout).toHaveTextContent('100%'))
  })

  it('renders the real distributions from the API', async () => {
    renderAt(<VisibilityLabPage />)
    expect(
      await screen.findByText('What actually happens in the dataset'),
    ).toBeInTheDocument()
    // the unfragmentable count comes from the API, not from this page
    const card = (await screen.findByText('Unfragmentable')).closest('div')
      ?.parentElement
    expect(card).toHaveTextContent('2')
    expect(
      screen.getByText('Could never be pushed below 90%'),
    ).toBeInTheDocument()
  })
})

describe('ResultsPage', () => {
  it('says results do not exist yet and shows no numbers', async () => {
    renderAt(<ResultsPage />)
    expect(
      await screen.findByText('Experiment results are not available yet'),
    ).toBeInTheDocument()
    expect(screen.getAllByText('awaiting Phase 5').length).toBeGreaterThan(0)
  })

  it('outlines every planned figure with a how-to-read note', async () => {
    renderAt(<ResultsPage />)
    await screen.findByText('Planned figures')

    const figures = screen.getAllByRole('figure')
    expect(figures).toHaveLength(5)
    for (const figure of figures) {
      expect(within(figure).getByText(/How to read this:/)).toBeInTheDocument()
    }
  })

  it('names the four conditions with distinct line styles', async () => {
    renderAt(<ResultsPage />)
    const legend = await screen.findByLabelText('Experimental conditions')
    expect(within(legend).getByText('Isolated')).toBeInTheDocument()
    expect(within(legend).getByText('FedAvg only')).toBeInTheDocument()
    expect(
      within(legend).getByText('FedAvg + embedding exchange'),
    ).toBeInTheDocument()
    expect(within(legend).getByText('Centralized')).toBeInTheDocument()
  })

  it('does not claim results are missing when the API has some', async () => {
    vi.unstubAllGlobals()
    stubApi({ resultsExist: true })
    renderAt(<ResultsPage />)

    expect(
      await screen.findByText('Results exist but are not charted yet'),
    ).toBeInTheDocument()
    expect(
      screen.queryByText('Experiment results are not available yet'),
    ).not.toBeInTheDocument()
  })
})

describe('MethodologyPage', () => {
  it('renders each frozen definition with its reason and values', async () => {
    renderAt(<MethodologyPage />)
    expect(await screen.findByRole('heading', { name: 'Cutoff' })).toBeInTheDocument()
    expect(screen.getByText(/Only 1,108 rows remain after the cutoff/)).toBeInTheDocument()
    expect(screen.getAllByText('Why it is defined this way')).toHaveLength(2)
    expect(screen.getByText('2022-09-11T00:00:00')).toBeInTheDocument()
    expect(screen.getByText('8')).toBeInTheDocument()
  })

  it('states plainly that this is not a production system', async () => {
    renderAt(<MethodologyPage />)
    expect(
      await screen.findByText(/not a\s+production detection system/),
    ).toBeInTheDocument()
  })
})
