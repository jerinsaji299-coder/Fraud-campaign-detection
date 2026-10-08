import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowDown, ArrowUp, Search, X } from 'lucide-react'

import { useCampaigns, useSummary } from '../api/hooks'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { Pill } from '../components/cards'
import { EmptyState, ErrorState, LoadingBlock } from '../components/states'
import { GROUP_LABELS, SPLIT_LABELS } from '../theme'
import type { Campaign, CampaignSortKey } from '../api/types'
import {
  EMPTY_FILTERS,
  activeChips,
  buildCampaignQuery,
  clearFilter,
  hasAnyFilter,
  nextSort,
  type ExplorerFilters,
} from './campaignFilters'

const PAGE_SIZE = 25

const COLUMNS: { key: CampaignSortKey; label: string; numeric?: boolean }[] = [
  { key: 'campaign_id', label: 'ID', numeric: true },
  { key: 'base_type', label: 'Pattern' },
  { key: 'n_txn', label: 'Txns', numeric: true },
  { key: 'n_accounts', label: 'Accounts', numeric: true },
  { key: 'n_banks', label: 'Banks', numeric: true },
  { key: 'start', label: 'Starts' },
  { key: 'duration_h', label: 'Duration', numeric: true },
]

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
      >
        <option value="">Any</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function TriState({
  label,
  value,
  onChange,
  trueLabel,
  falseLabel,
}: {
  label: string
  value: boolean | null
  onChange: (value: boolean | null) => void
  trueLabel: string
  falseLabel: string
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
      {label}
      <select
        value={value === null ? '' : String(value)}
        onChange={(event) =>
          onChange(event.target.value === '' ? null : event.target.value === 'true')
        }
        className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
      >
        <option value="">Any</option>
        <option value="true">{trueLabel}</option>
        <option value="false">{falseLabel}</option>
      </select>
    </label>
  )
}

function formatDuration(hours: number): string {
  if (hours < 24) return `${hours.toFixed(1)}h`
  return `${(hours / 24).toFixed(1)}d`
}

function CampaignRow({ campaign, onOpen }: { campaign: Campaign; onOpen: () => void }) {
  return (
    <tr
      tabIndex={0}
      role="link"
      aria-label={`Open campaign ${campaign.campaign_id}`}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onOpen()
        }
      }}
      className="cursor-pointer border-b border-grid/60 last:border-0 hover:bg-grid/40"
    >
      <td className="py-2 pr-3 font-medium tabular-nums text-ink">
        {campaign.campaign_id}
      </td>
      <td className="py-2 pr-3 text-ink2">{campaign.base_type}</td>
      <td className="py-2 pr-3 text-right tabular-nums text-ink2">{campaign.n_txn}</td>
      <td className="py-2 pr-3 text-right tabular-nums text-ink2">
        {campaign.n_accounts}
      </td>
      <td className="py-2 pr-3 text-right tabular-nums text-ink2">{campaign.n_banks}</td>
      <td className="py-2 pr-3 whitespace-nowrap text-ink2">
        {campaign.start.slice(0, 16).replace('T', ' ')}
      </td>
      <td className="py-2 pr-3 text-right tabular-nums text-ink2">
        {formatDuration(campaign.duration_h)}
      </td>
      <td className="py-2">
        <div className="flex flex-wrap gap-1">
          <Pill>{GROUP_LABELS[campaign.group] ?? campaign.group}</Pill>
          {campaign.split && <Pill>{campaign.split}</Pill>}
          {campaign.eval_ok && <Pill tone="good">evaluable</Pill>}
          {campaign.crosses_cutoff && <Pill tone="warning">censored</Pill>}
          {campaign.shares_train_account && <Pill tone="warning">shared acct</Pill>}
        </div>
      </td>
    </tr>
  )
}

export function CampaignExplorerPage() {
  const navigate = useNavigate()
  const [filters, setFilters] = useState<ExplorerFilters>(EMPTY_FILTERS)
  const [page, setPage] = useState(1)
  const [sort, setSort] = useState<{ sortBy: CampaignSortKey; sortDir: 'asc' | 'desc' }>({
    sortBy: 'campaign_id',
    sortDir: 'asc',
  })

  const summary = useSummary()
  const query = useMemo(
    () => buildCampaignQuery(filters, { page, pageSize: PAGE_SIZE, ...sort }),
    [filters, page, sort],
  )
  const { data, isLoading, error, isPlaceholderData } = useCampaigns(query)

  const update = (patch: Partial<ExplorerFilters>) => {
    setFilters((current) => ({ ...current, ...patch }))
    setPage(1)
  }

  const chips = activeChips(filters)
  const baseTypes = Object.keys(summary.data?.by_base_type ?? {}).sort()

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Campaign explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink2">
          Every laundering{' '}
          <GlossaryTerm term="campaign">campaign</GlossaryTerm> in the dataset.
          Filter, sort, or search by campaign id or account id, then open one to see
          its structure.
        </p>
      </header>

      <div className="space-y-3 rounded-xl border border-grid bg-surface p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex min-w-56 flex-1 flex-col gap-1 text-xs font-medium text-ink2">
            Search
            <span className="relative">
              <Search
                className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
                aria-hidden="true"
              />
              <input
                type="search"
                value={filters.search}
                onChange={(event) => update({ search: event.target.value })}
                placeholder="Campaign id, or part of an account id"
                className="w-full rounded-lg border border-grid bg-surface py-1.5 pl-8 pr-2 text-sm text-ink placeholder:text-muted"
              />
            </span>
          </label>

          <Select
            label="Pattern"
            value={filters.base_type}
            onChange={(value) => update({ base_type: value })}
            options={baseTypes.map((type) => ({ value: type, label: type }))}
          />
          <Select
            label="Split"
            value={filters.split}
            onChange={(value) => update({ split: value })}
            options={Object.entries(SPLIT_LABELS).map(([value, label]) => ({
              value,
              label,
            }))}
          />
          <Select
            label="Group"
            value={filters.group}
            onChange={(value) => update({ group: value })}
            options={Object.entries(GROUP_LABELS).map(([value, label]) => ({
              value,
              label,
            }))}
          />
          <TriState
            label="Evaluable"
            value={filters.eval_ok}
            onChange={(value) => update({ eval_ok: value })}
            trueLabel="Evaluable"
            falseLabel="Too small"
          />
          <TriState
            label="Cutoff"
            value={filters.crosses_cutoff}
            onChange={(value) => update({ crosses_cutoff: value })}
            trueLabel="Censored"
            falseLabel="Fully observed"
          />
          <TriState
            label="Shared account"
            value={filters.shares_train_account}
            onChange={(value) => update({ shares_train_account: value })}
            trueLabel="Shares a train account"
            falseLabel="No shared account"
          />
        </div>

        {chips.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 border-t border-grid pt-3">
            {chips.map((chip) => (
              <button
                key={chip.key}
                type="button"
                onClick={() => {
                  setFilters((current) => clearFilter(current, chip.key))
                  setPage(1)
                }}
                className="inline-flex items-center gap-1 rounded-full border border-grid px-2 py-0.5 text-xs text-ink2 hover:text-ink"
              >
                {chip.label}
                <X className="h-3 w-3" aria-hidden="true" />
                <span className="sr-only">Remove filter</span>
              </button>
            ))}
            <button
              type="button"
              onClick={() => {
                setFilters(EMPTY_FILTERS)
                setPage(1)
              }}
              className="text-xs font-medium text-ink underline underline-offset-2"
            >
              Clear all
            </button>
          </div>
        )}
      </div>

      {isLoading && <LoadingBlock label="Loading campaigns" />}
      {error && <ErrorState error={error} />}

      {data && (
        <>
          <p className="text-sm text-ink2" aria-live="polite">
            <strong className="text-ink">{data.total.toLocaleString()}</strong>{' '}
            {data.total === 1 ? 'campaign' : 'campaigns'}
            {hasAnyFilter(filters) ? ' match these filters' : ' in the dataset'}
            {data.pages > 1 && ` · page ${data.page} of ${data.pages}`}
          </p>

          {data.total === 0 ? (
            <EmptyState
              title="No campaigns match these filters"
              description="Try removing a filter chip above, or clear them all."
            />
          ) : (
            <div
              className={`overflow-x-auto rounded-xl border border-grid bg-surface ${
                isPlaceholderData ? 'opacity-60' : ''
              }`}
            >
              <table className="w-full text-sm">
                <caption className="sr-only">
                  Campaigns, sortable by column. Select a row to open a campaign.
                </caption>
                <thead>
                  <tr className="border-b border-grid text-xs uppercase tracking-wide text-muted">
                    {COLUMNS.map((column) => {
                      const active = sort.sortBy === column.key
                      const Icon = sort.sortDir === 'asc' ? ArrowUp : ArrowDown
                      return (
                        <th
                          key={column.key}
                          scope="col"
                          aria-sort={
                            active
                              ? sort.sortDir === 'asc'
                                ? 'ascending'
                                : 'descending'
                              : 'none'
                          }
                          className={`py-2 pr-3 font-medium ${
                            column.numeric ? 'text-right' : 'text-left'
                          }`}
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setSort((current) => nextSort(current, column.key))
                              setPage(1)
                            }}
                            className={`inline-flex items-center gap-1 ${
                              active ? 'text-ink' : 'hover:text-ink'
                            }`}
                          >
                            {column.label}
                            {active && <Icon className="h-3 w-3" aria-hidden="true" />}
                          </button>
                        </th>
                      )
                    })}
                    <th scope="col" className="py-2 text-left font-medium">
                      Flags
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((campaign) => (
                    <CampaignRow
                      key={campaign.campaign_id}
                      campaign={campaign}
                      onOpen={() => navigate(`/campaigns/${campaign.campaign_id}`)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data.pages > 1 && (
            <div className="flex items-center justify-between">
              <button
                type="button"
                disabled={data.page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="rounded-lg border border-grid px-3 py-1.5 text-sm text-ink disabled:opacity-40"
              >
                Previous
              </button>
              <span className="text-sm tabular-nums text-ink2">
                Page {data.page} of {data.pages}
              </span>
              <button
                type="button"
                disabled={data.page >= data.pages}
                onClick={() => setPage((p) => p + 1)}
                className="rounded-lg border border-grid px-3 py-1.5 text-sm text-ink disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
