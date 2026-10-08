/**
 * Pure filter logic for the Campaign Explorer, kept out of the component so
 * it can be tested directly.
 */

import type { CampaignQuery, CampaignSortKey } from '../api/types'
import { GROUP_LABELS, SPLIT_LABELS } from '../theme'

export interface ExplorerFilters {
  search: string
  base_type: string
  split: string
  group: string
  eval_ok: boolean | null
  crosses_cutoff: boolean | null
  shares_train_account: boolean | null
}

export const EMPTY_FILTERS: ExplorerFilters = {
  search: '',
  base_type: '',
  split: '',
  group: '',
  eval_ok: null,
  crosses_cutoff: null,
  shares_train_account: null,
}

export type FilterKey = keyof ExplorerFilters

export interface FilterChip {
  key: FilterKey
  label: string
}

const BOOL_LABELS: Record<string, [string, string]> = {
  eval_ok: ['Evaluable', 'Not evaluable'],
  crosses_cutoff: ['Censored by cutoff', 'Fully observed'],
  shares_train_account: ['Shares a train account', 'No shared account'],
}

/** Which filters are set, as removable chips. */
export function activeChips(filters: ExplorerFilters): FilterChip[] {
  const chips: FilterChip[] = []

  if (filters.search.trim()) {
    chips.push({ key: 'search', label: `Search: "${filters.search.trim()}"` })
  }
  if (filters.base_type) {
    chips.push({ key: 'base_type', label: `Type: ${filters.base_type}` })
  }
  if (filters.split) {
    chips.push({ key: 'split', label: SPLIT_LABELS[filters.split] ?? filters.split })
  }
  if (filters.group) {
    chips.push({ key: 'group', label: GROUP_LABELS[filters.group] ?? filters.group })
  }
  for (const key of ['eval_ok', 'crosses_cutoff', 'shares_train_account'] as const) {
    const value = filters[key]
    if (value !== null) {
      chips.push({ key, label: BOOL_LABELS[key][value ? 0 : 1] })
    }
  }

  return chips
}

export function clearFilter(
  filters: ExplorerFilters,
  key: FilterKey,
): ExplorerFilters {
  const blank = EMPTY_FILTERS[key]
  return { ...filters, [key]: blank }
}

export function hasAnyFilter(filters: ExplorerFilters): boolean {
  return activeChips(filters).length > 0
}

/** Only send parameters the user actually set, so the API URL stays clean
 * and the query cache key stays stable. */
export function buildCampaignQuery(
  filters: ExplorerFilters,
  options: {
    page: number
    pageSize: number
    sortBy: CampaignSortKey
    sortDir: 'asc' | 'desc'
  },
): CampaignQuery {
  const query: CampaignQuery = {
    page: options.page,
    page_size: options.pageSize,
    sort_by: options.sortBy,
    sort_dir: options.sortDir,
  }

  const search = filters.search.trim()
  if (search) query.search = search
  if (filters.base_type) query.base_type = filters.base_type
  if (filters.split) query.split = filters.split
  if (filters.group) query.group = filters.group
  if (filters.eval_ok !== null) query.eval_ok = filters.eval_ok
  if (filters.crosses_cutoff !== null) query.crosses_cutoff = filters.crosses_cutoff
  if (filters.shares_train_account !== null) {
    query.shares_train_account = filters.shares_train_account
  }

  return query
}

/** Clicking the active sort column flips direction; a new column starts ascending. */
export function nextSort(
  current: { sortBy: CampaignSortKey; sortDir: 'asc' | 'desc' },
  column: CampaignSortKey,
): { sortBy: CampaignSortKey; sortDir: 'asc' | 'desc' } {
  if (current.sortBy !== column) return { sortBy: column, sortDir: 'asc' }
  return { sortBy: column, sortDir: current.sortDir === 'asc' ? 'desc' : 'asc' }
}
