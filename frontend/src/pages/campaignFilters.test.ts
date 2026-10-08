import { describe, expect, it } from 'vitest'

import {
  EMPTY_FILTERS,
  activeChips,
  buildCampaignQuery,
  clearFilter,
  hasAnyFilter,
  nextSort,
} from './campaignFilters'

const OPTIONS = {
  page: 1,
  pageSize: 25,
  sortBy: 'campaign_id' as const,
  sortDir: 'asc' as const,
}

describe('buildCampaignQuery', () => {
  it('sends only paging and sorting when nothing is filtered', () => {
    expect(buildCampaignQuery(EMPTY_FILTERS, OPTIONS)).toEqual({
      page: 1,
      page_size: 25,
      sort_by: 'campaign_id',
      sort_dir: 'asc',
    })
  })

  it('includes each filter that is set', () => {
    const query = buildCampaignQuery(
      {
        ...EMPTY_FILTERS,
        search: '  1_A  ',
        base_type: 'CYCLE',
        split: 'test',
        group: 'fragmentable',
        eval_ok: true,
        crosses_cutoff: false,
      },
      OPTIONS,
    )

    expect(query.search).toBe('1_A')
    expect(query.base_type).toBe('CYCLE')
    expect(query.split).toBe('test')
    expect(query.group).toBe('fragmentable')
    expect(query.eval_ok).toBe(true)
    expect(query.crosses_cutoff).toBe(false)
    expect(query.shares_train_account).toBeUndefined()
  })

  it('keeps a false boolean rather than dropping it as empty', () => {
    const query = buildCampaignQuery(
      { ...EMPTY_FILTERS, eval_ok: false },
      OPTIONS,
    )
    expect(query.eval_ok).toBe(false)
  })

  it('omits a whitespace-only search', () => {
    expect(
      buildCampaignQuery({ ...EMPTY_FILTERS, search: '   ' }, OPTIONS).search,
    ).toBeUndefined()
  })
})

describe('activeChips', () => {
  it('is empty for untouched filters', () => {
    expect(activeChips(EMPTY_FILTERS)).toEqual([])
    expect(hasAnyFilter(EMPTY_FILTERS)).toBe(false)
  })

  it('labels each active filter, including false booleans', () => {
    const chips = activeChips({
      ...EMPTY_FILTERS,
      base_type: 'CYCLE',
      crosses_cutoff: false,
      shares_train_account: true,
    })

    expect(chips.map((chip) => chip.key)).toEqual([
      'base_type',
      'crosses_cutoff',
      'shares_train_account',
    ])
    expect(chips[0].label).toBe('Type: CYCLE')
    expect(chips[1].label).toBe('Fully observed')
    expect(chips[2].label).toBe('Shares a train account')
  })
})

describe('clearFilter', () => {
  it('resets one filter and leaves the others alone', () => {
    const filters = { ...EMPTY_FILTERS, base_type: 'CYCLE', eval_ok: true }
    const cleared = clearFilter(filters, 'base_type')

    expect(cleared.base_type).toBe('')
    expect(cleared.eval_ok).toBe(true)
  })

  it('resets a boolean filter back to null, not false', () => {
    const cleared = clearFilter({ ...EMPTY_FILTERS, eval_ok: false }, 'eval_ok')
    expect(cleared.eval_ok).toBeNull()
    expect(hasAnyFilter(cleared)).toBe(false)
  })
})

describe('nextSort', () => {
  it('starts a new column ascending', () => {
    expect(nextSort({ sortBy: 'campaign_id', sortDir: 'desc' }, 'n_txn')).toEqual({
      sortBy: 'n_txn',
      sortDir: 'asc',
    })
  })

  it('flips direction on the active column', () => {
    expect(nextSort({ sortBy: 'n_txn', sortDir: 'asc' }, 'n_txn')).toEqual({
      sortBy: 'n_txn',
      sortDir: 'desc',
    })
  })
})
