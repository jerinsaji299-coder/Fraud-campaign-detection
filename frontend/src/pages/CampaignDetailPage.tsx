import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Eye, Radar } from 'lucide-react'

import { useCampaign, useCampaignVisibility, useSummary } from '../api/hooks'
import { CampaignGraph } from '../components/CampaignGraph'
import { Timeline } from '../components/Timeline'
import { GlossaryTerm } from '../components/GlossaryTerm'
import { InstitutionBadge, Pill } from '../components/cards'
import { EmptyState, ErrorState, LoadingBlock, Skeleton } from '../components/states'
import { BIN_LABELS, GROUP_LABELS, SPLIT_LABELS, binColor } from '../theme'
import { useThemeMode } from '../lib/themeContext'
import type { BinKey } from '../theme'
import type { Campaign } from '../api/types'

const SEEDS = [0, 1, 2, 3, 4]
const TARGETS = [1.0, 0.75, 0.5, 0.25]
const CUTOFF = '2022-09-11T00:00:00'

function VisibilityBadge({ achieved, bin }: { achieved: number; bin: string }) {
  const { mode } = useThemeMode()
  return (
    <span className="inline-flex items-center gap-2 rounded-lg border border-grid px-2.5 py-1.5">
      <span
        aria-hidden="true"
        className="h-3 w-3 rounded-sm"
        style={{ background: binColor(bin, mode) }}
      />
      <span className="text-sm font-semibold tabular-nums text-ink">
        {Math.round(achieved * 100)}%
      </span>
      <span className="text-xs text-ink2">
        bin {BIN_LABELS[bin as BinKey] ?? bin}
      </span>
    </span>
  )
}

function MetadataSidebar({ campaign }: { campaign: Campaign }) {
  const rows: { label: string; value: React.ReactNode; note?: string }[] = [
    { label: 'Pattern', value: campaign.base_type },
    {
      label: 'Group',
      value: GROUP_LABELS[campaign.group] ?? campaign.group,
      note:
        campaign.group === 'hub'
          ? 'Hub-shaped: one account touches every transaction, so its bank always sees the whole campaign. Used as a control.'
          : campaign.group === 'unfragmentable'
            ? 'Fragmentable in shape, but too small to push below 90% visibility in practice. Grouped with the control.'
            : 'Its accounts can be spread across banks so that none sees the whole thing.',
    },
    {
      label: 'Split',
      value: campaign.split ? (SPLIT_LABELS[campaign.split] ?? campaign.split) : '—',
      note: 'Which part of the study this campaign belongs to, decided by the day it starts.',
    },
    { label: 'Transactions', value: campaign.n_txn },
    { label: 'Accounts', value: campaign.n_accounts },
    { label: 'Banks', value: campaign.n_banks },
    {
      label: 'Duration',
      value:
        campaign.duration_h < 24
          ? `${campaign.duration_h.toFixed(1)} hours`
          : `${(campaign.duration_h / 24).toFixed(1)} days`,
    },
    {
      label: 'Evaluated',
      value: campaign.eval_ok ? 'Yes' : 'No — too small',
      note: campaign.eval_ok
        ? 'At least 3 transactions and 3 accounts, so discovery lead time is meaningful.'
        : 'Fewer than 3 transactions or accounts, so lead time would be meaningless. Excluded from scoring.',
    },
    {
      label: 'Finished before cutoff',
      value: campaign.crosses_cutoff ? 'No — censored' : 'Yes',
      note: campaign.crosses_cutoff
        ? 'This campaign was still running at the cutoff, so its later transactions are not available as evidence.'
        : 'The whole campaign is observable within the study window.',
    },
    {
      label: 'Shares a train account',
      value: campaign.shares_train_account ? 'Yes' : 'No',
      note: campaign.shares_train_account
        ? 'An account here also appears in a training campaign, so a model might recognise it rather than generalise. Results are reported with and without these.'
        : undefined,
    },
  ]

  return (
    <aside className="space-y-3 rounded-xl border border-grid bg-surface p-4">
      <h2 className="text-sm font-semibold text-ink">About this campaign</h2>
      <dl className="space-y-3">
        {rows.map((row) => (
          <div key={row.label}>
            <dt className="text-xs font-medium uppercase tracking-wide text-muted">
              {row.label}
            </dt>
            <dd className="text-sm font-medium text-ink">{row.value}</dd>
            {row.note && (
              <dd className="mt-0.5 text-xs leading-snug text-ink2">{row.note}</dd>
            )}
          </div>
        ))}
      </dl>
    </aside>
  )
}

export function CampaignDetailPage() {
  const { campaignId } = useParams()
  const id = Number(campaignId)

  const [seed, setSeed] = useState(0)
  /** null = the natural bank split; otherwise a visibility target. */
  const [target, setTarget] = useState<number | null>(null)
  /** null = "not scrubbed yet", which means show the whole campaign. */
  const [scrubbedTo, setScrubbedTo] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [focusInstitution, setFocusInstitution] = useState<number | null>(null)
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null)

  const detail = useCampaign(Number.isFinite(id) ? id : undefined)
  const visibility = useCampaignVisibility(Number.isFinite(id) ? id : undefined, seed, target)
  const summary = useSummary()
  const cutoff = summary.data?.cutoff.timestamp ?? CUTOFF

  const assignment = useMemo(() => {
    if (target !== null && visibility.data) {
      return Object.fromEntries(
        visibility.data.accounts.map((account) => [account.account, account.institution]),
      )
    }
    if (detail.data) {
      return Object.fromEntries(
        detail.data.accounts.map((account) => [
          account.account,
          account.natural_institution,
        ]),
      )
    }
    return {}
  }, [target, visibility.data, detail.data])

  const institutionsPresent = useMemo(
    () => [...new Set(Object.values(assignment))].sort((a, b) => a - b),
    [assignment],
  )

  // Changing the view can move every account, so a bank that was being
  // inspected may no longer hold anything. Derive the effective focus rather
  // than correcting state in an effect, which would flash the stale view.
  const focus =
    focusInstitution !== null && institutionsPresent.includes(focusInstitution)
      ? focusInstitution
      : null

  if (detail.isLoading) return <LoadingBlock label="Loading campaign" />
  if (detail.error) return <ErrorState error={detail.error} />
  if (!detail.data) return null

  const { campaign, transactions } = detail.data
  // Before the user touches the slider, the whole campaign is shown.
  const revealedUpTo = scrubbedTo ?? transactions.length - 1

  const visibleCount =
    focus === null
      ? transactions.length
      : transactions.filter(
          (txn) => assignment[txn.src] === focus || assignment[txn.dst] === focus,
        ).length

  const selectedTransactions = selectedAccount
    ? transactions.filter(
        (txn) => txn.src === selectedAccount || txn.dst === selectedAccount,
      )
    : []

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <div>
        <Link
          to="/campaigns"
          className="inline-flex items-center gap-1.5 text-sm text-ink2 hover:text-ink"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
          All campaigns
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold text-ink">
            Campaign {campaign.campaign_id}
          </h1>
          <Pill>{campaign.base_type}</Pill>
          <Pill>{GROUP_LABELS[campaign.group] ?? campaign.group}</Pill>
          {campaign.split && <Pill>{campaign.split}</Pill>}
          {campaign.crosses_cutoff && <Pill tone="warning">censored</Pill>}
        </div>
      </div>

      {/* visibility controls */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-grid bg-surface p-4">
        <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
          View
          <select
            value={target === null ? 'natural' : String(target)}
            onChange={(event) =>
              setTarget(
                event.target.value === 'natural' ? null : Number(event.target.value),
              )
            }
            className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink"
          >
            <option value="natural">Natural banks</option>
            {TARGETS.map((value) => (
              <option key={value} value={value}>
                Target {Math.round(value * 100)}% visibility
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs font-medium text-ink2">
          Seed
          <select
            value={seed}
            onChange={(event) => setSeed(Number(event.target.value))}
            disabled={target === null}
            className="rounded-lg border border-grid bg-surface px-2 py-1.5 text-sm text-ink disabled:opacity-50"
          >
            {SEEDS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <div className="flex-1">
          {target === null ? (
            <p className="text-xs leading-snug text-ink2">
              Accounts sit with the institution that owns their real bank. Pick a
              target to see the same campaign deliberately split so that no single
              bank sees much of it.
            </p>
          ) : visibility.isLoading ? (
            <Skeleton className="h-9 w-48" />
          ) : visibility.error ? (
            <ErrorState error={visibility.error} />
          ) : visibility.data ? (
            <div className="flex flex-wrap items-center gap-3">
              <VisibilityBadge
                achieved={visibility.data.achieved_visibility}
                bin={visibility.data.bin}
              />
              {!visibility.data.campaign_reassigned && (
                <span className="text-xs text-ink2">
                  This campaign is never reassigned (hub-shaped or too small), so it
                  keeps its natural split at every target.
                </span>
              )}
            </div>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div className="space-y-4">
          {/* bank's-eye view */}
          <div className="rounded-xl border border-grid bg-surface p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-ink">
                <Eye className="h-4 w-4" aria-hidden="true" />
                Bank&rsquo;s-eye view
              </h2>
              <p className="text-sm text-ink2">
                {focus === null ? (
                  <>Showing everything, as no single bank ever could.</>
                ) : (
                  <>
                    This bank sees{' '}
                    <strong className="text-ink">{visibleCount}</strong> of{' '}
                    {transactions.length} transactions.
                  </>
                )}
              </p>
            </div>
            <div className="mt-3 flex flex-wrap gap-2" role="tablist" aria-label="Institution">
              <button
                type="button"
                role="tab"
                aria-selected={focus === null}
                onClick={() => setFocusInstitution(null)}
                className={`rounded-lg border px-2.5 py-1 text-sm font-medium ${
                  focus === null
                    ? 'border-axis bg-grid text-ink'
                    : 'border-grid text-ink2 hover:text-ink'
                }`}
              >
                All banks
              </button>
              {institutionsPresent.map((institution) => (
                <button
                  key={institution}
                  type="button"
                  role="tab"
                  aria-selected={focus === institution}
                  onClick={() => setFocusInstitution(institution)}
                  className={`rounded-lg border px-1.5 py-1 ${
                    focus === institution
                      ? 'border-axis bg-grid'
                      : 'border-grid hover:bg-grid/40'
                  }`}
                >
                  <InstitutionBadge institution={institution} className="border-0" />
                </button>
              ))}
            </div>

            <div className="mt-3">
              <CampaignGraph
                transactions={transactions}
                assignment={assignment}
                revealedUpTo={revealedUpTo}
                focusInstitution={focus}
                baseType={campaign.base_type}
                onSelectAccount={setSelectedAccount}
                selectedAccount={selectedAccount}
              />
            </div>
          </div>

          <Timeline
            transactions={transactions}
            value={revealedUpTo}
            onChange={setScrubbedTo}
            playing={playing}
            onPlayingChange={setPlaying}
            deadline={campaign.deadline}
            cutoff={cutoff}
          />

          {/* detection panel */}
          <div className="rounded-xl border border-grid bg-surface p-4">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-ink">
              <Radar className="h-4 w-4" aria-hidden="true" />
              Detection
            </h2>
            <div className="mt-3">
              <EmptyState
                title="No detection results for this campaign yet"
                description={
                  <>
                    Once the experiments run, each condition&rsquo;s detection time and{' '}
                    <GlossaryTerm term="lead time">lead time</GlossaryTerm> will appear
                    here, marked on the timeline above. Nothing is shown until then.
                  </>
                }
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <MetadataSidebar campaign={campaign} />

          {selectedAccount && (
            <div className="rounded-xl border border-axis bg-surface p-4">
              <h2 className="text-sm font-semibold text-ink">Account</h2>
              <p className="mt-1 break-all font-mono text-xs text-ink2">
                {selectedAccount}
              </p>
              <div className="mt-2">
                {assignment[selectedAccount] !== undefined && (
                  <InstitutionBadge institution={assignment[selectedAccount]} />
                )}
              </div>
              <p className="mt-3 text-xs text-ink2">
                In {selectedTransactions.length} of this campaign&rsquo;s{' '}
                {transactions.length} transactions.
              </p>
              <button
                type="button"
                onClick={() => setSelectedAccount(null)}
                className="mt-3 text-xs font-medium text-ink underline underline-offset-2"
              >
                Clear selection
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
