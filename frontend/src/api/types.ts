/**
 * TypeScript mirrors of the pydantic response models in
 * backend/app/schemas.py. Keep the two in sync: if an endpoint changes
 * shape, change it here in the same step.
 */

export interface Health {
  status: string
  artifacts_dir: string
  artifacts: Record<string, boolean>
  core_artifacts_available: boolean
  results_available: boolean
  /** Core artifacts the server could not find; empty when all are present. */
  missing_artifacts: string[]
  /** Set when the artifacts sit one directory too deep. */
  nested_artifacts_dir: string | null
  /** Human-readable explanation of whatever is wrong, if anything is. */
  hint: string | null
}

export interface RowCounts {
  total: number
  pre_cutoff: number
  post_cutoff: number
}

export interface DateRange {
  min: string
  max: string
}

export interface LaunderingCounts {
  rows: number
  in_campaign: number
  unassigned: number
}

export interface CampaignCounts {
  total: number
  eval_ok: number
  pattern_transactions: number
  reassignable: number
}

export interface SplitCounts {
  campaigns: number
  eval_ok: number
  crosses_cutoff: number
  shares_train_account: number
}

export interface CutoffInfo {
  timestamp: string
  rows_excluded: number
  laundering_rows_excluded: number
  laundering_share_excluded: number
  laundering_share_overall: number
}

export interface Summary {
  generated_at: string
  source: Record<string, string>
  rows: RowCounts
  date_range: DateRange
  banks: { total: number }
  laundering: LaunderingCounts
  campaigns: CampaignCounts
  by_base_type: Record<string, number>
  by_split: Record<string, SplitCounts>
  by_group: Record<string, number>
  by_group_eval_ok: Record<string, number>
  cutoff: CutoffInfo
  transactions_per_day: Record<string, number>
  campaigns_starting_per_day: Record<string, number>
  institutions: { k: number }
  visibility: { targets: number[]; seeds: number[]; bins: string[] }
}

export interface Campaign {
  campaign_id: number
  base_type: string
  n_txn: number
  n_accounts: number
  n_banks: number
  start: string
  end: string
  duration_h: number
  crosses_cutoff: boolean
  deadline: string
  eval_ok: boolean
  split: string | null
  group: string
  shares_train_account: boolean
  reassignable: boolean
}

export interface CampaignList {
  items: Campaign[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface CampaignTransaction {
  txn_index: number
  campaign_id: number
  timestamp: string
  from_bank: number
  to_bank: number
  src: string
  dst: string
  amount_paid: number
  payment_currency: string
  amount_received: number
  receiving_currency: string
  payment_format: string
  is_laundering: number
  src_natural_institution: number
  dst_natural_institution: number
}

export interface CampaignAccount {
  account: string
  natural_institution: number
}

export interface CampaignDetail {
  campaign: Campaign
  transactions: CampaignTransaction[]
  accounts: CampaignAccount[]
}

export interface AccountAssignment {
  account: string
  institution: number
  reassigned: boolean
}

export interface InstitutionView {
  institution: number
  visible_txn_indices: number[]
  n_visible: number
}

export interface CampaignVisibility {
  campaign_id: number
  seed: number
  target: number
  achieved_visibility: number
  bin: string
  campaign_reassigned: boolean
  n_transactions: number
  accounts: AccountAssignment[]
  institutions: InstitutionView[]
}

export interface CampaignVisibilityPoint {
  campaign_id: number
  achieved_visibility: number
  bin: string
  group: string
  reassigned: boolean
}

export interface TargetDistribution {
  target: number
  campaigns: CampaignVisibilityPoint[]
  bin_counts: Record<string, number>
  bin_counts_by_group: Record<string, Record<string, number>>
}

export interface VisibilityDistribution {
  seed: number
  targets: number[]
  bins: string[]
  per_target: TargetDistribution[]
  unfragmentable_campaigns: number
}

export interface MethodologyEntry {
  key: string
  title: string
  definition: string
  value: Record<string, unknown>
  reason: string
}

export interface Methodology {
  entries: MethodologyEntry[]
  n_entries: number
}

/** Filters accepted by GET /api/campaigns. */
export interface CampaignQuery {
  page?: number
  page_size?: number
  base_type?: string
  split?: string
  group?: string
  eval_ok?: boolean
  crosses_cutoff?: boolean
  shares_train_account?: boolean
  search?: string
  sort_by?: CampaignSortKey
  sort_dir?: 'asc' | 'desc'
}

export type CampaignSortKey =
  | 'campaign_id'
  | 'base_type'
  | 'n_txn'
  | 'n_accounts'
  | 'n_banks'
  | 'start'
  | 'end'
  | 'duration_h'

// ---------------------------------------------------------------------------
// the live demo panel (/api/demo/*)
// ---------------------------------------------------------------------------

export interface DemoWindowMeta {
  detection_time: string
  n_edges: number
  n_accounts: number
  n_laundering_edges: number
  n_campaign_edges: number
}

export interface DemoManifest {
  visibility_target: number
  seed: number
  lookback_h: number
  split: string
  k_institutions: number
  institutions_with_a_model: number[]
  demo_campaign: {
    campaign_id: number
    accounts: string[]
    institutions_holding: number[]
    achieved_visibility: number | null
  }
  selection_rule: string[]
  selection_disclaimer: string | null
  selection_audit: {
    rank_chosen: number | null
    n_candidates: number | null
    n_skipped: number
  }
  windows: DemoWindowMeta[]
  detection_times: string[]
  primary_points: string[]
  taus: Record<string, Record<string, number | null>>
  match_rule: { min_account_fraction?: number; min_accounts?: number }
  suggested_settings: {
    detection_time: string
    institution: number
    own_clusters: number
    averaged_clusters: number
    coverage: number | null
    hours_before_deadline: number | null
  }[]
  setting_outcomes: Record<string, number>
  suggested_settings_note: string | null
  models_mb: number | null
  modes: string[]
  development_only: string | null
  generated: string | null
}

export interface DemoClusterMatch {
  n_cluster_accounts: number
  n_campaign_accounts: number
  overlap: number
  purity: number
  coverage_accounts: number
  coverage_fraction: number
  min_fraction: number
  min_accounts: number
  matches: boolean
}

export interface DemoCluster {
  accounts: string[]
  n_accounts: number
  kind: string
  hit_campaigns: number[]
  matches_demo_campaign: boolean
  vs_demo_campaign: DemoClusterMatch | null
  institutions: number[]
}

export interface DemoRun {
  detection_time: string
  institution: number
  mode: string
  models_used?: number[]
  n_models_used?: number
  tau: number
  tau_source?: string
  operating_point?: string
  n_edges_in_window: number
  n_edges_in_view: number
  n_flagged_edges: number
  flagged_edges: {
    src: string
    dst: string
    score: number
    ts: string
    is_laundering: number
    campaign_id: number
  }[]
  flagged_edges_truncated?: boolean
  clusters: DemoCluster[]
  n_clusters: number
  alert_count: number
  cluster_kinds?: Record<string, number>
  demo_campaign_id?: number | null
  demo_campaign_detected?: boolean
  best_purity_vs_demo_campaign?: number | null
  best_coverage_vs_demo_campaign?: number | null
  lead_time?: {
    detected: boolean
    deadline: string | null
    campaign_end: string | null
    hours_before_deadline: number | null
    hours_before_end: number | null
    beat_the_deadline: boolean | null
  } | null
  empty_reason?: string
  latency_ms: {
    featurise?: number
    featurise_was_cached?: boolean
    predict?: number
    clusters?: number
    total?: number
  }
}

export interface DemoAggregate {
  available: boolean
  reason: string | null
  rows: Record<string, unknown>[]
  source?: string
}
