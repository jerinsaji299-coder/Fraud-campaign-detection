import { useQuery } from '@tanstack/react-query'

import { apiGet } from './client'
import type {
  CampaignDetail,
  CampaignList,
  CampaignQuery,
  CampaignVisibility,
  Health,
  Methodology,
  Summary,
  VisibilityDistribution,
} from './types'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<Health>('/api/health'),
    refetchInterval: 30_000,
    retry: false,
  })
}

export function useSummary() {
  return useQuery({
    queryKey: ['summary'],
    queryFn: () => apiGet<Summary>('/api/summary'),
  })
}

export function useCampaigns(query: CampaignQuery) {
  return useQuery({
    queryKey: ['campaigns', query],
    queryFn: () => apiGet<CampaignList>('/api/campaigns', { ...query }),
    placeholderData: (previous) => previous,
  })
}

export function useCampaign(campaignId: number | undefined) {
  return useQuery({
    queryKey: ['campaign', campaignId],
    queryFn: () => apiGet<CampaignDetail>(`/api/campaigns/${campaignId}`),
    enabled: campaignId !== undefined && Number.isFinite(campaignId),
  })
}

/** Per-account institutions for one campaign under one (seed, target). */
export function useCampaignVisibility(
  campaignId: number | undefined,
  seed: number,
  target: number | null,
) {
  return useQuery({
    queryKey: ['campaign-visibility', campaignId, seed, target],
    queryFn: () =>
      apiGet<CampaignVisibility>(`/api/campaigns/${campaignId}/visibility`, {
        seed,
        target,
      }),
    // target === null means "show the natural bank split", which needs no request
    enabled:
      campaignId !== undefined && Number.isFinite(campaignId) && target !== null,
  })
}

export function useVisibilityDistribution(seed: number) {
  return useQuery({
    queryKey: ['visibility-distribution', seed],
    queryFn: () =>
      apiGet<VisibilityDistribution>('/api/visibility/distribution', { seed }),
  })
}

export function useMethodology() {
  return useQuery({
    queryKey: ['methodology'],
    queryFn: () => apiGet<Methodology>('/api/methodology'),
    staleTime: Infinity,
  })
}

/** 404s until Phase 5 writes results; callers render an empty state. */
export function useResultsSummary() {
  return useQuery({
    queryKey: ['results-summary'],
    queryFn: () => apiGet<{ items: unknown[]; total: number }>('/api/results/summary'),
    retry: false,
  })
}
