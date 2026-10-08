/**
 * The single source for every series color in the app: one fixed color per
 * institution (8), per experimental condition (4), and per visibility bin (4).
 *
 * Colors come from a validated categorical palette; the exact hexes and their
 * light/dark steps were checked with the data-visualisation palette validator
 * (lightness band, chroma floor, colour-vision-deficiency separation,
 * normal-vision separation, contrast vs the chart surface).
 *
 * IMPORTANT: color is never the only cue. Eight categorical hues cannot be
 * told apart reliably under colour-vision deficiency when any two of them can
 * appear side by side, so every institution mark also carries its label
 * ("I3"), every condition carries a line dash pattern and a label, and every
 * bin carries its name. See README "Frontend - theme and colour system".
 */

export type Mode = 'light' | 'dark'

export type ConditionKey =
  | 'isolated'
  | 'fedavg_only'
  | 'fedavg_embedding'
  | 'centralized'

export type BinKey = '100' | '75' | '50' | 'low'

/** Categorical slots 1-8, light and dark steps. */
const CATEGORICAL: Record<Mode, string[]> = {
  light: [
    '#2a78d6', // blue
    '#eb6834', // orange
    '#1baf7a', // aqua
    '#eda100', // yellow
    '#e87ba4', // magenta
    '#008300', // green
    '#4a3aa7', // violet
    '#e34948', // red
  ],
  dark: [
    '#3987e5',
    '#d95926',
    '#199e70',
    '#c98500',
    '#d55181',
    '#008300',
    '#9085e9',
    '#e66767',
  ],
}

export const N_INSTITUTIONS = 8

/** Fixed color per institution. Always pair it with the institution label. */
export function institutionColor(institution: number, mode: Mode = 'light'): string {
  return CATEGORICAL[mode][institution % N_INSTITUTIONS]
}

export function institutionLabel(institution: number): string {
  return `I${institution}`
}

export const CONDITIONS: Record<
  ConditionKey,
  { label: string; short: string; slot: number; dash: string; description: string }
> = {
  isolated: {
    label: 'Isolated',
    short: 'ISO',
    slot: 0,
    dash: '',
    description: 'Each institution trains and detects alone.',
  },
  fedavg_only: {
    label: 'FedAvg only',
    short: 'FED',
    slot: 1,
    dash: '6 3',
    description:
      'Institutions share model weights. Measures the knowledge advantage.',
  },
  fedavg_embedding: {
    label: 'FedAvg + embedding exchange',
    short: 'FED+E',
    slot: 2,
    dash: '2 3',
    description:
      'Institutions also share embeddings of boundary accounts at detection time. Measures the evidence advantage.',
  },
  centralized: {
    label: 'Centralized',
    short: 'CEN',
    slot: 3,
    dash: '10 4 2 4',
    description: 'Full visibility upper bound.',
  },
}

export const CONDITION_ORDER: ConditionKey[] = [
  'isolated',
  'fedavg_only',
  'fedavg_embedding',
  'centralized',
]

export function conditionColor(condition: ConditionKey, mode: Mode = 'light'): string {
  return CATEGORICAL[mode][CONDITIONS[condition].slot]
}

/**
 * Visibility bins are ordered, so they use a single-hue ordinal ramp rather
 * than categorical hues. The ramp runs light-to-dark on a light surface and
 * dark-to-light on a dark one, so the "more visible" end always reads as the
 * strongest step against its own background.
 */
const BIN_RAMP: Record<Mode, Record<BinKey, string>> = {
  light: { low: '#86b6ef', '50': '#5598e7', '75': '#2a78d6', '100': '#184f95' },
  dark: { low: '#184f95', '50': '#256abf', '75': '#3987e5', '100': '#86b6ef' },
}

export const BIN_ORDER: BinKey[] = ['100', '75', '50', 'low']

export const BIN_LABELS: Record<BinKey, string> = {
  '100': '100% (>= 0.9)',
  '75': '75% (0.65-0.9)',
  '50': '50% (0.45-0.65)',
  low: 'Low (< 0.45)',
}

export function binColor(bin: string, mode: Mode = 'light'): string {
  return BIN_RAMP[mode][(bin as BinKey) ?? 'low'] ?? BIN_RAMP[mode].low
}

/** Chart chrome, matching the CSS custom properties in index.css. */
export const CHROME: Record<Mode, { grid: string; axis: string; ink: string; muted: string; surface: string }> = {
  light: {
    grid: '#e1e0d9',
    axis: '#c3c2b7',
    ink: '#0b0b0b',
    muted: '#898781',
    surface: '#fcfcfb',
  },
  dark: {
    grid: '#2c2c2a',
    axis: '#383835',
    ink: '#ffffff',
    muted: '#898781',
    surface: '#1a1a19',
  },
}

/** Plain-language labels for the research vocabulary used across pages. */
export const GROUP_LABELS: Record<string, string> = {
  hub: 'Hub',
  fragmentable: 'Fragmentable',
  unfragmentable: 'Unfragmentable',
}

export const SPLIT_LABELS: Record<string, string> = {
  train: 'Train (Sept 1-4)',
  val: 'Validation (Sept 5)',
  test: 'Test (Sept 6-7)',
  stress: 'Stress (Sept 8-10)',
}
