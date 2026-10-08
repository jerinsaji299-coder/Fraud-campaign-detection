/**
 * The ONE research metric the frontend is allowed to compute, and only for
 * the Visibility Lab toy, where the user invents their own tiny scenario that
 * the backend has never seen.
 *
 * This is a direct mirror of `visibility_of` in
 * backend/src/fraudcamp/visibility.py:
 *
 *   visibility = max over institutions of
 *                (fraction of the campaign's transactions that institution sees)
 *
 * An institution sees a transaction if the sender's OR the receiver's account
 * belongs to it. Every real number shown anywhere else in the app comes from
 * the backend.
 */

export interface Edge {
  src: string
  dst: string
}

export type Assignment = Record<string, number>

export function visibilityOf(
  edges: Edge[],
  assignment: Assignment,
  k: number,
): number {
  if (edges.length === 0) return 0

  let best = 0
  for (let institution = 0; institution < k; institution += 1) {
    let seen = 0
    for (const edge of edges) {
      if (
        assignment[edge.src] === institution ||
        assignment[edge.dst] === institution
      ) {
        seen += 1
      }
    }
    const fraction = seen / edges.length
    if (fraction > best) best = fraction
  }
  return best
}

/** Mirrors `bin_of` in backend/src/fraudcamp/visibility.py. */
export function binOf(achieved: number): '100' | '75' | '50' | 'low' {
  if (achieved >= 0.9) return '100'
  if (achieved >= 0.65) return '75'
  if (achieved >= 0.45) return '50'
  return 'low'
}

/** Which transactions a given institution can see, by index into `edges`. */
export function visibleEdgeIndices(
  edges: Edge[],
  assignment: Assignment,
  institution: number,
): number[] {
  const indices: number[] = []
  edges.forEach((edge, index) => {
    if (
      assignment[edge.src] === institution ||
      assignment[edge.dst] === institution
    ) {
      indices.push(index)
    }
  })
  return indices
}
