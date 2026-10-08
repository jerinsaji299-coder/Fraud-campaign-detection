import { describe, expect, it } from 'vitest'

import { binOf, visibilityOf, visibleEdgeIndices, type Edge } from './visibility'

describe('visibilityOf', () => {
  it('is 1.0 for a chain A->B->C however the ends are placed', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'B', dst: 'C' },
    ]
    // B sits on both transactions, so B's institution sees the whole chain
    expect(visibilityOf(edges, { A: 0, B: 1, C: 2 }, 3)).toBe(1)
    expect(visibilityOf(edges, { A: 2, B: 0, C: 1 }, 3)).toBe(1)
  })

  it('is 1.0 for a fan-out, because the hub bank sees every transaction', () => {
    const edges: Edge[] = [
      { src: 'H', dst: 'L1' },
      { src: 'H', dst: 'L2' },
      { src: 'H', dst: 'L3' },
    ]
    expect(visibilityOf(edges, { H: 0, L1: 1, L2: 2, L3: 3 }, 4)).toBe(1)
  })

  it('drops to 1/3 when each institution holds both ends of just one edge', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'C', dst: 'D' },
      { src: 'E', dst: 'F' },
    ]
    const assignment = { A: 0, B: 0, C: 1, D: 1, E: 2, F: 2 }
    expect(visibilityOf(edges, assignment, 3)).toBeCloseTo(1 / 3)
  })

  it('is 1.0 when everything sits in one institution', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'B', dst: 'C' },
      { src: 'C', dst: 'A' },
    ]
    expect(visibilityOf(edges, { A: 0, B: 0, C: 0 }, 4)).toBe(1)
  })

  it('cannot get a four-account cycle below 0.75 across only two banks', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'B', dst: 'C' },
      { src: 'C', dst: 'D' },
      { src: 'D', dst: 'A' },
    ]
    // Splitting the ring contiguously still leaves the two boundary
    // transactions visible to BOTH banks, so the best bank sees 3 of 4.
    expect(visibilityOf(edges, { A: 0, B: 0, C: 1, D: 1 }, 2)).toBe(0.75)
    // Alternating is worse still: every transaction touches institution 0.
    expect(visibilityOf(edges, { A: 0, B: 1, C: 0, D: 1 }, 2)).toBe(1)
  })

  it('reaches 0.5 on that cycle once there are four banks to spread it over', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'B', dst: 'C' },
      { src: 'C', dst: 'D' },
      { src: 'D', dst: 'A' },
    ]
    // One account per bank: each bank holds an endpoint of exactly 2 of the 4
    // transactions. This matches what the backend pipeline produces for the
    // four-account cycle in the test fixture (achieved 0.5 at target 0.25).
    expect(visibilityOf(edges, { A: 0, B: 1, C: 2, D: 3 }, 4)).toBe(0.5)
  })

  it('is 0 for an empty campaign', () => {
    expect(visibilityOf([], {}, 8)).toBe(0)
  })

  it('ignores accounts that were never assigned', () => {
    const edges: Edge[] = [{ src: 'A', dst: 'B' }]
    expect(visibilityOf(edges, {}, 3)).toBe(0)
  })
})

describe('binOf', () => {
  it('uses the frozen bin thresholds', () => {
    expect(binOf(1)).toBe('100')
    expect(binOf(0.9)).toBe('100')
    expect(binOf(0.89)).toBe('75')
    expect(binOf(0.65)).toBe('75')
    expect(binOf(0.64)).toBe('50')
    expect(binOf(0.45)).toBe('50')
    expect(binOf(0.44)).toBe('low')
    expect(binOf(0)).toBe('low')
  })
})

describe('visibleEdgeIndices', () => {
  it('lists the transactions one institution can see', () => {
    const edges: Edge[] = [
      { src: 'A', dst: 'B' },
      { src: 'C', dst: 'D' },
      { src: 'D', dst: 'A' },
    ]
    const assignment = { A: 0, B: 1, C: 1, D: 2 }
    expect(visibleEdgeIndices(edges, assignment, 0)).toEqual([0, 2])
    expect(visibleEdgeIndices(edges, assignment, 1)).toEqual([0, 1])
    expect(visibleEdgeIndices(edges, assignment, 2)).toEqual([1, 2])
  })
})
