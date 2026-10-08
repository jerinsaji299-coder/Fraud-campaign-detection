import { useEffect, useRef, useState } from 'react'
import cytoscape, { type Core, type ElementDefinition } from 'cytoscape'

import { CHROME, institutionColor, institutionLabel } from '../theme'
import { useThemeMode } from '../lib/themeContext'
import type { CampaignTransaction } from '../api/types'

interface Props {
  transactions: CampaignTransaction[]
  /** account id -> institution, for the currently selected view */
  assignment: Record<string, number>
  /** Only transactions up to this index are revealed (time replay). */
  revealedUpTo: number
  /** When set, everything this institution cannot see is faded. */
  focusInstitution: number | null
  baseType: string
  onSelectAccount: (account: string | null) => void
  selectedAccount: string | null
}

/**
 * The campaign as a graph: accounts are nodes, transactions are directed
 * edges. Cytoscape is used directly (no React wrapper) with a layout chosen
 * per topology, so a FAN-OUT always looks like a star and a CYCLE always
 * looks like a ring, identically every time the page is opened.
 */

function layoutFor(baseType: string) {
  if (baseType === 'CYCLE') {
    return { name: 'circle', padding: 24, animate: false } as const
  }
  if (['FAN-IN', 'FAN-OUT', 'GATHER-SCATTER', 'SCATTER-GATHER'].includes(baseType)) {
    return { name: 'concentric', padding: 24, animate: false, minNodeSpacing: 28 } as const
  }
  return {
    name: 'breadthfirst',
    padding: 24,
    animate: false,
    directed: true,
    spacingFactor: 1.1,
  } as const
}

function formatAmount(value: number, currency: string): string {
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${currency}`
}

export function CampaignGraph({
  transactions,
  assignment,
  revealedUpTo,
  focusInstitution,
  baseType,
  onSelectAccount,
  selectedAccount,
}: Props) {
  const { mode } = useThemeMode()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<Core | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(
    null,
  )

  // Build the graph once per campaign/topology; styling and visibility are
  // applied separately so replaying time does not relayout the graph.
  useEffect(() => {
    if (!containerRef.current) return

    const accounts = new Set<string>()
    for (const txn of transactions) {
      accounts.add(txn.src)
      accounts.add(txn.dst)
    }

    const elements: ElementDefinition[] = [
      ...[...accounts].sort().map((account) => ({
        data: { id: account, account },
      })),
      ...transactions.map((txn) => ({
        data: {
          id: `t${txn.txn_index}`,
          source: txn.src,
          target: txn.dst,
          index: txn.txn_index,
          tooltip: `${formatAmount(txn.amount_paid, txn.payment_currency)}\n${txn.timestamp
            .slice(0, 16)
            .replace('T', ' ')} · ${txn.payment_format}`,
        },
      })),
    ]

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: layoutFor(baseType),
      minZoom: 0.3,
      maxZoom: 3,
      style: [
        {
          selector: 'node',
          style: {
            width: 30,
            height: 30,
            label: 'data(institutionLabel)',
            'font-size': 10,
            'font-weight': 600,
            'text-valign': 'center',
            'text-halign': 'center',
            color: '#ffffff',
            'text-outline-width': 2,
            'border-width': 2,
          },
        },
        {
          selector: 'node:selected',
          style: { 'border-width': 4 },
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.9,
          },
        },
        {
          selector: '.hidden',
          style: { opacity: 0.08 },
        },
        {
          selector: '.unseen',
          style: { opacity: 0.15 },
        },
      ],
    })

    cy.on('mouseover', 'edge', (event) => {
      const position = event.renderedPosition ?? event.target.renderedMidpoint()
      setTooltip({
        x: position.x,
        y: position.y,
        text: String(event.target.data('tooltip')),
      })
    })
    cy.on('mouseout', 'edge', () => setTooltip(null))
    cy.on('tap', 'node', (event) => onSelectAccount(String(event.target.data('account'))))
    cy.on('tap', (event) => {
      if (event.target === cy) onSelectAccount(null)
    })

    cyRef.current = cy
    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [transactions, baseType, onSelectAccount])

  // Colors depend on the assignment and the theme.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.batch(() => {
      cy.nodes().forEach((node) => {
        const account = String(node.data('account'))
        const institution = assignment[account]
        const color =
          institution === undefined
            ? CHROME[mode].muted
            : institutionColor(institution, mode)
        node.data('institutionLabel', institution === undefined ? '?' : institutionLabel(institution))
        node.style({
          'background-color': color,
          'border-color': CHROME[mode].surface,
          'text-outline-color': color,
        })
      })
      cy.edges().style({
        'line-color': CHROME[mode].axis,
        'target-arrow-color': CHROME[mode].axis,
      })
    })
  }, [assignment, mode])

  // Time replay + bank's-eye view are pure class toggles.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.batch(() => {
      cy.edges().forEach((edge) => {
        const index = Number(edge.data('index'))
        const revealed = index <= revealedUpTo
        edge.toggleClass('hidden', !revealed)

        const source = String(edge.source().data('account'))
        const target = String(edge.target().data('account'))
        const seen =
          focusInstitution === null ||
          assignment[source] === focusInstitution ||
          assignment[target] === focusInstitution
        edge.toggleClass('unseen', revealed && !seen)
      })
      cy.nodes().forEach((node) => {
        const account = String(node.data('account'))
        const seen = focusInstitution === null || assignment[account] === focusInstitution
        node.toggleClass('unseen', !seen)
      })
    })
  }, [revealedUpTo, focusInstitution, assignment])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().unselect()
    if (selectedAccount) cy.getElementById(selectedAccount).select()
  }, [selectedAccount])

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="h-[26rem] w-full rounded-lg border border-grid bg-surface"
        role="img"
        aria-label={`Graph of the campaign: ${
          Object.keys(assignment).length
        } accounts and ${transactions.length} transactions. Accounts are labelled by institution.`}
      />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-10 max-w-56 whitespace-pre-line rounded-lg border border-grid bg-surface px-2 py-1 text-xs text-ink shadow-lg"
          style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
        >
          {tooltip.text}
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>Drag nodes to rearrange &middot; scroll to zoom &middot; click a node for details</span>
        <button
          type="button"
          onClick={() => cyRef.current?.fit(undefined, 24)}
          className="rounded border border-grid px-2 py-0.5 text-ink2 hover:text-ink"
        >
          Reset view
        </button>
      </div>
    </div>
  )
}
