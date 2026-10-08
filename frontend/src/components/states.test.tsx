import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ApiError } from '../api/client'
import { EmptyState, ErrorState } from './states'

describe('EmptyState', () => {
  it('shows a title and description instead of inventing content', () => {
    render(
      <EmptyState
        title="Experiment results are not available yet"
        description="They will appear here after Phase 5."
      />,
    )

    expect(
      screen.getByText('Experiment results are not available yet'),
    ).toBeInTheDocument()
    expect(
      screen.getByText('They will appear here after Phase 5.'),
    ).toBeInTheDocument()
  })
})

describe('ErrorState', () => {
  it('explains a 503 as "not exported yet" rather than as a failure', () => {
    render(
      <ErrorState
        error={new ApiError(503, "Artifact 'summary.json' has not been exported yet.")}
      />,
    )

    expect(screen.getByText('The data has not been exported yet')).toBeInTheDocument()
    // the API's own detail message is passed through, so the user is told
    // exactly which artifact is missing
    expect(
      screen.getByText(/Artifact 'summary\.json' has not been exported yet/),
    ).toBeInTheDocument()
  })

  it('surfaces the API detail message for other failures', () => {
    render(<ErrorState error={new ApiError(404, 'Campaign 999 not found.')} />)

    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('Campaign 999 not found.')).toBeInTheDocument()
  })

  it('handles a plain network error', () => {
    render(<ErrorState error={new Error('Failed to fetch')} />)
    expect(screen.getByText('Failed to fetch')).toBeInTheDocument()
  })
})
