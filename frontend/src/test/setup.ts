import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Testing Library only auto-registers cleanup when Vitest globals are on.
// Globals are off here, so unmount between tests explicitly — otherwise the
// DOM accumulates across tests in a file and queries find duplicates.
afterEach(cleanup)
