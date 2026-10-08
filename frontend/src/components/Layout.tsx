import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  BookOpen,
  Database,
  FlaskConical,
  Home,
  LineChart,
  Menu,
  Moon,
  Network,
  Sun,
} from 'lucide-react'

import { useHealth } from '../api/hooks'
import { useThemeMode } from '../lib/themeContext'

const NAV = [
  { to: '/', label: 'Start here', icon: Home, end: true },
  { to: '/dataset', label: 'Dataset', icon: Database, end: false },
  { to: '/campaigns', label: 'Campaigns', icon: Network, end: false },
  { to: '/visibility', label: 'Visibility Lab', icon: FlaskConical, end: false },
  { to: '/results', label: 'Results', icon: LineChart, end: false },
  { to: '/methodology', label: 'Methodology', icon: BookOpen, end: false },
]

function ApiStatus() {
  const { data, isLoading, isError } = useHealth()

  let tone = 'bg-muted'
  let label = 'Checking API'
  if (isError) {
    tone = 'bg-critical'
    label = 'API unreachable'
  } else if (data) {
    if (data.core_artifacts_available) {
      tone = 'bg-good'
      label = 'API ready'
    } else {
      tone = 'bg-warning'
      label = 'API up, no data exported'
    }
  }

  return (
    <span
      className="flex items-center gap-2 text-xs text-ink2"
      role="status"
      aria-live="polite"
    >
      <span
        className={`h-2 w-2 rounded-full ${isLoading ? 'bg-muted' : tone}`}
        aria-hidden="true"
      />
      {label}
    </span>
  )
}

function ThemeToggle() {
  const { mode, toggle } = useThemeMode()
  const Icon = mode === 'dark' ? Sun : Moon
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`}
      className="rounded-md border border-grid p-2 text-ink2 hover:text-ink"
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
    </button>
  )
}

export function Layout() {
  const [open, setOpen] = useState(false)

  const navLinks = (
    <ul className="space-y-1">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <li key={to}>
          <NavLink
            to={to}
            end={end}
            onClick={() => setOpen(false)}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-grid text-ink'
                  : 'text-ink2 hover:bg-grid/60 hover:text-ink'
              }`
            }
          >
            <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
            {label}
          </NavLink>
        </li>
      ))}
    </ul>
  )

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[16rem_1fr]">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2"
      >
        Skip to content
      </a>

      <aside className="hidden border-r border-grid bg-surface p-4 lg:block">
        <div className="mb-6 px-3">
          <div className="text-sm font-semibold leading-tight text-ink">
            Fraud Campaign Discovery
          </div>
          <div className="mt-0.5 text-xs text-muted">
            Cross-institution evidence
          </div>
        </div>
        <nav aria-label="Main">{navLinks}</nav>
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="flex items-center justify-between gap-3 border-b border-grid bg-surface px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-md border border-grid p-2 text-ink2 lg:hidden"
              aria-label="Toggle navigation"
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
            >
              <Menu className="h-4 w-4" aria-hidden="true" />
            </button>
            <span className="text-sm font-semibold text-ink lg:hidden">
              Fraud Campaign Discovery
            </span>
          </div>
          <div className="flex items-center gap-3">
            <ApiStatus />
            <ThemeToggle />
          </div>
        </header>

        {open && (
          <nav aria-label="Main" className="border-b border-grid bg-surface p-3 lg:hidden">
            {navLinks}
          </nav>
        )}

        <main id="main" className="min-w-0 flex-1 p-4 sm:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
