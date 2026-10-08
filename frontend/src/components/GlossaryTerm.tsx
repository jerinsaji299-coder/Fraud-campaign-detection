import { useId, useState, type ReactNode } from 'react'

import { glossaryLookup } from '../lib/glossary'

interface Props {
  term: string
  children?: ReactNode
}

/** A technical term with a plain-language tooltip on hover, focus or tap. */
export function GlossaryTerm({ term, children }: Props) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const definition = glossaryLookup(term)

  if (!definition) return <>{children ?? term}</>

  return (
    <span className="relative inline-block">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        className="cursor-help border-b border-dotted border-muted text-left font-medium"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
      >
        {children ?? term}
      </button>
      {open && (
        <span
          role="tooltip"
          id={id}
          className="absolute bottom-full left-0 z-50 mb-2 block w-72 rounded-lg border border-grid bg-surface p-3 text-sm font-normal leading-snug text-ink2 shadow-lg"
        >
          {definition}
        </span>
      )}
    </span>
  )
}
