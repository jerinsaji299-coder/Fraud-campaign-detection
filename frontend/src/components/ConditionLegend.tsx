import { CONDITIONS, CONDITION_ORDER, conditionColor } from '../theme'
import { useThemeMode } from '../lib/themeContext'

/** The four experimental conditions, each with its fixed color AND its line
 * dash pattern, so the conditions are never told apart by color alone. */
export function ConditionLegend({ compact = false }: { compact?: boolean }) {
  const { mode } = useThemeMode()

  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 text-xs" aria-label="Experimental conditions">
      {CONDITION_ORDER.map((key) => {
        const condition = CONDITIONS[key]
        const color = conditionColor(key, mode)
        return (
          <li key={key} className="flex items-center gap-2">
            <svg width="28" height="10" aria-hidden="true" className="shrink-0">
              <line
                x1="0"
                y1="5"
                x2="28"
                y2="5"
                stroke={color}
                strokeWidth="2"
                strokeDasharray={condition.dash || undefined}
              />
            </svg>
            <span className="font-medium text-ink">
              {compact ? condition.short : condition.label}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
