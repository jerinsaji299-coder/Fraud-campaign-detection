import { createContext, useContext } from 'react'

import type { Mode } from '../theme'

export interface ThemeValue {
  mode: Mode
  toggle: () => void
}

export const ThemeContext = createContext<ThemeValue>({
  mode: 'light',
  toggle: () => {},
})

export function useThemeMode(): ThemeValue {
  return useContext(ThemeContext)
}
