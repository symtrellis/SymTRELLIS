import type { ThemeMode } from '../types';

const themeStorageKey = 'symtrellis.theme';

export function readStoredTheme(): ThemeMode {
  const storedTheme = window.localStorage.getItem(themeStorageKey);

  if (storedTheme === 'dark' || storedTheme === 'light') {
    return storedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function writeStoredTheme(theme: ThemeMode) {
  window.localStorage.setItem(themeStorageKey, theme);
}
