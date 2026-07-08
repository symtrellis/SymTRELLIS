import type { ThemeMode } from '../types';

export type ViewerColors = {
  background: string;
  box: string;
  mesh: string;
  symmetryAccent: string;
  symmetryCylinder: string;
  symmetryCylinderOpacity: number;
  x: string;
  y: string;
  z: string;
};

export function viewerColors(theme: ThemeMode): ViewerColors {
  if (theme === 'dark') {
    return {
      background: '#141414',
      box: '#b8b8b4',
      mesh: '#a49f99',
      symmetryAccent: '#f2f2ee',
      symmetryCylinder: '#7aa2ff',
      symmetryCylinderOpacity: 0.16,
      x: '#ff453a',
      y: '#30d158',
      z: '#0a84ff',
    };
  }

  return {
    background: '#f5f5f3',
    box: '#202020',
    mesh: '#c7beb3',
    symmetryAccent: '#1f1f1f',
    symmetryCylinder: '#8fb7ff',
    symmetryCylinderOpacity: 0.2,
    x: '#ff3b30',
    y: '#34c759',
    z: '#007aff',
  };
}
