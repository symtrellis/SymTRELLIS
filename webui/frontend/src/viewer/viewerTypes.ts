import type { SymmetryOverlay, SymmetryTuple } from '../types';

export type ViewerGlbContent = {
  material: 'neutral' | 'source';
  url: string;
};

export type ViewerContent = {
  glb: ViewerGlbContent | null;
  overlays: SymmetryOverlay[];
  selectedOverlayId: string | null;
  selectableOverlayIds: string[];
  symmetryPreview: SymmetryTuple | null;
};

export const emptyViewerContent: ViewerContent = {
  glb: null,
  overlays: [],
  selectedOverlayId: null,
  selectableOverlayIds: [],
  symmetryPreview: null,
};
