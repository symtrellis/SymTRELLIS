import type { SymmetryOverlay, SymmetryTuple } from '../types';
import type { ViewerMaterial } from '../models/types';

export type ViewerGlbContent = {
  material: ViewerMaterial;
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
