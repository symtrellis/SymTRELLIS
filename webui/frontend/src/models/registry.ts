import type { EnabledModelId, ModelOption, ModelSpec } from './types';
import { trellis2ModelSpec } from './trellis2';

export const modelOptions: ModelOption[] = [
  { disabled: true, id: 'trellis', label: 'TRELLIS' },
  { disabled: false, id: 'trellis2', label: 'TRELLIS.2' },
  { disabled: true, id: 'sam3d_object', label: 'SAM-3D Object' },
];

export const modelSpecs: Record<EnabledModelId, ModelSpec> = {
  trellis2: trellis2ModelSpec,
};
