import type { DagEdge, DagNode } from './types';

export const dagNodes: DagNode[] = [
  { id: 'img_cond', label: 'IMG COND', shortLabel: 'IMG COND' },
  { id: 'nat_ss', label: 'NAT SS', shortLabel: 'NAT SS' },
  { id: 'nat_shape', label: 'NAT SHAPE', shortLabel: 'NAT SHAPE' },
  { id: 'detect_sym', label: 'DETECT SYM', shortLabel: 'DETECT SYM' },
  { id: 'manual_sym', label: 'MANUAL SYM', shortLabel: 'MANUAL SYM' },
  { id: 'sym_ss', label: 'SYM SS', shortLabel: 'SYM SS' },
  { id: 'sym_shape', label: 'SYM SHAPE', shortLabel: 'SYM SHAPE' },
  { id: 'texture', label: 'TEXTURE', shortLabel: 'TEXTURE' },
];

export const dagEdges: DagEdge[] = [
  { id: 'img_cond-manual_sym', source: 'img_cond', target: 'manual_sym' },
  { id: 'img_cond-nat_ss', source: 'img_cond', target: 'nat_ss' },
  { id: 'nat_ss-nat_shape', source: 'nat_ss', target: 'nat_shape' },
  { id: 'nat_shape-detect_sym', source: 'nat_shape', target: 'detect_sym' },
  { id: 'nat_shape-texture', source: 'nat_shape', target: 'texture' },
  { id: 'detect_sym-sym_ss', source: 'detect_sym', target: 'sym_ss' },
  { id: 'manual_sym-sym_ss', source: 'manual_sym', target: 'sym_ss' },
  { id: 'sym_ss-sym_shape', source: 'sym_ss', target: 'sym_shape' },
  { id: 'sym_shape-texture', source: 'sym_shape', target: 'texture' },
];
