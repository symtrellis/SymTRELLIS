import type {
  FinerSymmetryResult,
  RotationAxisCandidate,
  Vector3,
} from './types';

type RawRotationAxisCandidate = {
  axis: Vector3;
  dbscan_label: number;
  fold_e: number;
  fold_i: number;
  q: Vector3;
  ratio: number;
  rmse: number;
};

type RawC2AxisCandidate = {
  axis: Vector3;
  axis_cor: Vector3;
  dbscan_label: number;
  fold_c2: number;
  fold_i_val: number;
  q: Vector3;
  q_cor: Vector3;
  ratio: number;
  rmse: number;
};

type RawReflectionPlaneCandidate = {
  dbscan_label: number;
  fold_i_val: number;
  fold_pred?: number;
  n: Vector3;
  n_cor?: Vector3;
  ratio: number;
  rmse: number;
};

const candidateColors = [
  '#508cff',
  '#ff783c',
  '#34c759',
  '#af52de',
  '#ffcc00',
  '#00c7be',
  '#ff2d55',
  '#5856d6',
  '#8e8e93',
];

export async function detectRotationAxes(): Promise<RotationAxisCandidate[]> {
  // MOCK_DETECTION_API_START
  // Reads fixture JSON from public/mock until FastAPI detection endpoints exist.
  // Replace this block with backend detection API calls.
  const response = await fetch('/mock/detect_rotation_axes.json');
  const candidates = (await response.json()) as RawRotationAxisCandidate[];
  const result = candidates.map((candidate, index) => ({
    axis: candidate.axis,
    center: candidate.q,
    color: candidateColors[index % candidateColors.length],
    dbscanLabel: candidate.dbscan_label,
    foldE: candidate.fold_e,
    foldI: candidate.fold_i,
    id: `rotation-axis-${index}`,
    ratio: candidate.ratio,
    rmse: candidate.rmse,
  }));
  // MOCK_DETECTION_API_END

  return result;
}

export async function detectFinerSymmetry(): Promise<FinerSymmetryResult> {
  // MOCK_DETECTION_API_START
  // Reads fixture JSON from public/mock until FastAPI detection endpoints exist.
  // Replace this block with backend detection API calls.
  const [c2Response, containingResponse, perpendicularResponse] = await Promise.all([
    fetch('/mock/detect_c2_axes_perpendicular_to_axis.json'),
    fetch('/mock/detect_reflection_planes_containing_axis.json'),
    fetch('/mock/detect_reflection_planes_perpendicular_to_axis.json'),
  ]);
  const c2Axes = (await c2Response.json()) as RawC2AxisCandidate[];
  const containingPlanes = (await containingResponse.json()) as RawReflectionPlaneCandidate[];
  const perpendicularPlanes = (await perpendicularResponse.json()) as RawReflectionPlaneCandidate[];
  const result: FinerSymmetryResult = {
    c2Axes: c2Axes.map((candidate, index) => ({
      axis: candidate.axis,
      axisCorrected: candidate.axis_cor,
      center: candidate.q,
      centerCorrected: candidate.q_cor,
      color: candidateColors[(index + 1) % candidateColors.length],
      dbscanLabel: candidate.dbscan_label,
      foldC2: candidate.fold_c2,
      foldIValidation: candidate.fold_i_val,
      id: `c2-axis-${index}`,
      ratio: candidate.ratio,
      rmse: candidate.rmse,
    })),
    reflectionPlanesContainingAxis: containingPlanes.map((candidate, index) => ({
      color: candidateColors[(index + 3) % candidateColors.length],
      dbscanLabel: candidate.dbscan_label,
      foldIValidation: candidate.fold_i_val,
      foldPred: candidate.fold_pred,
      id: `vertical-plane-${index}`,
      normal: candidate.n,
      normalCorrected: candidate.n_cor ?? candidate.n,
      ratio: candidate.ratio,
      rmse: candidate.rmse,
      role: 'contains_major_axis',
    })),
    reflectionPlanesPerpendicularToAxis: perpendicularPlanes.map((candidate, index) => ({
      color: candidateColors[(index + 6) % candidateColors.length],
      dbscanLabel: candidate.dbscan_label,
      foldIValidation: candidate.fold_i_val,
      foldPred: candidate.fold_pred,
      id: `horizontal-plane-${index}`,
      normal: candidate.n,
      normalCorrected: candidate.n_cor ?? candidate.n,
      ratio: candidate.ratio,
      rmse: candidate.rmse,
      role: 'perpendicular_to_major_axis',
    })),
  };
  // MOCK_DETECTION_API_END

  return result;
}
