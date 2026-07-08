import * as THREE from 'three';
import type { SymmetryTuple } from '../types';
import type { ViewerColors } from './viewerTheme';

type AxialSymmetryFamily = 'Cn' | 'S2n' | 'Cnh' | 'Cnv' | 'Dn' | 'Dnd' | 'Dnh';
type PolyhedralSymmetryLabel = 'T' | 'Td' | 'Th' | 'O' | 'Oh' | 'I' | 'Ih';
type LocalPoint = [number, number, number];
type ParsedAxialSymmetry = {
  family: AxialSymmetryFamily;
  fold: number;
};
type SurfacePoint = {
  theta: number;
  y: number;
};

const symmetryCylinderHeight = 0.28;
const symmetryCylinderRadius = 0.5;
const polyhedralRadius = 0.5;
const polyhedralEdgeRadius = 0.0042;
const sqrt2 = Math.sqrt(2);
const sqrt5 = Math.sqrt(5);

export function createSymmetryPreviewGroup(symmetry: SymmetryTuple | null, colors: ViewerColors) {
  const group = new THREE.Group();

  if (!symmetry) {
    return group;
  }

  const axial = parseAxialSymmetryLabel(symmetry.label);
  if (axial) {
    group.add(createAxialSymmetryCylinder(symmetry, axial, colors));
    return group;
  }

  const polyhedral = parsePolyhedralSymmetryLabel(symmetry.label);
  if (polyhedral) {
    group.add(createPolyhedralSymmetryFrame(symmetry, polyhedral, colors));
  }

  return group;
}

function parseAxialSymmetryLabel(label: string): ParsedAxialSymmetry | null {
  const match = /^(C|D|S)(\d+)([dhv]?)$/.exec(label);

  if (!match) {
    return null;
  }

  const prefix = match[1];
  const fold = Number(match[2]);
  const suffix = match[3];

  if (prefix === 'C') {
    return { family: suffix === 'h' ? 'Cnh' : suffix === 'v' ? 'Cnv' : 'Cn', fold };
  }

  if (prefix === 'D') {
    return { family: suffix === 'd' ? 'Dnd' : suffix === 'h' ? 'Dnh' : 'Dn', fold };
  }

  if (fold % 2 === 0 && suffix === '') {
    return { family: 'S2n', fold: fold / 2 };
  }

  return null;
}

function parsePolyhedralSymmetryLabel(label: string): PolyhedralSymmetryLabel | null {
  if (
    label === 'T' ||
    label === 'Td' ||
    label === 'Th' ||
    label === 'O' ||
    label === 'Oh' ||
    label === 'I' ||
    label === 'Ih'
  ) {
    return label;
  }

  return null;
}

function createPolyhedralSymmetryFrame(
  symmetry: SymmetryTuple,
  label: PolyhedralSymmetryLabel,
  colors: ViewerColors,
) {
  const group = new THREE.Group();
  const { center, ex, ey, ez } = createSymmetryFrame(symmetry);
  const transform = new THREE.Matrix4().makeBasis(ex, ey, ez);
  const vertices = polyhedralVertices(label);
  const edges = polyhedralEdges(label, vertices);
  const material = new THREE.MeshBasicMaterial({
    color: colors.symmetryAccent,
    depthTest: true,
    depthWrite: true,
  });

  transform.setPosition(center);
  group.matrix.copy(transform);
  group.matrixAutoUpdate = false;

  edges.forEach(([start, end]) => {
    group.add(createPolyhedralEdge(vertices[start], vertices[end], material));
  });

  return group;
}

function polyhedralVertices(label: PolyhedralSymmetryLabel) {
  if (label === 'T' || label === 'Td' || label === 'Th') {
    const tetrahedron = scaleAxes([
      [0, 0, 1],
      ...ring(3, (2 * sqrt2) / 3, -1 / 3, 0),
    ]);

    if (label === 'Th') {
      return [...tetrahedron, ...tetrahedron.map((vertex) => vertex.clone().multiplyScalar(-1))];
    }

    return tetrahedron;
  }

  if (label === 'O' || label === 'Oh') {
    return scaleAxes([
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ]);
  }

  return scaleAxes([
    ...ring(5, Math.sqrt((10 + 2 * sqrt5) / 15), Math.sqrt((5 - 2 * sqrt5) / 15), Math.PI / 5),
    ...ring(5, Math.sqrt((10 - 2 * sqrt5) / 15), Math.sqrt((5 + 2 * sqrt5) / 15), Math.PI / 5),
  ]).flatMap((vertex) => [vertex, vertex.clone().multiplyScalar(-1)]);
}

function polyhedralEdges(label: PolyhedralSymmetryLabel, vertices: THREE.Vector3[]) {
  if (label === 'T' || label === 'Td') {
    return completeGraphEdges(0, 4);
  }

  if (label === 'Th') {
    return [...completeGraphEdges(0, 4), ...completeGraphEdges(4, 4)];
  }

  return nearestNeighborEdges(vertices);
}

function completeGraphEdges(offset: number, count: number) {
  const edges: Array<[number, number]> = [];

  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) {
      edges.push([offset + i, offset + j]);
    }
  }

  return edges;
}

function ring(count: number, radius: number, z: number, phase: number) {
  const points: LocalPoint[] = [];

  for (let i = 0; i < count; i += 1) {
    const theta = phase + (2 * Math.PI * i) / count;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta), z]);
  }

  return points;
}

function scaleAxes(axes: LocalPoint[]) {
  return axes.map((axis) => new THREE.Vector3(...axis).multiplyScalar(polyhedralRadius));
}

function nearestNeighborEdges(vertices: THREE.Vector3[]) {
  const edges: Array<[number, number]> = [];
  let shortest = Infinity;

  for (let i = 0; i < vertices.length; i += 1) {
    for (let j = i + 1; j < vertices.length; j += 1) {
      const distance = vertices[i].distanceTo(vertices[j]);
      if (distance > 1e-8 && distance < shortest) {
        shortest = distance;
      }
    }
  }

  for (let i = 0; i < vertices.length; i += 1) {
    for (let j = i + 1; j < vertices.length; j += 1) {
      if (Math.abs(vertices[i].distanceTo(vertices[j]) - shortest) < 1e-6) {
        edges.push([i, j]);
      }
    }
  }

  return edges;
}

function createPolyhedralEdge(
  start: THREE.Vector3,
  end: THREE.Vector3,
  material: THREE.MeshBasicMaterial,
) {
  const delta = end.clone().sub(start);
  const edge = new THREE.Mesh(
    new THREE.CylinderGeometry(polyhedralEdgeRadius, polyhedralEdgeRadius, delta.length(), 12),
    material,
  );

  edge.position.copy(start).add(end).multiplyScalar(0.5);
  edge.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return edge;
}

function createAxialSymmetryCylinder(
  symmetry: SymmetryTuple,
  axial: ParsedAxialSymmetry,
  colors: ViewerColors,
) {
  const group = new THREE.Group();
  const { center, ex, ey, ez } = createSymmetryFrame(symmetry);
  const transform = new THREE.Matrix4().makeBasis(ex, ez, ey);
  const markMaterial = new THREE.MeshBasicMaterial({
    color: colors.symmetryAccent,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  });
  const bandMaterial = new THREE.MeshBasicMaterial({
    color: colors.symmetryAccent,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  });
  const surface = new THREE.Mesh(
    new THREE.CylinderGeometry(
      symmetryCylinderRadius,
      symmetryCylinderRadius,
      symmetryCylinderHeight,
      96,
      1,
      true,
    ),
    new THREE.MeshBasicMaterial({
      color: colors.symmetryCylinder,
      depthTest: true,
      depthWrite: false,
      opacity: colors.symmetryCylinderOpacity,
      side: THREE.DoubleSide,
      transparent: true,
    }),
  );
  const ringMaterial = new THREE.LineBasicMaterial({
    color: colors.symmetryAccent,
    depthTest: true,
    depthWrite: true,
  });
  const axisMaterial = new THREE.LineBasicMaterial({
    color: colors.symmetryAccent,
    depthTest: true,
    depthWrite: false,
    opacity: 0.55,
    transparent: true,
  });

  transform.setPosition(center);
  group.matrix.copy(transform);
  group.matrixAutoUpdate = false;

  group.add(surface);
  group.add(createCylinderRing(-symmetryCylinderHeight / 2, symmetryCylinderRadius, ringMaterial));
  group.add(createCylinderRing(symmetryCylinderHeight / 2, symmetryCylinderRadius, ringMaterial));
  group.add(createCylinderAxisLine(symmetryCylinderHeight * 1.45, axisMaterial));
  group.add(createCylinderBand(symmetryCylinderRadius * 1.018, symmetryCylinderHeight * 0.06, bandMaterial));
  createWikiFamilyPatches(axial, markMaterial).forEach((patch) => group.add(patch));

  return group;
}

function createSymmetryFrame(symmetry: SymmetryTuple) {
  const center = new THREE.Vector3(...symmetry.center);
  const ez = new THREE.Vector3(...symmetry.majorAxis).normalize();
  const minor = new THREE.Vector3(...symmetry.minorAxis);
  const ex = minor.clone().sub(ez.clone().multiplyScalar(minor.dot(ez)));

  if (ex.lengthSq() < 1e-8) {
    const fallback = Math.abs(ez.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    ex.copy(fallback.sub(ez.clone().multiplyScalar(fallback.dot(ez))));
  }

  ex.normalize();
  const ey = new THREE.Vector3().crossVectors(ez, ex).normalize();

  return { center, ex, ey, ez };
}

function createWikiFamilyPatches(axial: ParsedAxialSymmetry, material: THREE.MeshBasicMaterial) {
  const patches: THREE.Mesh[] = [];
  const thetaWidth = Math.PI / 45;
  const yHigh = symmetryCylinderHeight * 0.19;

  if (axial.family === 'S2n') {
    const count = 2 * axial.fold;

    for (let i = 0; i < count; i += 1) {
      const shape = patchPoints((2 * Math.PI * i) / count, thetaWidth, yHigh);
      patches.push(
        createSurfacePatch(
          i % 2 === 0 ? [shape.c, shape.w, shape.se, shape.e] : [shape.c, shape.w, shape.ne, shape.e],
          symmetryCylinderRadius * 1.018,
          material,
        ),
      );
    }

    return patches;
  }

  for (let i = 0; i < axial.fold; i += 1) {
    const theta = (2 * Math.PI * i) / axial.fold;
    const shape = patchPoints(theta, thetaWidth, yHigh);

    if (axial.family === 'Cn') {
      patches.push(
        createSurfacePatch([shape.c, shape.w, shape.ne, shape.e], symmetryCylinderRadius * 1.018, material),
      );
    } else if (axial.family === 'Cnh') {
      patches.push(
        createSurfacePatch([shape.w, shape.ne, shape.se], symmetryCylinderRadius * 1.018, material),
      );
    } else if (axial.family === 'Cnv') {
      patches.push(
        createSurfacePatch([shape.c, shape.e, shape.n, shape.w], symmetryCylinderRadius * 1.018, material),
      );
    } else if (axial.family === 'Dn') {
      patches.push(
        createSurfacePatch([shape.ne, shape.e, shape.sw, shape.w], symmetryCylinderRadius * 1.018, material),
      );
    } else if (axial.family === 'Dnh') {
      patches.push(
        createSurfacePatch([shape.n, shape.e, shape.s, shape.w], symmetryCylinderRadius * 1.018, material),
      );
    } else {
      const lowerShape = patchPoints(theta + Math.PI / axial.fold, thetaWidth, yHigh);
      patches.push(
        createSurfacePatch([shape.c, shape.e, shape.n, shape.w], symmetryCylinderRadius * 1.018, material),
      );
      patches.push(
        createSurfacePatch(
          [lowerShape.c, lowerShape.e, lowerShape.s, lowerShape.w],
          symmetryCylinderRadius * 1.018,
          material,
        ),
      );
    }
  }

  return patches;
}

function patchPoints(theta: number, thetaWidth: number, yHigh: number) {
  return {
    c: { theta, y: 0 },
    e: { theta: theta + thetaWidth, y: 0 },
    n: { theta, y: yHigh },
    ne: { theta: theta + thetaWidth, y: yHigh },
    s: { theta, y: -yHigh },
    se: { theta: theta + thetaWidth, y: -yHigh },
    sw: { theta: theta - thetaWidth, y: -yHigh },
    w: { theta: theta - thetaWidth, y: 0 },
  };
}

function createCylinderRing(y: number, radius: number, material: THREE.LineBasicMaterial) {
  const points: THREE.Vector3[] = [];

  for (let i = 0; i < 96; i += 1) {
    const theta = (2 * Math.PI * i) / 96;
    points.push(new THREE.Vector3(Math.cos(theta) * radius, y, Math.sin(theta) * radius));
  }

  return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material);
}

function createCylinderAxisLine(height: number, material: THREE.LineBasicMaterial) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -height / 2, 0),
      new THREE.Vector3(0, height / 2, 0),
    ]),
    material,
  );
}

function createCylinderBand(radius: number, height: number, material: THREE.MeshBasicMaterial) {
  return new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 96, 1, true), material);
}

function createSurfacePatch(
  points: SurfacePoint[],
  radius: number,
  material: THREE.MeshBasicMaterial,
) {
  const geometry = new THREE.BufferGeometry();
  const indices: number[] = [];
  const vertices = points.flatMap((point) => cylinderPoint(point.theta, point.y, radius));

  for (let i = 1; i < points.length - 1; i += 1) {
    indices.push(0, i, i + 1);
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  return new THREE.Mesh(geometry, material);
}

function cylinderPoint(theta: number, y: number, radius: number) {
  return [Math.cos(theta) * radius, y, Math.sin(theta) * radius];
}
