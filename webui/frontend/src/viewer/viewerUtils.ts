import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { SymmetryOverlay, SymmetryTuple, ThemeMode } from '../types';

export type ViewGizmoTarget = 'default' | 'x+' | 'x-' | 'y+' | 'y-' | 'z+' | 'z-';

export type ViewGizmoPickable = {
  object: THREE.Mesh;
  target: ViewGizmoTarget;
};

export type ViewGizmo = {
  camera: THREE.OrthographicCamera;
  pickables: ViewGizmoPickable[];
  scene: THREE.Scene;
};

export type SymmetryOverlayRender = {
  group: THREE.Group;
  pickables: THREE.Object3D[];
};

type AxisKey = 'x' | 'y' | 'z';
type AxialSymmetryFamily = 'Cn' | 'S2n' | 'Cnh' | 'Cnv' | 'Dn' | 'Dnd' | 'Dnh';
type PolyhedralSymmetryLabel = 'T' | 'Td' | 'Th' | 'O' | 'Oh' | 'I' | 'Ih';
type LocalPoint = [number, number, number];
type ParsedAxialSymmetry = {
  family: AxialSymmetryFamily;
  fold: number;
};

export const DEFAULT_CAMERA_DIRECTION = new THREE.Vector3(
  Math.cos(Math.PI / 6),
  0,
  Math.sin(Math.PI / 6),
).multiplyScalar(2.4);
export const VIEW_GIZMO_RIGHT = 18;
export const VIEW_GIZMO_SIZE = 108;
export const VIEW_GIZMO_TOP = 58;

const WORLD_ORIGIN = new THREE.Vector3(0, 0, 0);
const WORLD_UP = new THREE.Vector3(0, 0, 1);
const SYMMETRY_CYLINDER_HEIGHT = 0.28;
const SYMMETRY_CYLINDER_RADIUS = 0.5;
const POLYHEDRAL_RADIUS = 0.5;
const POLYHEDRAL_EDGE_RADIUS = 0.0042;
const TOI_SQRT2 = Math.sqrt(2);
const TOI_SQRT5 = Math.sqrt(5);
const UP_CANDIDATES = [
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(-1, 0, 0),
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, -1, 0),
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(0, 0, -1),
];

export function viewerColors(theme: ThemeMode) {
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

export function createViewGizmo(colors: ReturnType<typeof viewerColors>): ViewGizmo {
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-0.78, 0.78, 0.78, -0.78, 0.01, 8);
  const pickables: ViewGizmoPickable[] = [];
  const lineLength = 0.56;
  const labelLength = 0.72;
  const dotRadius = 0.048;
  const centerRadius = 0.056;
  const whiteMaterial = new THREE.MeshBasicMaterial({ color: '#ffffff' });
  const dotGeometry = new THREE.SphereGeometry(dotRadius, 24, 16);
  const centerGeometry = new THREE.SphereGeometry(centerRadius, 24, 16);

  camera.position.set(0, -2.5, 1.8);
  camera.up.copy(WORLD_UP);
  camera.lookAt(WORLD_ORIGIN);

  const axes: Array<{ color: string; target: ViewGizmoTarget }> = [
    { color: colors.x, target: 'x+' },
    { color: colors.x, target: 'x-' },
    { color: colors.y, target: 'y+' },
    { color: colors.y, target: 'y-' },
    { color: colors.z, target: 'z+' },
    { color: colors.z, target: 'z-' },
  ];

  axes.forEach(({ color, target }) => {
    const direction = viewDirectionForTarget(target);
    const position = direction.clone().multiplyScalar(lineLength);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([WORLD_ORIGIN, position]);
    const line = new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color }));
    const rim = new THREE.Mesh(
      new THREE.SphereGeometry(dotRadius * 1.35, 24, 16),
      new THREE.MeshBasicMaterial({ color }),
    );
    const dot = new THREE.Mesh(dotGeometry, whiteMaterial);

    rim.position.copy(position);
    dot.position.copy(position);
    scene.add(line, rim, dot, createGizmoLabel(target, color, direction.multiplyScalar(labelLength)));
    pickables.push({ object: rim, target }, { object: dot, target });
  });

  const center = new THREE.Mesh(centerGeometry, whiteMaterial);
  scene.add(center);
  pickables.push({ object: center, target: 'default' });

  return { camera, pickables, scene };
}

function createGizmoLabel(text: ViewGizmoTarget, color: string, position: THREE.Vector3) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;

  const context = canvas.getContext('2d')!;
  context.font = '400 44px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(viewGizmoLabelText(text), canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const label = new THREE.Sprite(
    new THREE.SpriteMaterial({
      depthTest: false,
      depthWrite: false,
      map: texture,
      transparent: true,
    }),
  );
  label.position.copy(position);
  label.scale.set(0.43, 0.23, 1);
  return label;
}

function viewGizmoLabelText(target: ViewGizmoTarget) {
  if (target === 'default') {
    return '';
  }

  return `${target[1]}${target[0].toUpperCase()}`;
}

export function viewDirectionForTarget(target: ViewGizmoTarget) {
  if (target === 'x+') {
    return new THREE.Vector3(1, 0, 0);
  }

  if (target === 'x-') {
    return new THREE.Vector3(-1, 0, 0);
  }

  if (target === 'y+') {
    return new THREE.Vector3(0, 1, 0);
  }

  if (target === 'y-') {
    return new THREE.Vector3(0, -1, 0);
  }

  if (target === 'z+') {
    return new THREE.Vector3(0, 0, 1);
  }

  if (target === 'z-') {
    return new THREE.Vector3(0, 0, -1);
  }

  return DEFAULT_CAMERA_DIRECTION.clone().normalize();
}

export function viewUpForTarget(
  target: ViewGizmoTarget,
  direction: THREE.Vector3,
  currentQuaternion: THREE.Quaternion,
) {
  if (target === 'default') {
    return WORLD_UP.clone();
  }

  return nearestViewUp(direction, currentQuaternion);
}

function nearestViewUp(direction: THREE.Vector3, currentQuaternion: THREE.Quaternion) {
  const probe = new THREE.PerspectiveCamera();
  let bestUp = WORLD_UP;
  let bestScore = -1;

  UP_CANDIDATES.forEach((candidate) => {
    if (Math.abs(candidate.dot(direction)) > 0.985) {
      return;
    }

    probe.position.copy(direction);
    probe.up.copy(candidate);
    probe.lookAt(WORLD_ORIGIN);

    const score = Math.abs(probe.quaternion.dot(currentQuaternion));
    if (score > bestScore) {
      bestScore = score;
      bestUp = candidate;
    }
  });

  return bestUp.clone();
}

export function createCanonicalBox(color: string) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.EdgesGeometry(geometry);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 });
  return new THREE.LineSegments(edges, material);
}

export function createWorldAxes(colors: ReturnType<typeof viewerColors>) {
  const axes = new THREE.Group();
  const length = 0.72;
  const headLength = 0.035;
  const headWidth = 0.025;

  axes.add(
    createAxisArrow(
      'x',
      new THREE.Vector3(1, 0, 0),
      colors.x,
      length,
      headLength,
      headWidth,
    ),
  );
  axes.add(
    createAxisArrow(
      'y',
      new THREE.Vector3(0, 1, 0),
      colors.y,
      length,
      headLength,
      headWidth,
    ),
  );
  axes.add(
    createAxisArrow(
      'z',
      new THREE.Vector3(0, 0, 1),
      colors.z,
      length,
      headLength,
      headWidth,
    ),
  );
  axes.add(createAxisLabel('x', colors.x, new THREE.Vector3(length + 0.055, 0, 0)));
  axes.add(createAxisLabel('y', colors.y, new THREE.Vector3(0, length + 0.055, 0)));
  axes.add(createAxisLabel('z', colors.z, new THREE.Vector3(0, 0, length + 0.055)));

  return axes;
}

export function updateWorldAxesColors(axes: THREE.Object3D, colors: ReturnType<typeof viewerColors>) {
  axes.traverse((object) => {
    const axis = object.userData.axis as AxisKey | undefined;

    if (!axis) {
      return;
    }

    if (object instanceof THREE.ArrowHelper) {
      object.setColor(colors[axis]);
    }

    if (object instanceof CSS2DObject) {
      object.element.style.color = colors[axis];
    }
  });
}

function createAxisArrow(
  axis: AxisKey,
  direction: THREE.Vector3,
  color: string,
  length: number,
  headLength: number,
  headWidth: number,
) {
  const arrow = new THREE.ArrowHelper(
    direction,
    new THREE.Vector3(),
    length,
    color,
    headLength,
    headWidth,
  );
  arrow.userData.axis = axis;
  return arrow;
}

function createAxisLabel(axis: AxisKey, color: string, position: THREE.Vector3) {
  const element = document.createElement('div');
  element.className = 'axis-label';
  element.style.color = color;
  element.textContent = axis.toUpperCase();

  const label = new CSS2DObject(element);
  label.userData.axis = axis;
  label.position.copy(position);
  return label;
}

export function createSymmetryOverlayGroup(
  overlays: SymmetryOverlay[],
  selectedOverlayId: string | null,
  selectableOverlayIds: string[],
): SymmetryOverlayRender {
  const group = new THREE.Group();
  const pickables: THREE.Object3D[] = [];
  const selectableIds = new Set(selectableOverlayIds);

  overlays.forEach((overlay) => {
    const selected = overlay.id === selectedOverlayId;
    const render =
      overlay.kind === 'reflection_plane'
        ? createReflectionPlaneOverlay(overlay, selected)
        : createAxisOverlay(overlay, selected);

    group.add(render.group);
    if (selectableIds.has(overlay.id)) {
      pickables.push(...render.pickables);
    }
  });

  return { group, pickables };
}

export function createSymmetryPreviewGroup(
  symmetry: SymmetryTuple | null,
  colors: ReturnType<typeof viewerColors>,
) {
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
  colors: ReturnType<typeof viewerColors>,
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
  // Keep these local-frame axes aligned with symtrellis/symmetry/TOI.py.
  if (label === 'T' || label === 'Td' || label === 'Th') {
    const tetrahedron = scaleAxes([
      [0, 0, 1],
      ...ring(3, (2 * TOI_SQRT2) / 3, -1 / 3, 0),
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
    ...ring(5, Math.sqrt((10 + 2 * TOI_SQRT5) / 15), Math.sqrt((5 - 2 * TOI_SQRT5) / 15), Math.PI / 5),
    ...ring(5, Math.sqrt((10 - 2 * TOI_SQRT5) / 15), Math.sqrt((5 + 2 * TOI_SQRT5) / 15), Math.PI / 5),
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
  return axes.map((axis) => new THREE.Vector3(...axis).multiplyScalar(POLYHEDRAL_RADIUS));
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
    new THREE.CylinderGeometry(POLYHEDRAL_EDGE_RADIUS, POLYHEDRAL_EDGE_RADIUS, delta.length(), 12),
    material,
  );

  edge.position.copy(start).add(end).multiplyScalar(0.5);
  edge.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return edge;
}

function createAxialSymmetryCylinder(
  symmetry: SymmetryTuple,
  axial: ParsedAxialSymmetry,
  colors: ReturnType<typeof viewerColors>,
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
      SYMMETRY_CYLINDER_RADIUS,
      SYMMETRY_CYLINDER_RADIUS,
      SYMMETRY_CYLINDER_HEIGHT,
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
  group.add(createCylinderRing(-SYMMETRY_CYLINDER_HEIGHT / 2, SYMMETRY_CYLINDER_RADIUS, ringMaterial));
  group.add(createCylinderRing(SYMMETRY_CYLINDER_HEIGHT / 2, SYMMETRY_CYLINDER_RADIUS, ringMaterial));
  group.add(createCylinderAxisLine(SYMMETRY_CYLINDER_HEIGHT * 1.45, axisMaterial));
  group.add(createCylinderBand(SYMMETRY_CYLINDER_RADIUS * 1.006, SYMMETRY_CYLINDER_HEIGHT * 0.06, bandMaterial));
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

type SurfacePoint = {
  theta: number;
  y: number;
};

function createWikiFamilyPatches(axial: ParsedAxialSymmetry, material: THREE.MeshBasicMaterial) {
  const patches: THREE.Mesh[] = [];
  const thetaWidth = Math.PI / 45;
  const yHigh = SYMMETRY_CYLINDER_HEIGHT * 0.19;

  if (axial.family === 'S2n') {
    const count = 2 * axial.fold;

    for (let i = 0; i < count; i += 1) {
      const shape = patchPoints((2 * Math.PI * i) / count, thetaWidth, yHigh);
      patches.push(
        createSurfacePatch(
          i % 2 === 0 ? [shape.c, shape.w, shape.se, shape.e] : [shape.c, shape.w, shape.ne, shape.e],
          SYMMETRY_CYLINDER_RADIUS * 1.018,
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
      patches.push(createSurfacePatch([shape.c, shape.w, shape.ne, shape.e], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
    } else if (axial.family === 'Cnh') {
      patches.push(createSurfacePatch([shape.w, shape.ne, shape.se], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
    } else if (axial.family === 'Cnv') {
      patches.push(createSurfacePatch([shape.c, shape.e, shape.n, shape.w], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
    } else if (axial.family === 'Dn') {
      patches.push(createSurfacePatch([shape.ne, shape.e, shape.sw, shape.w], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
    } else if (axial.family === 'Dnh') {
      patches.push(createSurfacePatch([shape.n, shape.e, shape.s, shape.w], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
    } else {
      const lowerShape = patchPoints(theta + Math.PI / axial.fold, thetaWidth, yHigh);
      patches.push(createSurfacePatch([shape.c, shape.e, shape.n, shape.w], SYMMETRY_CYLINDER_RADIUS * 1.018, material));
      patches.push(
        createSurfacePatch(
          [lowerShape.c, lowerShape.e, lowerShape.s, lowerShape.w],
          SYMMETRY_CYLINDER_RADIUS * 1.018,
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

export function createAxisOverlay(
  overlay: Extract<SymmetryOverlay, { kind: 'rotation_axis' | 'c2_axis' }>,
  selected: boolean,
): SymmetryOverlayRender {
  const group = new THREE.Group();
  const axis = new THREE.Vector3(...overlay.axis).normalize();
  const length = overlay.kind === 'rotation_axis' ? 0.5 : 0.5;
  const shaftLength = length * 0.9;
  const headLength = length - shaftLength;
  const shaftRadius = selected ? 0.003 : 0.003;
  const headRadius = selected ? 0.015 : 0.016;
  const material = new THREE.MeshBasicMaterial({
    color: overlay.color,
    transparent: true,
    opacity: selected ? 1 : 0.82,
  });
  const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 18),
    material,
  );
  const head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLength, 24), material);

  shaft.position.copy(axis.clone().multiplyScalar(shaftLength / 2));
  shaft.quaternion.copy(orientation);
  shaft.userData.overlayId = overlay.id;
  head.position.copy(axis.clone().multiplyScalar(shaftLength + headLength / 2));
  head.quaternion.copy(orientation);
  head.userData.overlayId = overlay.id;
  group.add(shaft, head);

  if (overlay.label) {
    const labelElement = document.createElement('div');
    labelElement.className = 'symmetry-overlay-label';
    labelElement.style.color = overlay.color;
    labelElement.textContent = overlay.label;

    const label = new CSS2DObject(labelElement);
    label.position.copy(axis.clone().multiplyScalar(length + 0.08));
    group.add(label);
  }

  return { group, pickables: [shaft, head] };
}

export function createReflectionPlaneOverlay(
  overlay: Extract<SymmetryOverlay, { kind: 'reflection_plane' }>,
  selected: boolean,
): SymmetryOverlayRender {
  const group = new THREE.Group();
  const normal = new THREE.Vector3(...overlay.normal).normalize();
  const center = new THREE.Vector3(...overlay.center);
  const majorAxis = new THREE.Vector3(...overlay.majorAxis).normalize();
  const edgeY =
    overlay.role === 'contains_major_axis'
      ? majorAxis.clone().sub(normal.clone().multiplyScalar(majorAxis.dot(normal))).normalize()
      : new THREE.Vector3(1, 0, 0).sub(normal.clone().multiplyScalar(normal.x)).normalize();
  if (edgeY.lengthSq() < 1e-8) {
    edgeY.copy(new THREE.Vector3(0, 1, 0).sub(normal.clone().multiplyScalar(normal.y))).normalize();
  }
  const edgeX = edgeY.clone().cross(normal).normalize();
  const geometry = new THREE.PlaneGeometry(1, 1);
  const material = new THREE.MeshBasicMaterial({
    color: overlay.color,
    depthWrite: false,
    opacity: selected ? 0.3 : 0.18,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const plane = new THREE.Mesh(geometry, material);

  plane.position.copy(center);
  plane.setRotationFromMatrix(new THREE.Matrix4().makeBasis(edgeX, edgeY, normal));
  plane.userData.overlayId = overlay.id;

  group.add(plane);
  return { group, pickables: [plane] };
}

export function normalizeObjectToCanonicalBox(object: THREE.Object3D, targetSize = 0.88) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = targetSize / Math.max(size.x, size.y, size.z);

  object.position.sub(center);
  object.scale.multiplyScalar(scale);
}

// MOCK_TEST_GLB_START
export function orientMockGlbToZUp(object: THREE.Object3D) {
  // Fixture-only correction for public/mock/test.glb.
  // Real backend mesh previews and generated mock artifacts must already use canonical Z-up coordinates.
  object.rotation.x = Math.PI / 2;
  object.updateMatrixWorld(true);
}
// MOCK_TEST_GLB_END

export function applyViewerMaterial(object: THREE.Object3D, color: string) {
  const fallbackMaterial = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.03,
    roughness: 0.48,
  });

  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      if (!meshHasTexture(child)) {
        child.material = fallbackMaterial;
      }

      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}

type MaybeTexturedMaterial = THREE.Material & {
  map?: THREE.Texture | null;
};

function meshHasTexture(mesh: THREE.Mesh) {
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  return materials.some((material) => Boolean((material as MaybeTexturedMaterial).map));
}
