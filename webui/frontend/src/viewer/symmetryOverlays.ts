import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { SymmetryOverlay } from '../types';

export type SymmetryOverlayRender = {
  group: THREE.Group;
  pickables: THREE.Object3D[];
};

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

function createAxisOverlay(
  overlay: Extract<SymmetryOverlay, { kind: 'rotation_axis' | 'c2_axis' }>,
  selected: boolean,
): SymmetryOverlayRender {
  const group = new THREE.Group();
  const axis = new THREE.Vector3(...overlay.axis).normalize();
  const length = 0.5;
  const shaftLength = length * 0.9;
  const headLength = length - shaftLength;
  const material = new THREE.MeshBasicMaterial({
    color: overlay.color,
    opacity: selected ? 1 : 0.82,
    transparent: true,
  });
  const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, shaftLength, 18), material);
  const head = new THREE.Mesh(new THREE.ConeGeometry(selected ? 0.015 : 0.016, headLength, 24), material);

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

function createReflectionPlaneOverlay(
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
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({
      color: overlay.color,
      depthWrite: false,
      opacity: selected ? 0.3 : 0.18,
      side: THREE.DoubleSide,
      transparent: true,
    }),
  );

  plane.position.copy(center);
  plane.setRotationFromMatrix(new THREE.Matrix4().makeBasis(edgeX, edgeY, normal));
  plane.userData.overlayId = overlay.id;
  group.add(plane);

  return { group, pickables: [plane] };
}
