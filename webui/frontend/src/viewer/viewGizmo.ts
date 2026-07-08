import * as THREE from 'three';
import type { ViewerColors } from './viewerTheme';

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

export const DEFAULT_CAMERA_DIRECTION = new THREE.Vector3(
  Math.cos(Math.PI / 6),
  0,
  Math.sin(Math.PI / 6),
).multiplyScalar(2.4);
export const VIEW_GIZMO_RIGHT = 18;
export const VIEW_GIZMO_SIZE = 108;
export const VIEW_GIZMO_TOP = 58;

const worldOrigin = new THREE.Vector3(0, 0, 0);
const worldUp = new THREE.Vector3(0, 0, 1);

export function createViewGizmo(colors: ViewerColors): ViewGizmo {
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
  camera.up.copy(worldUp);
  camera.lookAt(worldOrigin);

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
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([worldOrigin, position]);
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
