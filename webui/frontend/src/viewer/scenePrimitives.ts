import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { ViewerColors } from './viewerTheme';

type AxisKey = 'x' | 'y' | 'z';

type DisposableObject = THREE.Object3D & {
  geometry?: THREE.BufferGeometry;
  material?: THREE.Material | THREE.Material[];
};

type TexturedMaterial = THREE.Material & {
  map?: THREE.Texture | null;
};

export function createCanonicalBox(color: string) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.EdgesGeometry(geometry);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 });
  return new THREE.LineSegments(edges, material);
}

export function createWorldAxes(colors: ViewerColors) {
  const axes = new THREE.Group();
  const length = 0.72;
  const headLength = 0.035;
  const headWidth = 0.025;

  axes.add(createAxisArrow('x', new THREE.Vector3(1, 0, 0), colors.x, length, headLength, headWidth));
  axes.add(createAxisArrow('y', new THREE.Vector3(0, 1, 0), colors.y, length, headLength, headWidth));
  axes.add(createAxisArrow('z', new THREE.Vector3(0, 0, 1), colors.z, length, headLength, headWidth));
  axes.add(createAxisLabel('x', colors.x, new THREE.Vector3(length + 0.055, 0, 0)));
  axes.add(createAxisLabel('y', colors.y, new THREE.Vector3(0, length + 0.055, 0)));
  axes.add(createAxisLabel('z', colors.z, new THREE.Vector3(0, 0, length + 0.055)));

  return axes;
}

export function updateWorldAxesColors(axes: THREE.Object3D, colors: ViewerColors) {
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

export function createViewerLights(theme: 'light' | 'dark') {
  const group = new THREE.Group();
  const hemisphereLight = new THREE.HemisphereLight(
    0xffffff,
    theme === 'dark' ? 0x343438 : 0xd6d0c6,
    theme === 'dark' ? 0.72 : 0.62,
  );
  const ambientLight = new THREE.AmbientLight(0xffffff, theme === 'dark' ? 0.25 : 0.3);
  const keyLight = new THREE.DirectionalLight(0xffffff, theme === 'dark' ? 3.5 : 3);
  const fillLight = new THREE.DirectionalLight(0xdde6ff, theme === 'dark' ? 0.85 : 0.58);
  const rimLight = new THREE.DirectionalLight(0xffffff, theme === 'dark' ? 1.4 : 1.05);

  keyLight.position.set(2.5, -3.5, 4.5);
  keyLight.castShadow = true;
  fillLight.position.set(-3, 2.5, 2);
  rimLight.position.set(-2, 3.8, 3.2);
  group.add(hemisphereLight, ambientLight, keyLight, fillLight, rimLight);

  return group;
}

export function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof CSS2DObject) {
      child.element.remove();
    }

    const disposable = child as DisposableObject;
    disposable.geometry?.dispose();

    if (Array.isArray(disposable.material)) {
      disposable.material.forEach(disposeMaterial);
    } else if (disposable.material) {
      disposeMaterial(disposable.material);
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

function disposeMaterial(material: THREE.Material) {
  const texture = (material as TexturedMaterial).map;
  texture?.dispose();
  material.dispose();
}
