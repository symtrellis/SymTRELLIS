import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { ViewerGlbContent } from './viewerTypes';
import type { ViewerColors } from './viewerTheme';
import { disposeObject } from './scenePrimitives';

type MaterialWithMap = THREE.Material & {
  map?: THREE.Texture | null;
};

export type GlbContentManager = {
  applyTheme: (colors: ViewerColors) => void;
  dispose: () => void;
  setContent: (content: ViewerGlbContent | null, colors: ViewerColors) => void;
};

export function createGlbContentManager(scene: THREE.Scene): GlbContentManager {
  const loader = new GLTFLoader();
  const gltfYUpToViewerZUp = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, 0, -1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
  );
  let activeContent: ViewerGlbContent | null = null;
  let model: THREE.Object3D | null = null;
  let mounted = true;
  let version = 0;

  const clearModel = () => {
    if (!model) {
      return;
    }

    scene.remove(model);
    disposeObject(model);
    model = null;
  };

  return {
    applyTheme(colors) {
      if (activeContent?.material === 'neutral' && model) {
        applyViewerMaterial(model, colors.mesh);
      }
    },
    dispose() {
      mounted = false;
      version += 1;
      clearModel();
    },
    setContent(content, colors) {
      if (
        activeContent?.url === content?.url &&
        activeContent?.material === content?.material
      ) {
        return;
      }

      activeContent = content;
      version += 1;
      clearModel();

      if (!content) {
        return;
      }

      const loadVersion = version;

      loader.load(content.url, (gltf) => {
        if (!mounted || loadVersion !== version) {
          disposeObject(gltf.scene);
          return;
        }

        model = gltf.scene;
        model.applyMatrix4(gltfYUpToViewerZUp);
        normalizeObjectToCanonicalBox(model);
        if (content.material === 'neutral') {
          applyViewerMaterial(model, colors.mesh);
        }
        scene.add(model);
      });
    },
  };
}

function normalizeObjectToCanonicalBox(object: THREE.Object3D, targetSize = 0.88) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const maxAxis = Math.max(size.x, size.y, size.z);
  if (maxAxis > 0) {
    const scale = targetSize / maxAxis;
    object.scale.multiplyScalar(scale);
    object.position.copy(center).multiplyScalar(-scale);
  }
}

function applyViewerMaterial(object: THREE.Object3D, color: string) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      const oldMaterial = child.material;
      const map = Array.isArray(oldMaterial)
        ? (oldMaterial[0] as MaterialWithMap | undefined)?.map
        : (oldMaterial as MaterialWithMap).map;

      child.material = new THREE.MeshStandardMaterial({
        color,
        map: map ?? null,
        metalness: 0.04,
        roughness: 0.62,
      });
      if (Array.isArray(oldMaterial)) {
        oldMaterial.forEach((material) => material.dispose());
      } else {
        oldMaterial.dispose();
      }
    }
  });
}
