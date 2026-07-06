import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { SymmetryOverlay, SymmetryTuple, ThemeMode } from '../types';
import {
  DEFAULT_CAMERA_DIRECTION,
  VIEW_GIZMO_RIGHT,
  VIEW_GIZMO_SIZE,
  VIEW_GIZMO_TOP,
  applyViewerMaterial,
  createCanonicalBox,
  createSymmetryPreviewGroup,
  createSymmetryOverlayGroup,
  createViewGizmo,
  createWorldAxes,
  normalizeObjectToCanonicalBox,
  orientMockGlbToZUp,
  updateWorldAxesColors,
  viewDirectionForTarget,
  viewUpForTarget,
  viewerColors,
} from './viewerUtils';
import type { ViewGizmoTarget } from './viewerUtils';

type ThreeViewerProps = {
  onOverlayPick: (overlayId: string) => void;
  overlays: SymmetryOverlay[];
  selectableOverlayIds: string[];
  selectedOverlayId: string | null;
  symmetryPreview: SymmetryTuple | null;
  theme: ThemeMode;
};

type CameraAnimation = {
  duration: number;
  startPosition: THREE.Vector3;
  startTime: number;
  startUp: THREE.Vector3;
  targetPosition: THREE.Vector3;
  targetUp: THREE.Vector3;
};

type ViewerRuntime = {
  applyTheme: (theme: ThemeMode) => void;
  setSymmetryPreview: (symmetry: SymmetryTuple | null) => void;
  setOverlays: (
    overlays: SymmetryOverlay[],
    selectedOverlayId: string | null,
    selectableOverlayIds: string[],
  ) => void;
};

export function ThreeViewer({
  onOverlayPick,
  overlays,
  selectableOverlayIds,
  selectedOverlayId,
  symmetryPreview,
  theme,
}: ThreeViewerProps) {
  const hostRef = useRef<HTMLElement>(null);
  const onOverlayPickRef = useRef(onOverlayPick);
  const runtimeRef = useRef<ViewerRuntime | null>(null);
  onOverlayPickRef.current = onOverlayPick;

  useEffect(() => {
    const host = hostRef.current;

    if (!host) {
      return;
    }

    let activeTheme = theme;
    let colors = viewerColors(activeTheme);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(colors.background);

    const camera = new THREE.PerspectiveCamera(38, host.clientWidth / host.clientHeight, 0.01, 100);
    camera.up.set(0, 0, 1);
    const mobileCameraScale = 1.8;
    const isCompactViewport = () => window.matchMedia('(max-width: 760px)').matches;
    const defaultCameraDistance = (compact: boolean) =>
      DEFAULT_CAMERA_DIRECTION.length() * (compact ? mobileCameraScale : 1);
    const setDefaultCameraPosition = (compact: boolean) => {
      camera.position.copy(DEFAULT_CAMERA_DIRECTION).multiplyScalar(compact ? mobileCameraScale : 1);
    };
    let compactViewport = isCompactViewport();
    let previousCompact = compactViewport;
    setDefaultCameraPosition(compactViewport);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = activeTheme === 'dark' ? 1.08 : 1;
    renderer.shadowMap.enabled = true;
    renderer.domElement.className = 'three-canvas';
    host.appendChild(renderer.domElement);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(host.clientWidth, host.clientHeight);
    labelRenderer.domElement.className = 'axis-label-layer';
    host.appendChild(labelRenderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.update();
    const rescaleCameraDistance = (scale: number) => {
      const offset = camera.position.clone().sub(controls.target).multiplyScalar(scale);
      camera.position.copy(controls.target).add(offset);
    };

    const hemisphereLight = new THREE.HemisphereLight(
      0xffffff,
      activeTheme === 'dark' ? 0x343438 : 0xd6d0c6,
      activeTheme === 'dark' ? 0.72 : 0.62,
    );
    scene.add(hemisphereLight);

    const ambientLight = new THREE.AmbientLight(0xffffff, activeTheme === 'dark' ? 0.25 : 0.3);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, activeTheme === 'dark' ? 3.5 : 3);
    keyLight.position.set(2.5, -3.5, 4.5);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xdde6ff, activeTheme === 'dark' ? 0.85 : 0.58);
    fillLight.position.set(-3, 2.5, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, activeTheme === 'dark' ? 1.4 : 1.05);
    rimLight.position.set(-2, 3.8, 3.2);
    scene.add(rimLight);

    let canonicalBox = createCanonicalBox(colors.box);
    const worldAxes = createWorldAxes(colors);
    scene.add(canonicalBox, worldAxes);

    let symmetryOverlay = createSymmetryOverlayGroup([], null, []);
    let symmetryPickables = symmetryOverlay.pickables;
    scene.add(symmetryOverlay.group);

    let activeSymmetryPreview: SymmetryTuple | null = null;
    let symmetryPreviewGroup = createSymmetryPreviewGroup(null, colors);
    scene.add(symmetryPreviewGroup);

    let viewGizmo = createViewGizmo(colors);
    let gizmoMeshes = viewGizmo.pickables.map((pickable) => pickable.object);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let cameraAnimation: CameraAnimation | null = null;
    let pendingOverlayPick: { dragged: boolean; id: string; x: number; y: number } | null = null;
    let usingDefaultView = true;
    const markCustomView = () => {
      usingDefaultView = false;
    };
    controls.addEventListener('start', markCustomView);
    let gizmoRect = {
      height: VIEW_GIZMO_SIZE,
      left: host.clientWidth - VIEW_GIZMO_RIGHT - VIEW_GIZMO_SIZE,
      top: VIEW_GIZMO_TOP,
      width: VIEW_GIZMO_SIZE,
    };

    const loader = new GLTFLoader();
    let model: THREE.Object3D | null = null;
    let mounted = true;
    loader.load('/mock/test.glb', (gltf) => {
      if (!mounted) {
        return;
      }

      model = gltf.scene;
      orientMockGlbToZUp(model);
      normalizeObjectToCanonicalBox(model);
      applyViewerMaterial(model, colors.mesh);
      scene.add(model);
    });

    const applyTheme = (nextTheme: ThemeMode) => {
      if (nextTheme === activeTheme) {
        return;
      }

      activeTheme = nextTheme;
      colors = viewerColors(activeTheme);

      scene.background = new THREE.Color(colors.background);
      renderer.toneMappingExposure = activeTheme === 'dark' ? 1.08 : 1;
      hemisphereLight.groundColor.set(activeTheme === 'dark' ? 0x343438 : 0xd6d0c6);
      hemisphereLight.intensity = activeTheme === 'dark' ? 0.72 : 0.62;
      ambientLight.intensity = activeTheme === 'dark' ? 0.25 : 0.3;
      keyLight.intensity = activeTheme === 'dark' ? 3.5 : 3;
      fillLight.intensity = activeTheme === 'dark' ? 0.85 : 0.58;
      rimLight.intensity = activeTheme === 'dark' ? 1.4 : 1.05;

      scene.remove(canonicalBox);
      disposeObject(canonicalBox);
      canonicalBox = createCanonicalBox(colors.box);
      updateWorldAxesColors(worldAxes, colors);
      scene.add(canonicalBox);

      disposeObject(viewGizmo.scene);
      viewGizmo = createViewGizmo(colors);
      gizmoMeshes = viewGizmo.pickables.map((pickable) => pickable.object);

      if (model) {
        applyViewerMaterial(model, colors.mesh);
      }

      setSymmetryPreview(activeSymmetryPreview);
    };

    const setOverlays = (
      nextOverlays: SymmetryOverlay[],
      nextSelectedOverlayId: string | null,
      nextSelectableOverlayIds: string[],
    ) => {
      scene.remove(symmetryOverlay.group);
      disposeObject(symmetryOverlay.group);
      symmetryOverlay = createSymmetryOverlayGroup(
        nextOverlays,
        nextSelectedOverlayId,
        nextSelectableOverlayIds,
      );
      symmetryPickables = symmetryOverlay.pickables;
      scene.add(symmetryOverlay.group);
    };

    const setSymmetryPreview = (nextSymmetry: SymmetryTuple | null) => {
      activeSymmetryPreview = nextSymmetry;
      scene.remove(symmetryPreviewGroup);
      disposeObject(symmetryPreviewGroup);
      symmetryPreviewGroup = createSymmetryPreviewGroup(activeSymmetryPreview, colors);
      scene.add(symmetryPreviewGroup);
    };

    runtimeRef.current = { applyTheme, setSymmetryPreview, setOverlays };

    const updateViewport = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      const compact = isCompactViewport();
      const layoutChanged = compact !== previousCompact;
      compactViewport = compact;

      camera.aspect = width / height;
      if (usingDefaultView) {
        setDefaultCameraPosition(compact);
      } else if (layoutChanged) {
        rescaleCameraDistance(compact ? mobileCameraScale : 1 / mobileCameraScale);
      }

      if (compact) {
        const panel = document.querySelector<HTMLElement>('.node-panel-anchor');
        const bottomInset = panel
          ? Math.ceil(height - panel.getBoundingClientRect().top + 12)
          : Math.round(height * 0.45);
        camera.setViewOffset(width, height + bottomInset, 0, bottomInset, width, height);
      } else {
        camera.clearViewOffset();
      }

      gizmoRect = {
        height: VIEW_GIZMO_SIZE,
        left: width - VIEW_GIZMO_RIGHT - VIEW_GIZMO_SIZE,
        top: VIEW_GIZMO_TOP,
        width: VIEW_GIZMO_SIZE,
      };

      controls.update();
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      labelRenderer.setSize(width, height);
      previousCompact = compact;
    };

    const startCameraMove = (target: ViewGizmoTarget) => {
      usingDefaultView = target === 'default';
      const direction = viewDirectionForTarget(target);
      const distance = usingDefaultView
        ? defaultCameraDistance(compactViewport)
        : camera.position.distanceTo(controls.target);
      const targetPosition = controls.target.clone().add(direction.clone().multiplyScalar(distance));

      cameraAnimation = {
        duration: 300,
        startPosition: camera.position.clone(),
        startTime: performance.now(),
        startUp: camera.up.clone(),
        targetPosition,
        targetUp: viewUpForTarget(target, direction, camera.quaternion),
      };
      controls.enabled = false;
    };

    const handlePointerDown = (event: PointerEvent) => {
      const canvasRect = renderer.domElement.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;
      const inside =
        x >= gizmoRect.left &&
        x <= gizmoRect.left + gizmoRect.width &&
        y >= gizmoRect.top &&
        y <= gizmoRect.top + gizmoRect.height;

      if (!compactViewport && inside) {
        event.preventDefault();
        event.stopImmediatePropagation();

        pointer.set(
          ((x - gizmoRect.left) / gizmoRect.width) * 2 - 1,
          -(((y - gizmoRect.top) / gizmoRect.height) * 2 - 1),
        );
        raycaster.setFromCamera(pointer, viewGizmo.camera);

        const hit = raycaster.intersectObjects(gizmoMeshes, false)[0];
        const pickable = viewGizmo.pickables.find((item) => item.object === hit?.object);
        if (pickable) {
          startCameraMove(pickable.target);
        }
        return;
      }

      pointer.set((x / canvasRect.width) * 2 - 1, -(y / canvasRect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);

      const overlayHit = raycaster.intersectObjects(symmetryPickables, false)[0];
      const overlayId = overlayHit?.object.userData.overlayId as string | undefined;
      pendingOverlayPick = overlayId ? { dragged: false, id: overlayId, x: event.clientX, y: event.clientY } : null;
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!pendingOverlayPick) {
        return;
      }

      const dx = event.clientX - pendingOverlayPick.x;
      const dy = event.clientY - pendingOverlayPick.y;
      if (Math.hypot(dx, dy) > 5) {
        pendingOverlayPick.dragged = true;
      }
    };

    const handlePointerUp = (event: PointerEvent) => {
      if (!pendingOverlayPick) {
        return;
      }

      const pick = pendingOverlayPick;
      pendingOverlayPick = null;

      if (pick.dragged) {
        return;
      }

      const canvasRect = renderer.domElement.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;
      pointer.set((x / canvasRect.width) * 2 - 1, -(y / canvasRect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);

      const overlayHit = raycaster.intersectObjects(symmetryPickables, false)[0];
      const overlayId = overlayHit?.object.userData.overlayId as string | undefined;
      if (overlayId === pick.id) {
        onOverlayPickRef.current(pick.id);
      }
    };

    const updateCameraAnimation = (time: number) => {
      if (!cameraAnimation) {
        return false;
      }

      const progress = Math.min((time - cameraAnimation.startTime) / cameraAnimation.duration, 1);
      const eased = progress * progress * (3 - 2 * progress);

      camera.position.copy(cameraAnimation.startPosition).lerp(cameraAnimation.targetPosition, eased);
      camera.up.copy(cameraAnimation.startUp).lerp(cameraAnimation.targetUp, eased).normalize();
      camera.lookAt(controls.target);

      if (progress === 1) {
        cameraAnimation = null;
        controls.enabled = true;
        controls.update();
      }

      return true;
    };

    const renderViewGizmo = () => {
      if (compactViewport) {
        return;
      }

      const width = host.clientWidth;
      const height = host.clientHeight;
      const direction = camera.position.clone().sub(controls.target).normalize();

      viewGizmo.camera.position.copy(direction.multiplyScalar(3));
      viewGizmo.camera.up.copy(camera.up);
      viewGizmo.camera.lookAt(0, 0, 0);

      const autoClear = renderer.autoClear;
      renderer.autoClear = false;
      renderer.clearDepth();
      renderer.setScissorTest(true);
      renderer.setScissor(
        gizmoRect.left,
        height - gizmoRect.top - gizmoRect.height,
        gizmoRect.width,
        gizmoRect.height,
      );
      renderer.setViewport(
        gizmoRect.left,
        height - gizmoRect.top - gizmoRect.height,
        gizmoRect.width,
        gizmoRect.height,
      );
      renderer.render(viewGizmo.scene, viewGizmo.camera);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, width, height);
      renderer.autoClear = autoClear;
    };

    updateViewport();
    window.addEventListener('resize', updateViewport);
    renderer.domElement.addEventListener('pointerdown', handlePointerDown, true);
    renderer.domElement.addEventListener('pointermove', handlePointerMove, true);
    renderer.domElement.addEventListener('pointerup', handlePointerUp, true);

    let frameId = 0;
    const render = (time: number) => {
      frameId = window.requestAnimationFrame(render);
      if (!updateCameraAnimation(time)) {
        controls.update();
      }

      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, host.clientWidth, host.clientHeight);
      renderer.render(scene, camera);
      renderViewGizmo();
      labelRenderer.render(scene, camera);
    };
    frameId = window.requestAnimationFrame(render);

    return () => {
      mounted = false;
      runtimeRef.current = null;
      window.cancelAnimationFrame(frameId);
      window.removeEventListener('resize', updateViewport);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown, true);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove, true);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp, true);
      controls.removeEventListener('start', markCustomView);
      controls.dispose();
      disposeObject(scene);
      disposeObject(viewGizmo.scene);
      renderer.dispose();
      host.removeChild(labelRenderer.domElement);
      host.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    runtimeRef.current?.applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    runtimeRef.current?.setOverlays(overlays, selectedOverlayId, selectableOverlayIds);
  }, [overlays, selectableOverlayIds, selectedOverlayId]);

  useEffect(() => {
    runtimeRef.current?.setSymmetryPreview(symmetryPreview);
  }, [symmetryPreview]);

  return (
    <main className="three-viewer" aria-label="3D viewer" data-viewer-theme={theme} ref={hostRef}>
      <div className="viewport-readout">
        <span>SymTRELLIS</span>
        <span>{theme}</span>
      </div>
    </main>
  );
}

type DisposableObject = THREE.Object3D & {
  geometry?: THREE.BufferGeometry;
  material?: THREE.Material | THREE.Material[];
};

type TexturedMaterial = THREE.Material & {
  map?: THREE.Texture | null;
};

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof CSS2DObject) {
      child.element.remove();
    }

    const disposable = child as DisposableObject;
    disposable.geometry?.dispose();

    if (Array.isArray(disposable.material)) {
      disposable.material.forEach(disposeMaterial);
    } else {
      disposable.material && disposeMaterial(disposable.material);
    }
  });
}

function disposeMaterial(material: THREE.Material) {
  const texture = (material as TexturedMaterial).map;
  texture?.dispose();
  material.dispose();
}
