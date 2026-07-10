import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { ThemeMode } from '../types';
import type { ViewerContent } from './viewerTypes';
import { createGlbContentManager } from './glbContent';
import { createSymmetryOverlayGroup } from './symmetryOverlays';
import { createSymmetryPreviewGroup } from './symmetryPreview';
import {
  createCanonicalBox,
  createViewerLights,
  createWorldAxes,
  disposeObject,
  updateWorldAxesColors,
} from './scenePrimitives';
import {
  DEFAULT_CAMERA_DIRECTION,
  VIEW_GIZMO_RIGHT,
  VIEW_GIZMO_SIZE,
  VIEW_GIZMO_TOP,
  createViewGizmo,
  viewDirectionForTarget,
} from './viewGizmo';
import type { ViewGizmoTarget } from './viewGizmo';
import { viewerColors } from './viewerTheme';

type ThreeViewerProps = {
  content: ViewerContent;
  dagVisible: boolean;
  onOverlayPicked?: (overlayId: string) => void;
  theme: ThemeMode;
};

type CameraAnimation = {
  duration: number;
  lockTopView: boolean;
  startPosition: THREE.Vector3;
  startQuaternion: THREE.Quaternion;
  startTime: number;
  targetPosition: THREE.Vector3;
  targetQuaternion: THREE.Quaternion;
};

type LockedTopDrag = {
  distance: number;
  moved: boolean;
  right: THREE.Vector3;
  startX: number;
  startY: number;
  up: THREE.Vector3;
  zAxis: THREE.Vector3;
};

type ViewInsets = { bottom: number; left: number; right: number };

type ViewInsetAnimation = { start: ViewInsets; startTime: number; target: ViewInsets };

type ViewerRuntime = {
  applyContent: (content: ViewerContent) => void;
  applyTheme: (theme: ThemeMode) => void;
  refreshLayout: (animate: boolean) => void;
};

export function ThreeViewer({ content, dagVisible, onOverlayPicked, theme }: ThreeViewerProps) {
  const hostRef = useRef<HTMLElement>(null);
  const runtimeRef = useRef<ViewerRuntime | null>(null);
  const initialContentRef = useRef(content);
  const initialThemeRef = useRef(theme);
  const onOverlayPickedRef = useRef(onOverlayPicked);

  useEffect(() => {
    onOverlayPickedRef.current = onOverlayPicked;
  }, [onOverlayPicked]);

  useEffect(() => {
    const host = hostRef.current;

    if (!host) {
      return;
    }

    let activeTheme = initialThemeRef.current;
    let colors = viewerColors(activeTheme);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(colors.background);

    const camera = new THREE.PerspectiveCamera(38, host.clientWidth / host.clientHeight, 0.01, 100);
    const worldUp = new THREE.Vector3(0, 0, 1);
    camera.up.copy(worldUp);

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

    let lights = createViewerLights(activeTheme);
    let canonicalBox = createCanonicalBox(colors.box);
    const worldAxes = createWorldAxes(colors);
    scene.add(lights, canonicalBox, worldAxes);
    let activeContent = initialContentRef.current;
    const glbContent = createGlbContentManager(scene);
    glbContent.setContent(activeContent.glb, colors);
    let overlayRender = createSymmetryOverlayGroup(
      activeContent.overlays,
      activeContent.selectedOverlayId,
      activeContent.selectableOverlayIds,
    );
    let overlayPickables = overlayRender.pickables;
    let symmetryPreview = createSymmetryPreviewGroup(activeContent.symmetryPreview, colors);
    scene.add(overlayRender.group, symmetryPreview);

    let viewGizmo = createViewGizmo(colors);
    let gizmoMeshes = viewGizmo.pickables.map((pickable) => pickable.object);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let cameraAnimation: CameraAnimation | null = null;
    let lockedTopDrag: LockedTopDrag | null = null;
    let lockedTopView = false;
    let viewInsetAnimation: ViewInsetAnimation | null = null;
    let viewInsets: ViewInsets = { bottom: 0, left: 0, right: 0 };
    let usingDefaultView = true;
    const markCustomView = () => {
      usingDefaultView = false;
    };
    controls.addEventListener('start', markCustomView);
    let overlayPointerDown: { x: number; y: number } | null = null;

    let gizmoRect = {
      height: VIEW_GIZMO_SIZE,
      left: host.clientWidth - VIEW_GIZMO_RIGHT - VIEW_GIZMO_SIZE,
      top: VIEW_GIZMO_TOP,
      width: VIEW_GIZMO_SIZE,
    };

    const applyTheme = (nextTheme: ThemeMode) => {
      if (nextTheme === activeTheme) {
        return;
      }

      activeTheme = nextTheme;
      colors = viewerColors(activeTheme);
      scene.background = new THREE.Color(colors.background);
      renderer.toneMappingExposure = activeTheme === 'dark' ? 1.08 : 1;

      scene.remove(lights, canonicalBox);
      disposeObject(canonicalBox);
      lights = createViewerLights(activeTheme);
      canonicalBox = createCanonicalBox(colors.box);
      scene.add(lights, canonicalBox);

      updateWorldAxesColors(worldAxes, colors);

      disposeObject(viewGizmo.scene);
      viewGizmo = createViewGizmo(colors);
      gizmoMeshes = viewGizmo.pickables.map((pickable) => pickable.object);
      glbContent.applyTheme(colors);

      scene.remove(overlayRender.group, symmetryPreview);
      disposeObject(overlayRender.group);
      disposeObject(symmetryPreview);
      overlayRender = createSymmetryOverlayGroup(
        activeContent.overlays,
        activeContent.selectedOverlayId,
        activeContent.selectableOverlayIds,
      );
      overlayPickables = overlayRender.pickables;
      symmetryPreview = createSymmetryPreviewGroup(activeContent.symmetryPreview, colors);
      scene.add(overlayRender.group, symmetryPreview);
    };

    const applyContent = (nextContent: ViewerContent) => {
      activeContent = nextContent;
      glbContent.setContent(activeContent.glb, colors);
      scene.remove(overlayRender.group, symmetryPreview);
      disposeObject(overlayRender.group);
      disposeObject(symmetryPreview);
      overlayRender = createSymmetryOverlayGroup(
        activeContent.overlays,
        activeContent.selectedOverlayId,
        activeContent.selectableOverlayIds,
      );
      overlayPickables = overlayRender.pickables;
      symmetryPreview = createSymmetryPreviewGroup(activeContent.symmetryPreview, colors);
      scene.add(overlayRender.group, symmetryPreview);
    };

    const applyViewInsets = (insets: ViewInsets) => {
      const width = host.clientWidth;
      const height = host.clientHeight;

      if (compactViewport) {
        camera.setViewOffset(width, height + insets.bottom, 0, insets.bottom, width, height);
      } else if (insets.left > 0 || insets.right > 0) {
        camera.setViewOffset(
          width + insets.left + insets.right,
          height,
          insets.right,
          0,
          width,
          height,
        );
      } else {
        camera.clearViewOffset();
      }

      camera.updateProjectionMatrix();
    };

    const updateViewport = (animateInsets = false) => {
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

      const panel = document.querySelector<HTMLElement>('.node-panel-anchor');
      const panelRect = panel?.getBoundingClientRect();
      const dag = document.querySelector<HTMLElement>('.dag-anchor');
      const dagRect = dag?.getBoundingClientRect();
      const nextInsets: ViewInsets = compact
        ? {
            bottom: panelRect
              ? Math.ceil(height - panelRect.top + 12)
              : Math.round(height * 0.45),
            left: 0,
            right: 0,
          }
        : {
            bottom: 0,
            left: panelRect && panelRect.right > 0 ? Math.ceil(panelRect.right + 12) : 0,
            right: dagRect ? Math.ceil(width - dagRect.left + 12) : 0,
          };

      const insetsChanged =
        nextInsets.bottom !== viewInsets.bottom ||
        nextInsets.left !== viewInsets.left ||
        nextInsets.right !== viewInsets.right;
      if (animateInsets && insetsChanged) {
        viewInsetAnimation = {
          start: { ...viewInsets },
          startTime: performance.now(),
          target: nextInsets,
        };
      } else {
        viewInsetAnimation = null;
        viewInsets = nextInsets;
        applyViewInsets(viewInsets);
      }

      gizmoRect = {
        height: VIEW_GIZMO_SIZE,
        left: width - VIEW_GIZMO_RIGHT - VIEW_GIZMO_SIZE,
        top: VIEW_GIZMO_TOP,
        width: VIEW_GIZMO_SIZE,
      };

      controls.update();
      renderer.setSize(width, height);
      labelRenderer.setSize(width, height);
      previousCompact = compact;
    };

    const updateViewInsetAnimation = (time: number) => {
      if (!viewInsetAnimation) {
        return;
      }

      const progress = Math.min((time - viewInsetAnimation.startTime) / 150, 1);
      const eased = progress * progress * (3 - 2 * progress);
      viewInsets = {
        bottom:
          viewInsetAnimation.start.bottom +
          (viewInsetAnimation.target.bottom - viewInsetAnimation.start.bottom) * eased,
        left:
          viewInsetAnimation.start.left +
          (viewInsetAnimation.target.left - viewInsetAnimation.start.left) * eased,
        right:
          viewInsetAnimation.start.right +
          (viewInsetAnimation.target.right - viewInsetAnimation.start.right) * eased,
      };
      applyViewInsets(viewInsets);

      if (progress === 1) {
        viewInsets = viewInsetAnimation.target;
        viewInsetAnimation = null;
        applyViewInsets(viewInsets);
      }
    };

    runtimeRef.current = {
      applyContent,
      applyTheme,
      refreshLayout: (animate) => updateViewport(animate),
    };
    const handleResize = () => updateViewport(false);

    const startCameraMove = (target: ViewGizmoTarget) => {
      usingDefaultView = target === 'default';
      lockedTopDrag = null;
      lockedTopView = false;
      const direction = viewDirectionForTarget(target);
      const distance = usingDefaultView
        ? defaultCameraDistance(compactViewport)
        : camera.position.distanceTo(controls.target);
      const targetPosition = controls.target.clone().add(direction.clone().multiplyScalar(distance));
      const right = new THREE.Vector3();
      if (target === 'z+' || target === 'z-') {
        right.set(1, 0, 0).applyQuaternion(camera.quaternion);
        right.addScaledVector(direction, -right.dot(direction)).normalize();
      } else {
        right.crossVectors(worldUp, direction).normalize();
      }
      const up = new THREE.Vector3().crossVectors(direction, right).normalize();
      const targetQuaternion = new THREE.Quaternion().setFromRotationMatrix(
        new THREE.Matrix4().makeBasis(right, up, direction),
      );

      cameraAnimation = {
        duration: 300,
        lockTopView: target === 'z+' || target === 'z-',
        startPosition: camera.position.clone(),
        startQuaternion: camera.quaternion.clone(),
        startTime: performance.now(),
        targetPosition,
        targetQuaternion,
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

      if (lockedTopView) {
        event.preventDefault();
        event.stopImmediatePropagation();

        lockedTopDrag = {
          distance: camera.position.distanceTo(controls.target),
          moved: false,
          right: new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize(),
          startX: event.clientX,
          startY: event.clientY,
          up: new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize(),
          zAxis: camera.position.clone().sub(controls.target).normalize(),
        };
        overlayPointerDown = null;
        return;
      }

      overlayPointerDown = { x: event.clientX, y: event.clientY };
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!lockedTopDrag) {
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();

      const dx = event.clientX - lockedTopDrag.startX;
      const dy = event.clientY - lockedTopDrag.startY;
      if (Math.hypot(dx, dy) <= 1) {
        return;
      }

      lockedTopDrag.moved = true;
      const dragScale = 0.006;
      const direction = lockedTopDrag.zAxis
        .clone()
        .addScaledVector(lockedTopDrag.right, -dx * dragScale)
        .addScaledVector(lockedTopDrag.up, dy * dragScale)
        .normalize();
      camera.position.copy(controls.target).add(direction.multiplyScalar(lockedTopDrag.distance));
      camera.up.copy(worldUp);
      camera.lookAt(controls.target);
    };

    const handlePointerUp = (event: PointerEvent) => {
      if (lockedTopDrag) {
        event.preventDefault();
        event.stopImmediatePropagation();

        const moved = lockedTopDrag.moved;
        lockedTopDrag = null;
        if (moved) {
          lockedTopView = false;
          controls.enabled = true;
          controls.update();
        }
        return;
      }

      if (!overlayPointerDown || overlayPickables.length === 0) {
        overlayPointerDown = null;
        return;
      }

      const moved = Math.hypot(event.clientX - overlayPointerDown.x, event.clientY - overlayPointerDown.y);
      overlayPointerDown = null;

      if (moved > 4) {
        return;
      }

      const canvasRect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - canvasRect.left) / canvasRect.width) * 2 - 1,
        -(((event.clientY - canvasRect.top) / canvasRect.height) * 2 - 1),
      );
      raycaster.setFromCamera(pointer, camera);

      const hit = raycaster.intersectObjects(overlayPickables, false)[0];
      const overlayId = hit?.object.userData.overlayId as string | undefined;
      if (overlayId) {
        onOverlayPickedRef.current?.(overlayId);
      }
    };

    const updateCameraAnimation = (time: number) => {
      if (!cameraAnimation) {
        return false;
      }

      const progress = Math.min((time - cameraAnimation.startTime) / cameraAnimation.duration, 1);
      const eased = progress * progress * (3 - 2 * progress);

      camera.position.copy(cameraAnimation.startPosition).lerp(cameraAnimation.targetPosition, eased);
      camera.quaternion.slerpQuaternions(
        cameraAnimation.startQuaternion,
        cameraAnimation.targetQuaternion,
        eased,
      );
      camera.up.copy(worldUp);

      if (progress === 1) {
        const lockTopView = cameraAnimation.lockTopView;
        cameraAnimation = null;
        lockedTopView = lockTopView;
        controls.enabled = !lockTopView;
        if (!lockTopView) {
          controls.update();
        }
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
      viewGizmo.camera.quaternion.copy(camera.quaternion);

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
    window.addEventListener('resize', handleResize);
    renderer.domElement.addEventListener('pointerdown', handlePointerDown, true);
    renderer.domElement.addEventListener('pointermove', handlePointerMove, true);
    renderer.domElement.addEventListener('pointerup', handlePointerUp);

    let frameId = 0;
    const render = (time: number) => {
      frameId = window.requestAnimationFrame(render);
      updateViewInsetAnimation(time);
      if (!updateCameraAnimation(time)) {
        if (!lockedTopView && !lockedTopDrag) {
          controls.update();
        }
      }

      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, host.clientWidth, host.clientHeight);
      renderer.render(scene, camera);
      renderViewGizmo();
      labelRenderer.render(scene, camera);
    };
    frameId = window.requestAnimationFrame(render);

    return () => {
      runtimeRef.current = null;
      window.cancelAnimationFrame(frameId);
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown, true);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove, true);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp);
      controls.removeEventListener('start', markCustomView);
      controls.dispose();
      glbContent.dispose();
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
    runtimeRef.current?.applyContent(content);
  }, [content]);

  useEffect(() => {
    runtimeRef.current?.refreshLayout(true);
  }, [dagVisible]);

  return (
    <main className="three-viewer" aria-label="3D viewer" data-viewer-theme={theme} ref={hostRef}>
      <div className="viewport-readout">
        <span>
          SymTRELLIS (
          <a href="https://arxiv.org/abs/2606.04108" target="_blank" rel="noreferrer">
            paper
          </a>
          |
          <a href="https://github.com/quantaji/SymTRELLIS" target="_blank" rel="noreferrer">
            code
          </a>
          )
        </span>
        <span>{theme}</span>
      </div>
    </main>
  );
}
