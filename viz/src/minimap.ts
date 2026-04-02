import * as THREE from 'three';

const MINIMAP_WIDTH = 180;
const MINIMAP_HEIGHT = 120;

// Frustum encompassing all nodes with padding
const WORLD_LEFT = -1100;
const WORLD_RIGHT = 1100;
const WORLD_TOP = 700;
const WORLD_BOTTOM = -800;

export function setupMinimap(
  scene: THREE.Scene,
  mainCamera: THREE.OrthographicCamera,
  markDirtyFn: () => void,
): { updateMinimap: () => void } {
  // --- Minimap canvas ---
  const minimapCanvas = document.createElement('canvas');
  minimapCanvas.width = MINIMAP_WIDTH * window.devicePixelRatio;
  minimapCanvas.height = MINIMAP_HEIGHT * window.devicePixelRatio;
  minimapCanvas.style.width = `${MINIMAP_WIDTH}px`;
  minimapCanvas.style.height = `${MINIMAP_HEIGHT}px`;
  minimapCanvas.style.position = 'fixed';
  minimapCanvas.style.bottom = '16px';
  minimapCanvas.style.right = '16px';
  minimapCanvas.style.border = '1px solid #ccc';
  minimapCanvas.style.borderRadius = '4px';
  minimapCanvas.style.background = 'white';
  minimapCanvas.style.zIndex = '100';
  minimapCanvas.style.cursor = 'pointer';
  document.body.appendChild(minimapCanvas);

  // --- Minimap renderer ---
  const minimapRenderer = new THREE.WebGLRenderer({
    canvas: minimapCanvas,
    antialias: true,
    alpha: false,
  });
  minimapRenderer.setPixelRatio(window.devicePixelRatio);
  minimapRenderer.setSize(MINIMAP_WIDTH, MINIMAP_HEIGHT);
  minimapRenderer.setClearColor(0xffffff, 1);

  // --- Minimap camera ---
  const minimapCamera = new THREE.OrthographicCamera(
    WORLD_LEFT, WORLD_RIGHT, WORLD_TOP, WORLD_BOTTOM, 0.1, 100,
  );
  minimapCamera.position.set(0, 0, 10);
  minimapCamera.lookAt(0, 0, 0);

  // --- Viewport indicator ---
  const viewportGeometry = new THREE.BufferGeometry();
  const viewportPositions = new Float32Array(4 * 3); // 4 corners, xyz each
  viewportGeometry.setAttribute('position', new THREE.BufferAttribute(viewportPositions, 3));
  viewportGeometry.setIndex([0, 1, 1, 2, 2, 3, 3, 0]);
  const viewportMaterial = new THREE.LineBasicMaterial({
    color: 0x2980b9,
    linewidth: 1,
  });
  const viewportIndicator = new THREE.LineSegments(viewportGeometry, viewportMaterial);
  viewportIndicator.frustumCulled = false;

  function updateMinimap(): void {
    // Update viewport indicator corners from main camera
    const left = mainCamera.left + mainCamera.position.x;
    const right = mainCamera.right + mainCamera.position.x;
    const top = mainCamera.top + mainCamera.position.y;
    const bottom = mainCamera.bottom + mainCamera.position.y;
    const z = 5; // above scene content

    const positions = viewportGeometry.attributes.position as THREE.BufferAttribute;
    positions.setXYZ(0, left, top, z);
    positions.setXYZ(1, right, top, z);
    positions.setXYZ(2, right, bottom, z);
    positions.setXYZ(3, left, bottom, z);
    positions.needsUpdate = true;

    // Temporarily add viewport indicator, render, then remove
    scene.add(viewportIndicator);
    minimapRenderer.render(scene, minimapCamera);
    scene.remove(viewportIndicator);
  }

  // --- Click-to-navigate ---
  minimapCanvas.addEventListener('click', (event: MouseEvent) => {
    const rect = minimapCanvas.getBoundingClientRect();
    const normalizedX = (event.clientX - rect.left) / rect.width;
    const normalizedY = (event.clientY - rect.top) / rect.height;

    const worldX = WORLD_LEFT + normalizedX * (WORLD_RIGHT - WORLD_LEFT);
    const worldY = WORLD_TOP + normalizedY * (WORLD_BOTTOM - WORLD_TOP);

    mainCamera.position.x = worldX;
    mainCamera.position.y = worldY;
    mainCamera.updateProjectionMatrix();
    markDirtyFn();
  });

  return { updateMinimap };
}
