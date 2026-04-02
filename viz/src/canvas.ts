import * as THREE from 'three';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

const BACKGROUND_COLOR = 0xf8f9fa;
const GRID_COLOR = 0xe0e0e0;
const GRID_SPACING = 200;
const GRID_EXTENT = 2000;
const ZOOM_FACTOR = 1.1;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5.0;

const container = document.getElementById('app')!;

// --- Scene ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(BACKGROUND_COLOR);

// --- Camera ---
const aspect = container.clientWidth / container.clientHeight;
const halfWidth = 1000;
const halfHeight = halfWidth / aspect;
const camera = new THREE.OrthographicCamera(
  -halfWidth, halfWidth, halfHeight, -halfHeight, 0.1, 100
);
camera.position.set(0, 0, 10);
camera.lookAt(0, 0, 0);

// --- WebGL Renderer ---
const webglRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
webglRenderer.setPixelRatio(window.devicePixelRatio);
webglRenderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(webglRenderer.domElement);

// --- CSS2D Renderer ---
const css2dRenderer = new CSS2DRenderer();
css2dRenderer.setSize(container.clientWidth, container.clientHeight);
css2dRenderer.domElement.style.position = 'absolute';
css2dRenderer.domElement.style.top = '0';
css2dRenderer.domElement.style.left = '0';
css2dRenderer.domElement.style.pointerEvents = 'none';
container.appendChild(css2dRenderer.domElement);

// --- Grid ---
function createGrid(): void {
  const material = new THREE.LineBasicMaterial({ color: GRID_COLOR, transparent: true, opacity: 0.5 });
  const points: THREE.Vector3[] = [];

  for (let i = -GRID_EXTENT; i <= GRID_EXTENT; i += GRID_SPACING) {
    // Vertical lines
    points.push(new THREE.Vector3(i, -GRID_EXTENT, -2));
    points.push(new THREE.Vector3(i, GRID_EXTENT, -2));
    // Horizontal lines
    points.push(new THREE.Vector3(-GRID_EXTENT, i, -2));
    points.push(new THREE.Vector3(GRID_EXTENT, i, -2));
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const grid = new THREE.LineSegments(geometry, material);
  scene.add(grid);
}
createGrid();

// --- Render on demand ---
let dirty = true;

function markDirty(): void {
  dirty = true;
}

function renderLoop(): void {
  requestAnimationFrame(renderLoop);
  if (dirty) {
    webglRenderer.render(scene, camera);
    css2dRenderer.render(scene, camera);
    dirty = false;
  }
}
renderLoop();

// --- Helper: world position from screen coords ---
function getWorldPosition(screenX: number, screenY: number): THREE.Vector2 {
  const rect = container.getBoundingClientRect();
  const normalizedX = (screenX - rect.left) / rect.width;
  const normalizedY = (screenY - rect.top) / rect.height;
  const worldX = camera.left + normalizedX * (camera.right - camera.left) + camera.position.x;
  const worldY = camera.top + normalizedY * (camera.bottom - camera.top) + camera.position.y;
  return new THREE.Vector2(worldX, worldY);
}

// --- Zoom level tracking ---
let zoomLevel = 1.0;

function applyZoom(newZoom: number): void {
  const baseHalfWidth = halfWidth;
  const baseHalfHeight = halfHeight;
  camera.left = -baseHalfWidth / newZoom;
  camera.right = baseHalfWidth / newZoom;
  camera.top = baseHalfHeight / newZoom;
  camera.bottom = -baseHalfHeight / newZoom;
  camera.updateProjectionMatrix();
}

// --- Pan ---
let isPanning = false;
let panStartX = 0;
let panStartY = 0;

container.addEventListener('pointerdown', (event: PointerEvent) => {
  // Only pan on primary button and when clicking the canvas background
  if (event.button !== 0) return;
  const target = event.target as HTMLElement;
  if (target.closest('.graph-node') || target.closest('.edge-label')) return;

  isPanning = true;
  panStartX = event.clientX;
  panStartY = event.clientY;
  container.setPointerCapture(event.pointerId);
});

container.addEventListener('pointermove', (event: PointerEvent) => {
  if (!isPanning) return;

  const deltaScreenX = event.clientX - panStartX;
  const deltaScreenY = event.clientY - panStartY;

  const worldPerPixelX = (camera.right - camera.left) / container.clientWidth;
  const worldPerPixelY = (camera.top - camera.bottom) / container.clientHeight;

  camera.position.x -= deltaScreenX * worldPerPixelX;
  camera.position.y += deltaScreenY * worldPerPixelY;
  camera.updateProjectionMatrix();

  panStartX = event.clientX;
  panStartY = event.clientY;
  markDirty();
});

container.addEventListener('pointerup', (event: PointerEvent) => {
  if (isPanning) {
    isPanning = false;
    container.releasePointerCapture(event.pointerId);
  }
});

// --- Zoom toward cursor ---
container.addEventListener('wheel', (event: WheelEvent) => {
  event.preventDefault();

  const worldBefore = getWorldPosition(event.clientX, event.clientY);

  const direction = event.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
  zoomLevel = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoomLevel * direction));
  applyZoom(zoomLevel);

  const worldAfter = getWorldPosition(event.clientX, event.clientY);

  camera.position.x += worldBefore.x - worldAfter.x;
  camera.position.y += worldBefore.y - worldAfter.y;
  camera.updateProjectionMatrix();
  markDirty();
}, { passive: false });

// --- Resize ---
const resizeObserver = new ResizeObserver(() => {
  const width = container.clientWidth;
  const height = container.clientHeight;
  if (width === 0 || height === 0) return;

  webglRenderer.setSize(width, height);
  css2dRenderer.setSize(width, height);

  const newAspect = width / height;
  const baseHalfWidth = halfWidth;
  const baseHalfHeight = baseHalfWidth / newAspect;

  camera.left = -baseHalfHeight * newAspect / zoomLevel;
  camera.right = baseHalfHeight * newAspect / zoomLevel;
  camera.top = baseHalfHeight / zoomLevel;
  camera.bottom = -baseHalfHeight / zoomLevel;
  camera.updateProjectionMatrix();
  markDirty();
});
resizeObserver.observe(container);

export { scene, camera, webglRenderer, css2dRenderer, markDirty, getWorldPosition };
