import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { NodeEntry } from './nodes';
import type { GraphEdge } from './types';

export interface EdgeEntry {
  line: THREE.Line;
  labelObject: CSS2DObject;
  labelElement: HTMLDivElement;
  source: string;
  target: string;
}

const EDGE_COLOR = 0x999999;
const EDGE_HOVER_COLOR = 0x4a90d9;
const EDGE_OPACITY = 0.6;
const CURVE_SAMPLES = 30;
const CURVATURE = 0.15;

function injectStyles(): void {
  const styleId = 'graph-edge-styles';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    .edge-label {
      font-size: 9px;
      background: rgba(100, 100, 100, 0.85);
      color: white;
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
      pointer-events: auto;
      opacity: 0;
      transition: opacity 0.2s;
    }
    .edge-label:hover {
      opacity: 1;
    }
  `;
  document.head.appendChild(style);
}

/** Track how many edges share each node pair so we can alternate curvature direction. */
function buildPairCounts(edges: GraphEdge[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const edge of edges) {
    const pairKey = [edge.source, edge.target].sort().join('::');
    counts.set(pairKey, (counts.get(pairKey) ?? 0) + 1);
  }
  return counts;
}

function computeBezierPoints(
  sourcePos: THREE.Vector3,
  targetPos: THREE.Vector3,
  pairIndex: number,
): THREE.Vector3[] {
  const mid = new THREE.Vector3().addVectors(sourcePos, targetPos).multiplyScalar(0.5);
  const delta = new THREE.Vector3().subVectors(targetPos, sourcePos);
  const distance = delta.length();

  // Perpendicular direction in 2D (rotate 90 degrees in XY plane)
  const perp = new THREE.Vector3(-delta.y, delta.x, 0).normalize();
  const sign = pairIndex % 2 === 0 ? 1 : -1;
  const offset = CURVATURE * distance * sign * (Math.floor(pairIndex / 2) + 1);

  const controlPoint = mid.clone().addScaledVector(perp, offset);
  controlPoint.z = -1;

  const curve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(sourcePos.x, sourcePos.y, -1),
    controlPoint,
    new THREE.Vector3(targetPos.x, targetPos.y, -1),
  );

  return curve.getPoints(CURVE_SAMPLES);
}

function getMidpoint(
  sourcePos: THREE.Vector3,
  targetPos: THREE.Vector3,
  pairIndex: number,
): THREE.Vector3 {
  const mid = new THREE.Vector3().addVectors(sourcePos, targetPos).multiplyScalar(0.5);
  const delta = new THREE.Vector3().subVectors(targetPos, sourcePos);
  const distance = delta.length();
  const perp = new THREE.Vector3(-delta.y, delta.x, 0).normalize();
  const sign = pairIndex % 2 === 0 ? 1 : -1;
  const offset = CURVATURE * distance * sign * (Math.floor(pairIndex / 2) + 1);

  const controlPoint = mid.clone().addScaledVector(perp, offset);

  // Quadratic bezier at t=0.5: 0.25*P0 + 0.5*C + 0.25*P2
  return new THREE.Vector3(
    0.25 * sourcePos.x + 0.5 * controlPoint.x + 0.25 * targetPos.x,
    0.25 * sourcePos.y + 0.5 * controlPoint.y + 0.25 * targetPos.y,
    -0.5,
  );
}

export function createEdges(
  scene: THREE.Scene,
  edges: GraphEdge[],
  nodeMap: Map<string, NodeEntry>,
): Map<string, EdgeEntry> {
  injectStyles();

  const edgeMap = new Map<string, EdgeEntry>();
  const pairCounts = buildPairCounts(edges);
  const pairIndices = new Map<string, number>();

  for (const edge of edges) {
    const sourceEntry = nodeMap.get(edge.source);
    const targetEntry = nodeMap.get(edge.target);
    if (!sourceEntry || !targetEntry) continue;

    const sourcePos = sourceEntry.mesh.position;
    const targetPos = targetEntry.mesh.position;

    // Determine index within this node pair
    const pairKey = [edge.source, edge.target].sort().join('::');
    const pairIndex = pairIndices.get(pairKey) ?? 0;
    pairIndices.set(pairKey, pairIndex + 1);

    // --- Bezier curve line ---
    const points = computeBezierPoints(sourcePos, targetPos, pairIndex);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: EDGE_COLOR,
      transparent: true,
      opacity: EDGE_OPACITY,
    });
    const line = new THREE.Line(geometry, material);
    scene.add(line);

    // --- Edge label ---
    const labelElement = document.createElement('div');
    labelElement.className = 'edge-label';
    labelElement.textContent = edge.label;
    labelElement.style.pointerEvents = 'auto';

    const labelObject = new CSS2DObject(labelElement);
    const labelPos = getMidpoint(sourcePos, targetPos, pairIndex);
    labelObject.position.copy(labelPos);
    scene.add(labelObject);

    // --- Hover behavior ---
    labelElement.addEventListener('mouseenter', () => {
      (line.material as THREE.LineBasicMaterial).color.setHex(EDGE_HOVER_COLOR);
      (line.material as THREE.LineBasicMaterial).opacity = 1.0;
    });
    labelElement.addEventListener('mouseleave', () => {
      (line.material as THREE.LineBasicMaterial).color.setHex(EDGE_COLOR);
      (line.material as THREE.LineBasicMaterial).opacity = EDGE_OPACITY;
    });

    // Store the pairIndex on the entry for updateEdge
    const entry: EdgeEntry & { _pairIndex: number } = {
      line,
      labelObject,
      labelElement,
      source: edge.source,
      target: edge.target,
      _pairIndex: pairIndex,
    };
    edgeMap.set(edge.id, entry);
  }

  return edgeMap;
}

export function updateEdge(
  edgeId: string,
  edgeMap: Map<string, EdgeEntry>,
  nodeMap: Map<string, NodeEntry>,
): void {
  const entry = edgeMap.get(edgeId) as (EdgeEntry & { _pairIndex?: number }) | undefined;
  if (!entry) return;

  const sourceEntry = nodeMap.get(entry.source);
  const targetEntry = nodeMap.get(entry.target);
  if (!sourceEntry || !targetEntry) return;

  const sourcePos = sourceEntry.mesh.position;
  const targetPos = targetEntry.mesh.position;
  const pairIndex = entry._pairIndex ?? 0;

  // Recompute line geometry
  const newPoints = computeBezierPoints(sourcePos, targetPos, pairIndex);
  entry.line.geometry.dispose();
  entry.line.geometry = new THREE.BufferGeometry().setFromPoints(newPoints);

  // Recompute label position
  const labelPos = getMidpoint(sourcePos, targetPos, pairIndex);
  entry.labelObject.position.copy(labelPos);
}
