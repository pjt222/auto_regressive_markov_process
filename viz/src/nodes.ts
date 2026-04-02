import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { GraphNode, STATUS_COLORS } from './types';

export interface NodeEntry {
  mesh: THREE.Mesh;
  css2dObject: CSS2DObject;
  element: HTMLDivElement;
}

const NODE_WIDTH = 180;
const NODE_HEIGHT = 80;
const CORNER_RADIUS = 8;
const BORDER_PADDING = 3;

function createRoundedRectShape(width: number, height: number, radius: number): THREE.Shape {
  const halfW = width / 2;
  const halfH = height / 2;
  const shape = new THREE.Shape();

  shape.moveTo(-halfW + radius, -halfH);
  shape.lineTo(halfW - radius, -halfH);
  shape.quadraticCurveTo(halfW, -halfH, halfW, -halfH + radius);
  shape.lineTo(halfW, halfH - radius);
  shape.quadraticCurveTo(halfW, halfH, halfW - radius, halfH);
  shape.lineTo(-halfW + radius, halfH);
  shape.quadraticCurveTo(-halfW, halfH, -halfW, halfH - radius);
  shape.lineTo(-halfW, -halfH + radius);
  shape.quadraticCurveTo(-halfW, -halfH, -halfW + radius, -halfH);

  return shape;
}

function injectStyles(): void {
  const styleId = 'graph-node-styles';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    .graph-node {
      max-width: 200px;
      padding: 8px 12px;
      font-size: 11px;
      border-radius: 8px;
      background: transparent;
      text-align: center;
      cursor: pointer;
      user-select: none;
      transition: all 0.2s;
    }
    .node-title {
      font-weight: bold;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .node-formula {
      font-size: 10px;
      opacity: 0.85;
      overflow: hidden;
      max-height: 40px;
    }
    .node-summary {
      font-size: 10px;
      color: #666;
      margin-top: 4px;
    }
  `;
  document.head.appendChild(style);
}

export function createNodes(scene: THREE.Scene, nodes: GraphNode[]): Map<string, NodeEntry> {
  injectStyles();

  const nodeMap = new Map<string, NodeEntry>();

  for (const node of nodes) {
    const colors = STATUS_COLORS[node.status];

    // --- Fill mesh ---
    const fillShape = createRoundedRectShape(NODE_WIDTH, NODE_HEIGHT, CORNER_RADIUS);
    const fillGeometry = new THREE.ShapeGeometry(fillShape);
    const fillMaterial = new THREE.MeshBasicMaterial({ color: colors.fill });
    const fillMesh = new THREE.Mesh(fillGeometry, fillMaterial);
    fillMesh.position.set(node.position.x, node.position.y, 0);

    // --- Border mesh (slightly larger, behind) ---
    const borderShape = createRoundedRectShape(
      NODE_WIDTH + BORDER_PADDING * 2,
      NODE_HEIGHT + BORDER_PADDING * 2,
      CORNER_RADIUS + 1
    );
    const borderGeometry = new THREE.ShapeGeometry(borderShape);
    const borderMaterial = new THREE.MeshBasicMaterial({ color: colors.border });
    const borderMesh = new THREE.Mesh(borderGeometry, borderMaterial);
    borderMesh.position.set(0, 0, -0.1);
    fillMesh.add(borderMesh);

    scene.add(fillMesh);

    // --- HTML overlay via CSS2DObject ---
    const element = document.createElement('div');
    element.className = 'graph-node';
    element.dataset.nodeId = node.id;
    element.dataset.status = node.status;

    const titleDiv = document.createElement('div');
    titleDiv.className = 'node-title';
    titleDiv.style.color = colors.text;
    titleDiv.textContent = node.title;
    element.appendChild(titleDiv);

    if (node.formulas.length > 0) {
      const formulaDiv = document.createElement('div');
      formulaDiv.className = 'node-formula';
      formulaDiv.innerHTML = node.formulas
        .map(formula => katex.renderToString(formula, { throwOnError: false, displayMode: false }))
        .join('<br>');
      element.appendChild(formulaDiv);
    }

    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'node-summary';
    summaryDiv.style.display = 'none';
    summaryDiv.textContent = node.summary;
    element.appendChild(summaryDiv);

    // Allow clicks through the pointer-events:none CSS2D overlay
    element.style.pointerEvents = 'auto';

    const css2dObject = new CSS2DObject(element);
    css2dObject.position.set(0, 0, 0);
    fillMesh.add(css2dObject);

    nodeMap.set(node.id, { mesh: fillMesh, css2dObject, element });
  }

  return nodeMap;
}
