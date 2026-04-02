import * as THREE from 'three';
import { NodeEntry } from './nodes';
import { EdgeEntry, updateEdge } from './edges';
import { Status, STATUS_COLORS } from './types';

function injectInteractionStyles(): void {
  const styleId = 'interaction-styles';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    .node-highlighted {
      box-shadow: 0 0 12px rgba(41, 128, 185, 0.6);
    }
    .viz-legend {
      position: fixed;
      top: 10px;
      left: 10px;
      background: white;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 12px;
      font-family: sans-serif;
      z-index: 100;
      pointer-events: none;
    }
    .viz-legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }
    .viz-legend-item:last-child {
      margin-bottom: 0;
    }
    .viz-legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }
  `;
  document.head.appendChild(style);
}

function createLegend(): void {
  const legend = document.createElement('div');
  legend.className = 'viz-legend';

  const statusLabels: Record<Status, string> = {
    [Status.Support]: 'Support',
    [Status.Falsified]: 'Falsified',
    [Status.Inconclusive]: 'Inconclusive',
    [Status.InProgress]: 'In Progress',
    [Status.Secondary]: 'Secondary',
  };

  for (const status of Object.values(Status)) {
    const colors = STATUS_COLORS[status];
    const item = document.createElement('div');
    item.className = 'viz-legend-item';

    const dot = document.createElement('div');
    dot.className = 'viz-legend-dot';
    dot.style.background = colors.fill;
    dot.style.border = `2px solid ${colors.border}`;

    const label = document.createElement('span');
    label.textContent = statusLabels[status];
    label.style.color = '#333';

    item.appendChild(dot);
    item.appendChild(label);
    legend.appendChild(item);
  }

  document.body.appendChild(legend);
}

export function setupInteraction(
  nodeMap: Map<string, NodeEntry>,
  edgeMap: Map<string, EdgeEntry>,
  camera: THREE.OrthographicCamera,
  markDirty: () => void,
): void {
  injectInteractionStyles();
  createLegend();

  // --- Node click → expand/collapse ---
  for (const [_id, entry] of nodeMap) {
    entry.element.addEventListener('click', () => {
      const expanded = entry.element.dataset.expanded === 'true';
      entry.element.dataset.expanded = expanded ? 'false' : 'true';

      const summaryEl = entry.element.querySelector('.node-summary') as HTMLElement | null;
      const formulaEl = entry.element.querySelector('.node-formula') as HTMLElement | null;

      if (!expanded) {
        if (summaryEl) summaryEl.style.display = 'block';
        if (formulaEl) formulaEl.style.maxHeight = '200px';
      } else {
        if (summaryEl) summaryEl.style.display = 'none';
        if (formulaEl) formulaEl.style.maxHeight = '40px';
      }

      markDirty();
    });
  }

  // --- Node drag ---
  let dragging = false;
  let dragNodeId: string | null = null;
  let dragStartPointerX = 0;
  let dragStartPointerY = 0;
  let dragStartWorldX = 0;
  let dragStartWorldY = 0;

  for (const [nodeId, entry] of nodeMap) {
    entry.element.addEventListener('pointerdown', (event: PointerEvent) => {
      if (event.button !== 0) return;
      event.stopPropagation();

      dragging = true;
      dragNodeId = nodeId;
      dragStartPointerX = event.clientX;
      dragStartPointerY = event.clientY;
      dragStartWorldX = entry.mesh.position.x;
      dragStartWorldY = entry.mesh.position.y;
    });
  }

  window.addEventListener('pointermove', (event: PointerEvent) => {
    if (!dragging || dragNodeId === null) return;

    const worldUnitsPerPixelX = (camera.right - camera.left) / window.innerWidth;
    const worldUnitsPerPixelY = (camera.top - camera.bottom) / window.innerHeight;

    const deltaScreenX = event.clientX - dragStartPointerX;
    const deltaScreenY = event.clientY - dragStartPointerY;

    const entry = nodeMap.get(dragNodeId)!;
    entry.mesh.position.x = dragStartWorldX + deltaScreenX * worldUnitsPerPixelX;
    entry.mesh.position.y = dragStartWorldY - deltaScreenY * worldUnitsPerPixelY;

    markDirty();
  });

  window.addEventListener('pointerup', () => {
    if (!dragging || dragNodeId === null) return;

    const movedNodeId = dragNodeId;
    dragging = false;
    dragNodeId = null;

    // Update connected edges
    for (const [edgeId, edgeEntry] of edgeMap) {
      if (edgeEntry.source === movedNodeId || edgeEntry.target === movedNodeId) {
        updateEdge(edgeId, edgeMap, nodeMap);
      }
    }

    markDirty();
  });

  // --- Edge label hover → highlight ---
  for (const [_edgeId, edgeEntry] of edgeMap) {
    const labelElement = edgeEntry.labelElement;
    const lineMaterial = edgeEntry.line.material as THREE.LineBasicMaterial;
    const originalColor = lineMaterial.color.getHex();
    const originalOpacity = lineMaterial.opacity;
    const originalLabelOpacity = labelElement.style.opacity;

    const sourceNodeEntry = nodeMap.get(edgeEntry.source);
    const targetNodeEntry = nodeMap.get(edgeEntry.target);

    labelElement.addEventListener('mouseenter', () => {
      labelElement.style.opacity = '1';
      lineMaterial.color.set('#2980b9');
      lineMaterial.opacity = 1.0;

      if (sourceNodeEntry) sourceNodeEntry.element.classList.add('node-highlighted');
      if (targetNodeEntry) targetNodeEntry.element.classList.add('node-highlighted');

      markDirty();
    });

    labelElement.addEventListener('mouseleave', () => {
      labelElement.style.opacity = originalLabelOpacity || '';
      lineMaterial.color.setHex(originalColor);
      lineMaterial.opacity = originalOpacity;

      if (sourceNodeEntry) sourceNodeEntry.element.classList.remove('node-highlighted');
      if (targetNodeEntry) targetNodeEntry.element.classList.remove('node-highlighted');

      markDirty();
    });
  }
}
