import { scene, camera, webglRenderer, css2dRenderer, markDirty } from './canvas';
import { createNodes } from './nodes';
import { createEdges } from './edges';
import { setupInteraction } from './interaction';
import { setupMinimap } from './minimap';
import { nodes, edges } from './graph-data';

// Create nodes and edges
const nodeMap = createNodes(scene, nodes);
const edgeMap = createEdges(scene, edges, nodeMap);

// Setup interaction
setupInteraction(nodeMap, edgeMap, camera, markDirty);

// Setup minimap
const { updateMinimap } = setupMinimap(scene, camera, markDirty);

// Render loop
function animate(): void {
  requestAnimationFrame(animate);
  webglRenderer.render(scene, camera);
  css2dRenderer.render(scene, camera);
  updateMinimap();
}

animate();
