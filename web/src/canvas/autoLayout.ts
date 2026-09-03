/**
 * Cycle-safe longest-path auto-layout for Blueprint graphs.
 */

import type { GraphDefinition } from '../api/contracts';

export interface AutoLayoutOptions {
  layerSpacing?: number;
  nodeSpacing?: number;
  startX?: number;
  startY?: number;
}

export function computeAutoLayout(
  graph: GraphDefinition,
  options?: AutoLayoutOptions
): Record<string, { x: number; y: number }> {
  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return {};
  }

  const layerSpacing = options?.layerSpacing ?? 280;
  const nodeSpacing = options?.nodeSpacing ?? 140;
  const startX = options?.startX ?? 80;
  const startY = options?.startY ?? 100;

  const nodeMap = new Map<string, (typeof graph.nodes)[0]>();
  const inDegree: Record<string, number> = {};
  const adj: Record<string, string[]> = {};
  const layers: Record<string, number> = {};

  for (const node of graph.nodes) {
    nodeMap.set(node.id, node);
    inDegree[node.id] = 0;
    adj[node.id] = [];
    layers[node.id] = 0;
  }

  // Populate graph edges
  for (const edge of graph.edges || []) {
    const src = edge.source?.node_id;
    const tgt = edge.target?.node_id;
    if (src && tgt && nodeMap.has(src) && nodeMap.has(tgt) && src !== tgt) {
      adj[src].push(tgt);
      inDegree[tgt] = (inDegree[tgt] || 0) + 1;
    }
  }

  // Queue initial roots (in-degree 0)
  const queue: string[] = [];
  const queued = new Set<string>();

  for (const node of graph.nodes) {
    if (inDegree[node.id] === 0) {
      queue.push(node.id);
      queued.add(node.id);
    }
  }

  // Longest-path relaxation: visit set ensures each node is queued at most once,
  // while layers can still be relaxed to max(existing, parent + 1)
  while (queue.length > 0) {
    const u = queue.shift()!;
    const currentLayer = layers[u] || 0;

    for (const v of adj[u] || []) {
      layers[v] = Math.max(layers[v] || 0, currentLayer + 1);
      if (!queued.has(v)) {
        queued.add(v);
        queue.push(v);
      }
    }
  }

  // Group nodes by computed layer
  const layerBuckets: Record<number, string[]> = {};
  for (const node of graph.nodes) {
    const l = layers[node.id] || 0;
    if (!layerBuckets[l]) {
      layerBuckets[l] = [];
    }
    layerBuckets[l].push(node.id);
  }

  const sortedLayers = Object.keys(layerBuckets)
    .map(Number)
    .sort((a, b) => a - b);

  const positions: Record<string, { x: number; y: number }> = {};
  for (const l of sortedLayers) {
    const nodeIds = layerBuckets[l];
    nodeIds.forEach((id, idx) => {
      positions[id] = {
        x: startX + l * layerSpacing,
        y: startY + idx * nodeSpacing,
      };
    });
  }

  return positions;
}
