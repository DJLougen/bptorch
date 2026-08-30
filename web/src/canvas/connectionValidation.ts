/**
 * Client-side connection compatibility checks used while authoring edges.
 */

import type { Connection } from '@xyflow/react';
import type {
  GraphDefinition,
  NodeDefinitionSummary,
  PortDefinition,
  ShapeDim,
  TensorSpec,
} from '../api/contracts';

export type ConnectionValidationResult =
  | {
      valid: true;
      sourceNodeId: string;
      sourcePortId: string;
      targetNodeId: string;
      targetPortId: string;
    }
  | { valid: false; reason: string };

function shapeDimLabel(dim: ShapeDim): string {
  if (typeof dim !== 'object' || dim === null) return '?';
  if ('name' in dim) return dim.name;
  if ('key' in dim) return dim.key;
  if ('value' in dim) return String(dim.value);
  if ('kind' in dim && dim.kind === 'unknown') return '?';
  return '?';
}

function shapeDimKey(dim: ShapeDim): string | null {
  if (typeof dim !== 'object' || dim === null) return null;
  if ('name' in dim) return `sym:${dim.name}`;
  if ('key' in dim) return `cfg:${dim.key}`;
  if ('value' in dim) return `lit:${dim.value}`;
  if ('kind' in dim && dim.kind === 'unknown') return null;
  return null;
}

/** Compare source tensor shape against a target port's expected shape (when known). */
export function areShapesCompatible(
  sourceSpec: TensorSpec | undefined,
  targetPort: PortDefinition
): { compatible: boolean; reason?: string } {
  const sourceShape = sourceSpec?.shape;
  const targetShape = targetPort.default_shape;

  if (!sourceShape?.length || !targetShape?.length) {
    return { compatible: true };
  }

  if (sourceShape.length !== targetShape.length) {
    return {
      compatible: false,
      reason: `tensor ranks do not match (${sourceShape.length}D → ${targetShape.length}D)`,
    };
  }

  for (let i = 0; i < sourceShape.length; i += 1) {
    const sourceKey = shapeDimKey(sourceShape[i]);
    const targetKey = shapeDimKey(targetShape[i]);
    if (sourceKey && targetKey && sourceKey !== targetKey) {
      return {
        compatible: false,
        reason: `shape mismatch at dim ${i} (${shapeDimLabel(sourceShape[i])} → ${shapeDimLabel(targetShape[i])})`,
      };
    }
  }

  return { compatible: true };
}

export function validateConnection(
  connection: Connection,
  graph: GraphDefinition,
  catalog: NodeDefinitionSummary[],
  resolvedShapes: Record<string, Record<string, TensorSpec>> = {}
): ConnectionValidationResult {
  const sourceNodeId = connection.source;
  const targetNodeId = connection.target;
  const sourcePortId = connection.sourceHandle;
  const targetPortId = connection.targetHandle;

  if (!sourceNodeId || !targetNodeId || !sourcePortId || !targetPortId) {
    return { valid: false, reason: 'both endpoints must name a node and port' };
  }

  const sourceNode = graph.nodes.find((node) => node.id === sourceNodeId);
  const targetNode = graph.nodes.find((node) => node.id === targetNodeId);
  if (!sourceNode) {
    return { valid: false, reason: `source node "${sourceNodeId}" does not exist` };
  }
  if (!targetNode) {
    return { valid: false, reason: `target node "${targetNodeId}" does not exist` };
  }
  if (sourceNodeId === targetNodeId) {
    return { valid: false, reason: 'a node cannot connect to itself' };
  }

  const sourceDefinition = catalog.find(
    (definition) => definition.type_id === sourceNode.definition_id
  );
  const targetDefinition = catalog.find(
    (definition) => definition.type_id === targetNode.definition_id
  );
  if (!sourceDefinition) {
    return {
      valid: false,
      reason: `source node type "${sourceNode.definition_id}" is not in the node catalog`,
    };
  }
  if (!targetDefinition) {
    return {
      valid: false,
      reason: `target node type "${targetNode.definition_id}" is not in the node catalog`,
    };
  }

  const sourcePort = sourceDefinition.default_outputs.find((port) => port.id === sourcePortId);
  if (!sourcePort || sourcePort.direction !== 'output') {
    return {
      valid: false,
      reason: `"${sourceNode.display_name}.${sourcePortId}" is not a catalog output`,
    };
  }

  const targetPort = targetDefinition.default_inputs.find((port) => port.id === targetPortId);
  if (!targetPort || targetPort.direction !== 'input') {
    return {
      valid: false,
      reason: `"${targetNode.display_name}.${targetPortId}" is not a catalog input`,
    };
  }

  const sourceKind = sourcePort.kind ?? 'data';
  const targetKind = targetPort.kind ?? 'data';
  if (sourceKind !== targetKind) {
    return {
      valid: false,
      reason: `port kinds do not match (${sourceKind} → ${targetKind})`,
    };
  }

  const sourceDtype = sourcePort.tensor_type?.dtype_family;
  const targetDtype = targetPort.tensor_type?.dtype_family;
  if (
    sourceKind === 'data' &&
    sourceDtype &&
    targetDtype &&
    sourceDtype !== 'any' &&
    targetDtype !== 'any' &&
    sourceDtype !== targetDtype
  ) {
    return {
      valid: false,
      reason: `tensor dtype families do not match (${sourceDtype} → ${targetDtype})`,
    };
  }

  if (sourceKind === 'data') {
    const shapeCheck = areShapesCompatible(
      resolvedShapes[sourceNodeId]?.[sourcePortId],
      targetPort
    );
    if (!shapeCheck.compatible) {
      return {
        valid: false,
        reason: shapeCheck.reason ?? 'tensor shapes are incompatible',
      };
    }
  }

  const targetIsSingle = (targetPort.multiplicity ?? 'single') === 'single';
  const targetAlreadyConnected = graph.edges.some(
    (edge) =>
      edge.target.node_id === targetNodeId && edge.target.port_id === targetPortId
  );
  if (targetIsSingle && targetAlreadyConnected) {
    return {
      valid: false,
      reason: `"${targetNode.display_name}.${targetPortId}" accepts only one connection`,
    };
  }

  return {
    valid: true,
    sourceNodeId,
    sourcePortId,
    targetNodeId,
    targetPortId,
  };
}
