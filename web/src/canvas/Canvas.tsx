/**
 * Main React Flow Canvas component with Blueprint projection and palette drop handler.
 */

import React, { useCallback, useMemo } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
} from '@xyflow/react';
import type {
  Connection,
  Edge as FlowEdge,
  Node as FlowNode,
  IsValidConnection,
} from '@xyflow/react';
import type { Edge, NodeDefinitionSummary, NodeInstance } from '../api/contracts';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import { useTraceStore } from '../stores/traceStore';
import { useValidationStore } from '../stores/validationStore';
import { BlueprintEdge } from './BlueprintEdge';
import { BlueprintNode, BlueprintNodeData } from './BlueprintNode';
import { ConnectionAuthoringContext } from './ConnectionAuthoringContext';
import { validateConnection } from './connectionValidation';

const nodeTypes = {
  blueprintNode: BlueprintNode,
};

const edgeTypes = {
  blueprintEdge: BlueprintEdge,
};

export { validateConnection } from './connectionValidation';

export const Canvas: React.FC<{ catalog: NodeDefinitionSummary[] }> = ({ catalog }) => {
  const { project, openGraphId, addNode, removeNode, connectEdge, removeEdge, moveNodes } =
    useProjectStore();
  const { selectNode, selectEdge } = useUIStore();
  const { resolvedShapes } = useValidationStore();
  const addLog = useTraceStore((state) => state.addLog);
  const reactFlowInstance = useReactFlow();

  const currentGraph = project.model.graphs[openGraphId];
  const positions = project.ui.node_positions[openGraphId] || {};

  // Build React Flow nodes from canonical IR
  const nodes: FlowNode[] = useMemo(() => {
    if (!currentGraph) return [];

    return currentGraph.nodes.map((node, idx) => {
      const defn = catalog.find((c) => c.type_id === node.definition_id);
      const pos = positions[node.id] || { x: 100 + idx * 250, y: 150 };

      const nodeData: BlueprintNodeData = {
        id: node.id,
        definitionId: node.definition_id,
        displayName: node.display_name,
        category: defn?.category || 'Layers',
        icon: defn?.icon || null,
        isComposite: defn?.is_composite || Boolean(node.definition_id.startsWith('custom.')),
        properties: node.properties,
        inputs: defn?.default_inputs || [],
        outputs: defn?.default_outputs || [],
        breakpoint: node.metadata.breakpoint,
      };

      return {
        id: node.id,
        type: 'blueprintNode',
        position: pos,
        data: nodeData as unknown as Record<string, unknown>,
      };
    });
  }, [currentGraph, catalog, positions]);

  // Build React Flow edges from canonical IR
  const edges: FlowEdge[] = useMemo(() => {
    if (!currentGraph) return [];

    return currentGraph.edges.map((edge) => {
      const sourceNode = currentGraph.nodes.find((node) => node.id === edge.source.node_id);
      const sourceDefinition = catalog.find(
        (definition) => definition.type_id === sourceNode?.definition_id
      );
      const sourcePort = sourceDefinition?.default_outputs.find(
        (port) => port.id === edge.source.port_id
      );
      const isExec =
        sourcePort?.kind === 'exec' ||
        edge.source.port_id.includes('exec') ||
        edge.source.port_id.includes('then_') ||
        edge.source.port_id === 'loop_body';

      const srcSpec = resolvedShapes[edge.source.node_id]?.[edge.source.port_id];
      let shapeLabel: string | undefined;
      if (srcSpec && srcSpec.shape) {
        const dims = srcSpec.shape.map((d) => {
          if (typeof d === 'object') {
            if ('name' in d) return d.name;
            if ('key' in d) return d.key;
            if ('value' in d) return String(d.value);
          }
          return '?';
        });
        shapeLabel = `[${dims.join(', ')}]`;
      }

      return {
        id: edge.id,
        source: edge.source.node_id,
        sourceHandle: edge.source.port_id,
        target: edge.target.node_id,
        targetHandle: edge.target.port_id,
        type: 'blueprintEdge',
        data: {
          shapeLabel,
          isExec,
          sourceNodeId: edge.source.node_id,
          sourcePortId: edge.source.port_id,
          targetNodeId: edge.target.node_id,
          targetPortId: edge.target.port_id,
        },
      };
    });
  }, [catalog, currentGraph, resolvedShapes]);

  // Handle edge connection
  const handleConnect = useCallback(
    (params: Connection) => {
      if (!currentGraph) {
        addLog('warn', 'Connection rejected: the active graph is unavailable.');
        return;
      }

      const validation = validateConnection(params, currentGraph, catalog, resolvedShapes);
      if (!validation.valid) {
        addLog('warn', `Connection rejected: ${validation.reason}.`);
        return;
      }

      const edgeIdBase = `e_${validation.sourceNodeId}_${validation.targetNodeId}_${Date.now()}`;
      let edgeId = edgeIdBase;
      let suffix = 2;
      while (currentGraph.edges.some((edge) => edge.id === edgeId)) {
        edgeId = `${edgeIdBase}_${suffix}`;
        suffix += 1;
      }

      const newEdge: Edge = {
        id: edgeId,
        source: {
          node_id: validation.sourceNodeId,
          port_id: validation.sourcePortId,
        },
        target: {
          node_id: validation.targetNodeId,
          port_id: validation.targetPortId,
        },
      };
      connectEdge(newEdge);
    },
    [addLog, catalog, connectEdge, currentGraph, resolvedShapes]
  );

  const connectionAuthoring = useMemo(
    () => ({
      validate: (connection: Connection) =>
        currentGraph
          ? validateConnection(connection, currentGraph, catalog, resolvedShapes)
          : { valid: false as const, reason: 'the active graph is unavailable' },
    }),
    [catalog, currentGraph, resolvedShapes]
  );

  const isValidConnection = useCallback<IsValidConnection<FlowEdge>>(
    (connection) => {
      const normalized: Connection = {
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null,
      };
      return connectionAuthoring.validate(normalized).valid;
    },
    [connectionAuthoring]
  );

  // Handle node drag stop
  const handleNodeDragStop = useCallback(
    (_: unknown, node: FlowNode) => {
      moveNodes({
        [node.id]: { x: node.position.x, y: node.position.y },
      });
    },
    [moveNodes]
  );

  // Handle drag over from palette
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop from palette
  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const typeId = event.dataTransfer.getData('application/neural-blueprint-node');
      if (!typeId) return;

      const defn = catalog.find((c) => c.type_id === typeId);
      if (!defn) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNodeId = `node_${defn.type_id.split('.')[1].split('@')[0]}_${Date.now()}`;
      const newNode: NodeInstance = {
        id: newNodeId,
        definition_id: defn.type_id,
        display_name: defn.display_name,
        properties: {},
        metadata: { breakpoint: false, disabled: false },
      };

      addNode(newNode, position);
    },
    [catalog, reactFlowInstance, addNode]
  );

  return (
    <div
      style={{ width: '100%', height: '100%', position: 'relative' }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => {
        selectNode(null);
        selectEdge(null);
      }}
    >
      <ConnectionAuthoringContext.Provider value={connectionAuthoring}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onConnect={handleConnect}
          isValidConnection={isValidConnection}
          onNodeDragStop={handleNodeDragStop}
          onNodesDelete={(deletedNodes) => {
            deletedNodes.forEach((n) => removeNode(n.id));
          }}
          onEdgesDelete={(deletedEdges) => {
            deletedEdges.forEach((e) => removeEdge(e.id));
          }}
          fitView
          minZoom={0.2}
          maxZoom={2.5}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f2430" />
          <Controls style={{ background: '#181b24', border: '1px solid #272c3b' }} />
          <MiniMap
            style={{ background: '#12141c', border: '1px solid #272c3b' }}
            nodeColor={() => '#38bdf8'}
            maskColor="rgba(0, 0, 0, 0.7)"
          />
        </ReactFlow>
      </ConnectionAuthoringContext.Provider>
    </div>
  );
};
