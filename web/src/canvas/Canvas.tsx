/**
 * Main React Flow Canvas component with Blueprint projection and palette drop handler.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  SelectionMode,
  useReactFlow,
} from '@xyflow/react';
import type {
  Connection,
  Edge as FlowEdge,
  Node as FlowNode,
  IsValidConnection,
} from '@xyflow/react';
import type { Edge, NodeDefinitionSummary, NodeInstance, NodePosition } from '../api/contracts';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import { useTraceStore } from '../stores/traceStore';
import { useValidationStore } from '../stores/validationStore';
import { BlueprintEdge } from './BlueprintEdge';
import { BlueprintNode, BlueprintNodeData } from './BlueprintNode';
import { CanvasContextMenu } from './CanvasContextMenu';
import { computeAutoLayout } from './autoLayout';
import { COMPOSITE_TYPE_MAP } from '../components/BreadcrumbBar';
import { ConnectionAuthoringContext } from './ConnectionAuthoringContext';
import { validateConnection } from './connectionValidation';
const nodeTypes = {
  blueprintNode: BlueprintNode,
};

const edgeTypes = {
  blueprintEdge: BlueprintEdge,
};

export { validateConnection } from './connectionValidation';

const EMPTY_POSITIONS: Record<string, NodePosition> = {};
const EMPTY_COLLAPSED_NODE_IDS: string[] = [];
const MULTI_SELECTION_KEY_CODE = ['Meta', 'Control'];

export const Canvas: React.FC<{ catalog: NodeDefinitionSummary[] }> = ({ catalog }) => {
  const {
    project,
    openGraphId,
    addNode,
    removeNode,
    connectEdge,
    removeEdge,
    moveNodes,
    extractSubgraph,
    createEditableModuleCopy,
    setNodeBreakpoint,
    updateNodeMetadata,
    openGraph,
    toggleNodeCollapsed,
    setEdgeWaypoints,
    alignSelected,
  } = useProjectStore();
  const { selectNode, selectNodes, selectEdge, selectedNodeId, selectedNodeIds } = useUIStore();
  const { resolvedShapes } = useValidationStore();
  const addLog = useTraceStore((state) => state.addLog);
  const reactFlowInstance = useReactFlow();
  const lastPointerPos = useRef<{ x: number; y: number } | null>(null);
  const quickAddInputRef = useRef<HTMLInputElement>(null);
  const clipboardRef = useRef<{
    nodes: NodeInstance[];
    edges: Edge[];
    positions: Record<string, NodePosition>;
  } | null>(null);
  const [quickAdd, setQuickAdd] = useState<{
    open: boolean;
    query: string;
    flowPos: { x: number; y: number };
  }>({
    open: false,
    query: '',
    flowPos: { x: 0, y: 0 },
  });
  const [contextMenu, setContextMenu] = useState<{
    kind: 'pane' | 'node' | 'edge';
    x: number;
    y: number;
    nodeId?: string;
    edgeId?: string;
  } | null>(null);

  const currentGraph = project.model.graphs[openGraphId];
  const rawPositions = project.ui.node_positions[openGraphId];
  const positions = rawPositions || EMPTY_POSITIONS;
  const rawCollapsed = project.ui.collapsed_node_ids?.[openGraphId];
  const collapsedIds = rawCollapsed || EMPTY_COLLAPSED_NODE_IDS;
  // Build React Flow nodes from canonical IR
  const nodes: FlowNode[] = useMemo(() => {
    if (!currentGraph) return [];

    return currentGraph.nodes.map((node, idx) => {
      const defn = catalog.find((c) => c.type_id === node.definition_id);
      const pos = positions[node.id] || { x: 100 + idx * 250, y: 150 };

      const customGraph = node.definition_id.startsWith('custom.')
        ? project.model.graphs[node.definition_id.slice('custom.'.length)]
        : undefined;

      const nodeData: BlueprintNodeData = {
        id: node.id,
        definitionId: node.definition_id,
        displayName: node.display_name,
        category: defn?.category || (customGraph ? 'Composite Modules' : 'Layers'),
        icon: defn?.icon || null,
        isComposite: defn?.is_composite || Boolean(customGraph) || Boolean(node.definition_id.startsWith('custom.')),
        properties: node.properties,
        inputs: defn?.default_inputs?.length ? defn.default_inputs : (customGraph?.interface?.inputs || []),
        outputs: defn?.default_outputs?.length ? defn.default_outputs : (customGraph?.interface?.outputs || []),
        breakpoint: node.metadata.breakpoint,
        disabled: node.metadata.disabled,
        collapsed: collapsedIds.includes(node.id),
      };

      return {
        id: node.id,
        type: 'blueprintNode',
        position: pos,
        selected: Boolean(selectedNodeIds?.includes(node.id)) || selectedNodeId === node.id,
        data: nodeData as unknown as Record<string, unknown>,
      };
    });
  }, [currentGraph, catalog, rawPositions, selectedNodeId, selectedNodeIds, project.model.graphs, rawCollapsed]);

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
    (_: unknown, node: FlowNode, draggedNodes?: FlowNode[]) => {
      const moved: Record<string, { x: number; y: number }> = {};
      if (draggedNodes && draggedNodes.length > 1) {
        draggedNodes.forEach((n) => {
          moved[n.id] = { x: n.position.x, y: n.position.y };
        });
      } else {
        moved[node.id] = { x: node.position.x, y: node.position.y };
      }
      moveNodes(moved);
    },
    [moveNodes]
  );

  // Handle drag over from palette
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop from palette
  // Handle drop from palette with splice-on-wire support
  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const typeId = event.dataTransfer.getData('application/bptorch-node');
      if (!typeId) return;

      const defn = catalog.find((c) => c.type_id === typeId);
      if (!defn) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNodeId = `node_${defn.type_id.split('.')[1]?.split('@')[0] || 'node'}_${Date.now()}`;
      const newNode: NodeInstance = {
        id: newNodeId,
        definition_id: defn.type_id,
        display_name: defn.display_name,
        properties: {},
        metadata: { breakpoint: false, disabled: false },
      };
      const store = useProjectStore.getState();
      const activeProj = store.project;
      const activeGraph = activeProj.model.graphs[store.openGraphId];
      const activePositions = activeProj.ui.node_positions[store.openGraphId] || {};

      // Check if dropped near an existing wire for auto-splicing
      let splicedEdge: Edge | null = null;
      if (activeGraph?.edges) {
        for (const edge of activeGraph.edges) {
          const srcPos = activePositions[edge.source.node_id];
          const tgtPos = activePositions[edge.target.node_id];
          if (srcPos && tgtPos) {
            const minX = Math.min(srcPos.x, tgtPos.x) - 40;
            const maxX = Math.max(srcPos.x, tgtPos.x) + 40;
            const minY = Math.min(srcPos.y, tgtPos.y) - 60;
            const maxY = Math.max(srcPos.y, tgtPos.y) + 60;
            if (position.x >= minX && position.x <= maxX && position.y >= minY && position.y <= maxY) {
              const dx = tgtPos.x - srcPos.x;
              const dy = tgtPos.y - srcPos.y;
              const lenSq = dx * dx + dy * dy;
              if (lenSq > 0) {
                const u = Math.max(0, Math.min(1, ((position.x - srcPos.x) * dx + (position.y - srcPos.y) * dy) / lenSq));
                const px = srcPos.x + u * dx;
                const py = srcPos.y + u * dy;
                if (Math.hypot(position.x - px, position.y - py) < 70) {
                  splicedEdge = edge;
                  break;
                }
              }
            }
          }
        }
      }

      addNode(newNode, position);

      if (splicedEdge && defn.default_inputs.length > 0 && defn.default_outputs.length > 0) {
        const inPort = defn.default_inputs[0].id;
        const outPort = defn.default_outputs[0].id;
        removeEdge(splicedEdge.id);
        connectEdge({
          id: `e_${splicedEdge.source.node_id}_${newNodeId}_${Date.now()}`,
          source: splicedEdge.source,
          target: { node_id: newNodeId, port_id: inPort },
        });
        connectEdge({
          id: `e_${newNodeId}_${splicedEdge.target.node_id}_${Date.now() + 1}`,
          source: { node_id: newNodeId, port_id: outPort },
          target: splicedEdge.target,
        });
        addLog('info', `Spliced '${defn.display_name}' into wire between '${splicedEdge.source.node_id}' and '${splicedEdge.target.node_id}'.`);
      }
    },
    [catalog, reactFlowInstance, addNode, connectEdge, removeEdge, currentGraph, positions, addLog]
  );
  const filteredCatalog = useMemo(() => {
    if (!quickAdd.open) return [];
    const q = quickAdd.query.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (c) =>
        c.display_name.toLowerCase().includes(q) ||
        c.type_id.toLowerCase().includes(q)
    );
  }, [quickAdd.open, quickAdd.query, catalog]);

  const handleQuickAddSelect = useCallback(
    (defn: NodeDefinitionSummary) => {
      const stem = defn.type_id.split('.')[1]?.split('@')[0] || 'node';
      const newNodeId = `node_${stem}_${Date.now()}`;
      const newNode: NodeInstance = {
        id: newNodeId,
        definition_id: defn.type_id,
        display_name: defn.display_name,
        properties: {},
        metadata: { breakpoint: false, disabled: false },
      };
      addNode(newNode, quickAdd.flowPos);
      setQuickAdd((prev) => ({ ...prev, open: false, query: '' }));
    },
    [addNode, quickAdd.flowPos]
  );

  const handleCopy = useCallback(() => {
    const store = useProjectStore.getState();
    const activeProj = store.project;
    const activeGraph = activeProj.model.graphs[store.openGraphId];
    const activePos = activeProj.ui.node_positions[store.openGraphId] || {};
    const uiState = useUIStore.getState();
    const selIds =
      uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0
        ? uiState.selectedNodeIds
        : uiState.selectedNodeId
        ? [uiState.selectedNodeId]
        : [];

    if (selIds.length > 0 && activeGraph) {
      const selSet = new Set(selIds);
      const copiedNodes = activeGraph.nodes.filter((n) => selSet.has(n.id));
      const copiedEdges = activeGraph.edges.filter(
        (e) => selSet.has(e.source.node_id) && selSet.has(e.target.node_id)
      );
      const copiedPositions: Record<string, NodePosition> = {};
      selIds.forEach((id) => {
        if (activePos[id]) copiedPositions[id] = activePos[id];
      });
      clipboardRef.current = {
        nodes: copiedNodes,
        edges: copiedEdges,
        positions: copiedPositions,
      };
      addLog('info', `Copied ${copiedNodes.length} node(s) to clipboard.`);
    }
  }, [addLog]);

  const handlePaste = useCallback(() => {
    if (clipboardRef.current && clipboardRef.current.nodes.length > 0) {
      const { nodes: clipNodes, edges: clipEdges, positions: clipPositions } =
        clipboardRef.current;
      const ts = Date.now();
      const idMap: Record<string, string> = {};
      const newNodes: NodeInstance[] = clipNodes.map((orig, idx) => {
        const stem = orig.definition_id.split('.')[1]?.split('@')[0] || 'node';
        const newId = `node_${stem}_${ts}_${idx}`;
        idMap[orig.id] = newId;
        return {
          ...orig,
          id: newId,
          properties: { ...orig.properties },
          metadata: { ...orig.metadata },
        };
      });

      const newPositions: Record<string, NodePosition> = {};
      clipNodes.forEach((orig) => {
        const oldPos = clipPositions[orig.id] || { x: 100, y: 100 };
        newPositions[idMap[orig.id]] = { x: oldPos.x + 40, y: oldPos.y + 40 };
      });

      const newEdges: Edge[] = clipEdges.map((orig, idx) => ({
        id: `e_${idMap[orig.source.node_id]}_${idMap[orig.target.node_id]}_${ts}_${idx}`,
        source: { node_id: idMap[orig.source.node_id], port_id: orig.source.port_id },
        target: { node_id: idMap[orig.target.node_id], port_id: orig.target.port_id },
      }));

      useProjectStore.getState().batchAddNodesAndEdges(newNodes, newEdges, newPositions);
      useUIStore.getState().selectNodes(newNodes.map((n) => n.id));
      addLog('info', `Pasted ${newNodes.length} node(s).`);
    }
  }, [addLog]);

  const handleDuplicate = useCallback(() => {
    const store = useProjectStore.getState();
    const activeProj = store.project;
    const activeGraph = activeProj.model.graphs[store.openGraphId];
    const activePos = activeProj.ui.node_positions[store.openGraphId] || {};
    const uiState = useUIStore.getState();
    const selIds =
      uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0
        ? uiState.selectedNodeIds
        : uiState.selectedNodeId
        ? [uiState.selectedNodeId]
        : [];

    if (selIds.length > 0 && activeGraph) {
      const selSet = new Set(selIds);
      const dupNodes = activeGraph.nodes.filter((n) => selSet.has(n.id));
      const dupEdges = activeGraph.edges.filter(
        (e) => selSet.has(e.source.node_id) && selSet.has(e.target.node_id)
      );
      const ts = Date.now();
      const idMap: Record<string, string> = {};
      const newNodes: NodeInstance[] = dupNodes.map((orig, idx) => {
        const stem = orig.definition_id.split('.')[1]?.split('@')[0] || 'node';
        const newId = `node_${stem}_${ts}_${idx}`;
        idMap[orig.id] = newId;
        return {
          ...orig,
          id: newId,
          properties: { ...orig.properties },
          metadata: { ...orig.metadata },
        };
      });

      const newPositions: Record<string, NodePosition> = {};
      dupNodes.forEach((orig) => {
        const oldPos = activePos[orig.id] || { x: 100, y: 100 };
        newPositions[idMap[orig.id]] = { x: oldPos.x + 40, y: oldPos.y + 40 };
      });

      const newEdges: Edge[] = dupEdges.map((orig, idx) => ({
        id: `e_${idMap[orig.source.node_id]}_${idMap[orig.target.node_id]}_${ts}_${idx}`,
        source: { node_id: idMap[orig.source.node_id], port_id: orig.source.port_id },
        target: { node_id: idMap[orig.target.node_id], port_id: orig.target.port_id },
      }));

      useProjectStore.getState().batchAddNodesAndEdges(newNodes, newEdges, newPositions);
      useUIStore.getState().selectNodes(newNodes.map((n) => n.id));
      addLog('info', `Duplicated ${newNodes.length} node(s).`);
    }
  }, [addLog]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target?.isContentEditable
      ) {
        return;
      }

      const cmdOrCtrl = event.metaKey || event.ctrlKey;

      if (cmdOrCtrl && (event.key === 'z' || event.key === 'Z')) {
        event.preventDefault();
        if (event.shiftKey) {
          useProjectStore.getState().redo();
        } else {
          useProjectStore.getState().undo();
        }
        return;
      }
      if (event.ctrlKey && (event.key === 'y' || event.key === 'Y')) {
        event.preventDefault();
        useProjectStore.getState().redo();
        return;
      }

      if (event.key === 'Backspace' || event.key === 'Delete') {
        event.preventDefault();
        const uiState = useUIStore.getState();
        if (uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0) {
          uiState.selectedNodeIds.forEach((id) => removeNode(id));
          uiState.selectNodes([]);
        } else if (uiState.selectedNodeId) {
          removeNode(uiState.selectedNodeId);
          uiState.selectNode(null);
        } else if (uiState.selectedEdgeId) {
          removeEdge(uiState.selectedEdgeId);
          uiState.selectEdge(null);
        }
        return;
      }

      // Arrow keys nudge
      if (
        event.key === 'ArrowLeft' ||
        event.key === 'ArrowRight' ||
        event.key === 'ArrowUp' ||
        event.key === 'ArrowDown'
      ) {
        const uiState = useUIStore.getState();
        const selIds =
          uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0
            ? uiState.selectedNodeIds
            : uiState.selectedNodeId
            ? [uiState.selectedNodeId]
            : [];
        if (selIds.length > 0) {
          event.preventDefault();
          const step = event.shiftKey ? 40 : 10;
          let dx = 0;
          let dy = 0;
          if (event.key === 'ArrowLeft') dx = -step;
          if (event.key === 'ArrowRight') dx = step;
          if (event.key === 'ArrowUp') dy = -step;
          if (event.key === 'ArrowDown') dy = step;

          const store = useProjectStore.getState();
          const activePos = store.project.ui.node_positions[store.openGraphId] || {};
          const nextPositions: Record<string, NodePosition> = {};
          for (const id of selIds) {
            const cur = activePos[id] || { x: 0, y: 0 };
            nextPositions[id] = { x: cur.x + dx, y: cur.y + dy };
          }
          store.moveNodes(nextPositions);
          return;
        }
      }

      // Zoom to selection (Shift+2)
      if (event.shiftKey && (event.key === '2' || event.key === '@' || event.code === 'Digit2')) {
        const uiState = useUIStore.getState();
        const selIds =
          uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0
            ? uiState.selectedNodeIds
            : uiState.selectedNodeId
            ? [uiState.selectedNodeId]
            : [];
        if (selIds.length > 0) {
          event.preventDefault();
          reactFlowInstance.fitView({ nodes: selIds.map((id) => ({ id })), padding: 0.2 });
          return;
        }
      }

      if (cmdOrCtrl && (event.key === 'a' || event.key === 'A')) {
        event.preventDefault();
        const activeGraph =
          useProjectStore.getState().project.model.graphs[
            useProjectStore.getState().openGraphId
          ];
        if (activeGraph?.nodes) {
          useUIStore.getState().selectNodes(activeGraph.nodes.map((n) => n.id));
        }
        return;
      }

      // Copy (Ctrl/Cmd+C)
      if (cmdOrCtrl && (event.key === 'c' || event.key === 'C')) {
        event.preventDefault();
        handleCopy();
        return;
      }

      // Paste (Ctrl/Cmd+V)
      if (cmdOrCtrl && (event.key === 'v' || event.key === 'V')) {
        event.preventDefault();
        handlePaste();
        return;
      }

      // Duplicate (Ctrl/Cmd+D)
      if (cmdOrCtrl && (event.key === 'd' || event.key === 'D')) {
        event.preventDefault();
        handleDuplicate();
        return;
      }
      // Group into Module (Ctrl/Cmd+G)
      if (cmdOrCtrl && (event.key === 'g' || event.key === 'G')) {
        event.preventDefault();
        const uiState = useUIStore.getState();
        const selIds =
          uiState.selectedNodeIds && uiState.selectedNodeIds.length > 0
            ? uiState.selectedNodeIds
            : uiState.selectedNodeId
            ? [uiState.selectedNodeId]
            : [];

        if (selIds.length > 0) {
          const newGraphId = useProjectStore.getState().extractSubgraph(selIds);
          if (newGraphId) {
            useUIStore.getState().selectNodes([]);
            addLog('info', `Extracted ${selIds.length} node(s) into subgraph '${newGraphId}'.`);
          }
        }
        return;
      }


      if (
        event.key === ' ' &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        !event.shiftKey
      ) {
        event.preventDefault();
        const screenPos = lastPointerPos.current ?? {
          x: window.innerWidth / 2,
          y: window.innerHeight / 2,
        };
        const pos = reactFlowInstance.screenToFlowPosition(screenPos);
        setQuickAdd({ open: true, query: '', flowPos: pos });
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [reactFlowInstance, removeNode, removeEdge, handleCopy, handlePaste, handleDuplicate]);

  return (
    <div
      style={{ width: '100%', height: '100%', position: 'relative' }}
      onPointerMove={(e) => {
        lastPointerPos.current = { x: e.clientX, y: e.clientY };
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          selectNode(null);
          selectEdge(null);
        }
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
          onNodeClick={(_event, node) => {
            selectNode(node.id);
          }}
          onPaneClick={() => {
            selectNode(null);
            selectEdge(null);
          }}
          onSelectionChange={({ nodes: selNodes }) => {
            if (selNodes && selNodes.length > 1) {
              const newIds = selNodes.map((n) => n.id);
              const curIds = useUIStore.getState().selectedNodeIds || [];
              if (
                newIds.length !== curIds.length ||
                !newIds.every((id, idx) => id === curIds[idx])
              ) {
                selectNodes(newIds);
              }
            }
          }}
          selectionMode={SelectionMode.Partial}
          selectionKeyCode="Shift"
          multiSelectionKeyCode={MULTI_SELECTION_KEY_CODE}
          onNodesDelete={(deletedNodes) => {
            deletedNodes.forEach((n) => removeNode(n.id));
          }}
          onEdgesDelete={(deletedEdges) => {
            deletedEdges.forEach((e) => removeEdge(e.id));
          }}
          onPaneContextMenu={(event) => {
            event.preventDefault();
            setContextMenu({
              kind: 'pane',
              x: event.clientX,
              y: event.clientY,
            });
          }}
          onNodeContextMenu={(event, node) => {
            event.preventDefault();
            selectNode(node.id);
            setContextMenu({
              kind: 'node',
              x: event.clientX,
              y: event.clientY,
              nodeId: node.id,
            });
          }}
          onEdgeContextMenu={(event, edge) => {
            event.preventDefault();
            selectEdge(edge.id);
            setContextMenu({
              kind: 'edge',
              x: event.clientX,
              y: event.clientY,
              edgeId: edge.id,
            });
          }}
          connectionRadius={28}
          onEdgeDoubleClick={(event, edge) => {
            event.preventDefault();
            const flowPos = reactFlowInstance.screenToFlowPosition({
              x: event.clientX,
              y: event.clientY,
            });
            const currentPoints = project.ui.edge_waypoints?.[openGraphId]?.[edge.id] ?? [];
            setEdgeWaypoints(edge.id, [...currentPoints, flowPos]);
          }}
          fitView
          minZoom={0.2}
          maxZoom={2.5}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f2430" />
          <Controls style={{ background: '#181b24', border: '1px solid #272c3b' }} />
          <MiniMap
            style={{ background: '#12141c', border: '1px solid #272c3b', pointerEvents: 'none' }}
            nodeColor={() => '#38bdf8'}
            maskColor="rgba(0, 0, 0, 0.7)"
            pannable={false}
            zoomable={false}
          />
        </ReactFlow>
      </ConnectionAuthoringContext.Provider>
      {quickAdd.open && (
        <div
          style={{
            position: 'absolute',
            top: 60,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 320,
            maxHeight: 400,
            background: '#181b24',
            border: '1px solid #272c3b',
            borderRadius: 8,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 100,
            overflow: 'hidden',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ padding: 8, borderBottom: '1px solid #272c3b' }}>
            <input
              ref={quickAddInputRef}
              autoFocus
              type="text"
              placeholder="Quick add node... (Enter to add, Esc to close)"
              value={quickAdd.query}
              onChange={(e) =>
                setQuickAdd((prev) => ({ ...prev, query: e.target.value }))
              }
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  e.stopPropagation();
                  setQuickAdd((prev) => ({ ...prev, open: false }));
                } else if (e.key === 'Enter') {
                  e.stopPropagation();
                  if (filteredCatalog.length > 0) {
                    handleQuickAddSelect(filteredCatalog[0]);
                  }
                }
              }}
              style={{
                width: '100%',
                background: '#12141c',
                border: '1px solid #272c3b',
                borderRadius: 4,
                color: '#e2e8f0',
                padding: '6px 8px',
                fontSize: 12,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>
          <div
            style={{
              overflowY: 'auto',
              maxHeight: 320,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {filteredCatalog.length === 0 ? (
              <div
                style={{
                  padding: '12px 16px',
                  color: '#64748b',
                  fontSize: 12,
                }}
              >
                No matching nodes
              </div>
            ) : (
              filteredCatalog.map((item) => (
                <button
                  key={item.type_id}
                  type="button"
                  onClick={() => handleQuickAddSelect(item)}
                  style={{
                    padding: '8px 12px',
                    textAlign: 'left',
                    background: 'transparent',
                    border: 'none',
                    borderBottom: '1px solid #1f2430',
                    color: '#e2e8f0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 500 }}>
                    {item.display_name}
                  </span>
                  <span style={{ fontSize: 10, color: '#64748b' }}>
                    {item.type_id}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
      {contextMenu && (
        <CanvasContextMenu
          kind={contextMenu.kind}
          x={contextMenu.x}
          y={contextMenu.y}
          nodeId={contextMenu.nodeId}
          edgeId={contextMenu.edgeId}
          onClose={() => setContextMenu(null)}
          onPaste={handlePaste}
          canPaste={Boolean(clipboardRef.current && clipboardRef.current.nodes.length > 0)}
          onQuickAdd={() => {
            const pos = reactFlowInstance.screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });
            setQuickAdd({ open: true, query: '', flowPos: pos });
          }}
          onAutoLayout={() => {
            const g = project.model.graphs[openGraphId];
            if (g) {
              moveNodes(computeAutoLayout(g));
            }
          }}
          onCopy={handleCopy}
          onDuplicate={handleDuplicate}
          onDeleteNode={() => {
            if (contextMenu.nodeId) {
              removeNode(contextMenu.nodeId);
            }
          }}
          onGroupModule={() => {
            const ids =
              selectedNodeIds && selectedNodeIds.length > 0
                ? selectedNodeIds
                : contextMenu.nodeId
                ? [contextMenu.nodeId]
                : [];
            if (ids.length > 0) {
              const newG = extractSubgraph(ids);
              if (newG) {
                selectNodes([]);
                addLog('info', `Extracted ${ids.length} node(s) into subgraph '${newG}'.`);
              }
            }
          }}
          canCreateEditableCopy={(() => {
            if (!contextMenu.nodeId) return false;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            if (!node) return false;
            const defn = catalog.find((d) => d.type_id === node.definition_id);
            return Boolean(defn?.is_composite && !node.definition_id.startsWith('custom.'));
          })()}
          onCreateEditableCopy={() => {
            if (contextMenu.nodeId) {
              createEditableModuleCopy(contextMenu.nodeId);
            }
          }}
          isBreakpoint={(() => {
            if (!contextMenu.nodeId) return false;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            return Boolean(node?.metadata?.breakpoint);
          })()}
          onToggleBreakpoint={() => {
            if (!contextMenu.nodeId) return;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            if (node) {
              setNodeBreakpoint(node.id, !node.metadata?.breakpoint);
            }
          }}
          isDisabled={(() => {
            if (!contextMenu.nodeId) return false;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            return Boolean(node?.metadata?.disabled);
          })()}
          onToggleDisabled={() => {
            if (!contextMenu.nodeId) return;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            if (node) {
              updateNodeMetadata(node.id, { disabled: !node.metadata?.disabled });
            }
          }}
          canOpenSubgraph={(() => {
            if (!contextMenu.nodeId) return false;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            if (!node) return false;
            let targetId: string | undefined = undefined;
            if (node.definition_id.startsWith('custom.')) {
              targetId = node.definition_id.slice('custom.'.length);
            } else if (COMPOSITE_TYPE_MAP[node.definition_id]) {
              targetId = COMPOSITE_TYPE_MAP[node.definition_id];
            } else if (node.definition_id.includes('attention')) {
              targetId = 'graph_attention';
            } else if (node.definition_id.includes('mlp')) {
              targetId = 'graph_mlp';
            } else if (node.definition_id.includes('block')) {
              targetId = 'graph_block';
            }
            return Boolean(targetId && project.model.graphs[targetId]);
          })()}
          onOpenSubgraph={() => {
            if (!contextMenu.nodeId) return;
            const node = currentGraph?.nodes.find((n) => n.id === contextMenu.nodeId);
            if (!node) return;
            let targetId: string | undefined = undefined;
            if (node.definition_id.startsWith('custom.')) {
              targetId = node.definition_id.slice('custom.'.length);
            } else if (COMPOSITE_TYPE_MAP[node.definition_id]) {
              targetId = COMPOSITE_TYPE_MAP[node.definition_id];
            } else if (node.definition_id.includes('attention')) {
              targetId = 'graph_attention';
            } else if (node.definition_id.includes('mlp')) {
              targetId = 'graph_mlp';
            } else if (node.definition_id.includes('block')) {
              targetId = 'graph_block';
            }
            if (targetId && project.model.graphs[targetId]) {
              openGraph(targetId);
            }
          }}
          isCollapsed={(() => {
            if (!contextMenu.nodeId) return false;
            return (project.ui.collapsed_node_ids?.[openGraphId] ?? []).includes(contextMenu.nodeId);
          })()}
          onToggleCollapse={() => {
            if (contextMenu.nodeId) {
              toggleNodeCollapsed(contextMenu.nodeId);
            }
          }}
          hasMultiSelection={(selectedNodeIds?.length ?? 0) > 1}
          onAlignLeft={() => {
            if (selectedNodeIds && selectedNodeIds.length > 1) {
              alignSelected('left', selectedNodeIds);
            }
          }}
          onAlignTop={() => {
            if (selectedNodeIds && selectedNodeIds.length > 1) {
              alignSelected('top', selectedNodeIds);
            }
          }}
          onZoomToSelection={() => {
            const ids =
              selectedNodeIds && selectedNodeIds.length > 0
                ? selectedNodeIds
                : contextMenu.nodeId
                ? [contextMenu.nodeId]
                : [];
            if (ids.length > 0) {
              reactFlowInstance.fitView({ nodes: ids.map((id) => ({ id })), padding: 0.2 });
            }
          }}
          hasWaypoints={Boolean(
            contextMenu.edgeId &&
            project.ui.edge_waypoints?.[openGraphId]?.[contextMenu.edgeId]?.length
          )}
          onRemoveWaypoints={() => {
            if (contextMenu.edgeId) {
              setEdgeWaypoints(contextMenu.edgeId, []);
            }
          }}
          onDeleteEdge={() => {
            if (contextMenu.edgeId) {
              removeEdge(contextMenu.edgeId);
            }
          }}
        />
      )}
    </div>
  );
};
