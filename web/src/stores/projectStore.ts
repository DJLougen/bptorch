/**
 * Canonical Project & Architecture IR Zustand Store.
 */

import { create } from 'zustand';
import type {
  Edge,
  GraphDefinition,
  NodeInstance,
  NodeMetadata,
  NodePosition,
  PortDefinition,
  Project,
  TrainingConfig,
  Viewport,
} from '../api/contracts';
import { COMPOSITE_TYPE_MAP } from '../components/BreadcrumbBar';
import { useTraceStore } from './traceStore';

export const PROJECT_STORAGE_KEY = 'bptorch.project.v1';

type ProjectStorageReader = Pick<Storage, 'getItem'>;
type ProjectStorageWriter = Pick<Storage, 'setItem'>;

export type PersistedProjectReadResult =
  | { status: 'loaded'; project: Project }
  | { status: 'missing' }
  | { status: 'invalid'; error: string };

function isNodeInstance(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }

  const node = value as Partial<NodeInstance>;
  return (
    typeof node.id === 'string' &&
    typeof node.definition_id === 'string' &&
    typeof node.display_name === 'string' &&
    typeof node.properties === 'object' &&
    node.properties !== null &&
    !Array.isArray(node.properties) &&
    typeof node.metadata === 'object' &&
    node.metadata !== null &&
    !Array.isArray(node.metadata)
  );
}

function isPortReference(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }

  const port = value as { node_id?: unknown; port_id?: unknown };
  return typeof port.node_id === 'string' && typeof port.port_id === 'string';
}

function isEdge(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }

  const edge = value as Partial<Edge>;
  return (
    typeof edge.id === 'string' &&
    isPortReference(edge.source) &&
    isPortReference(edge.target)
  );
}

function isGraphDefinition(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }

  const graph = value as Partial<GraphDefinition>;
  const graphInterface = graph.interface;
  if (
    typeof graphInterface !== 'object' ||
    graphInterface === null ||
    Array.isArray(graphInterface)
  ) {
    return false;
  }

  return (
    typeof graph.id === 'string' &&
    typeof graph.name === 'string' &&
    typeof graph.kind === 'string' &&
    Array.isArray(graphInterface.inputs) &&
    Array.isArray(graphInterface.outputs) &&
    Array.isArray(graph.nodes) &&
    graph.nodes.every(isNodeInstance) &&
    Array.isArray(graph.edges) &&
    graph.edges.every(isEdge)
  );
}

/**
 * Small runtime guard for browser-restored and user-imported canonical projects.
 * Detailed semantic validation remains the backend validator's responsibility.
 */
export function isProject(value: unknown): value is Project {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }

  const candidate = value as Partial<Project>;
  const metadata = candidate.project;
  const model = candidate.model;
  const ui = candidate.ui;

  if (
    candidate.schema_version !== 1 ||
    typeof metadata !== 'object' ||
    metadata === null ||
    Array.isArray(metadata) ||
    typeof model !== 'object' ||
    model === null ||
    Array.isArray(model) ||
    typeof ui !== 'object' ||
    ui === null ||
    Array.isArray(ui)
  ) {
    return false;
  }

  if (
    typeof metadata.id !== 'string' ||
    typeof metadata.name !== 'string' ||
    typeof metadata.created_at !== 'string' ||
    typeof metadata.updated_at !== 'string' ||
    typeof model.root_graph_id !== 'string' ||
    typeof model.config !== 'object' ||
    model.config === null ||
    Array.isArray(model.config) ||
    typeof model.graphs !== 'object' ||
    model.graphs === null ||
    Array.isArray(model.graphs) ||
    !Array.isArray(model.weight_bindings) ||
    typeof ui.graph_viewports !== 'object' ||
    ui.graph_viewports === null ||
    Array.isArray(ui.graph_viewports) ||
    typeof ui.node_positions !== 'object' ||
    ui.node_positions === null ||
    Array.isArray(ui.node_positions) ||
    typeof ui.open_graph_id !== 'string'
  ) {
    return false;
  }

  const viewportsAreValid = Object.values(ui.graph_viewports).every(
    (viewport) =>
      typeof viewport === 'object' &&
      viewport !== null &&
      !Array.isArray(viewport) &&
      Number.isFinite(viewport.x) &&
      Number.isFinite(viewport.y) &&
      Number.isFinite(viewport.zoom)
  );
  const nodePositionsAreValid = Object.values(ui.node_positions).every(
    (graphPositions) =>
      typeof graphPositions === 'object' &&
      graphPositions !== null &&
      !Array.isArray(graphPositions) &&
      Object.values(graphPositions).every(
        (position) =>
          typeof position === 'object' &&
          position !== null &&
          !Array.isArray(position) &&
          Number.isFinite(position.x) &&
          Number.isFinite(position.y)
      )
  );

  if (!viewportsAreValid || !nodePositionsAreValid) {
    return false;
  }

  const graphs = model.graphs;
  return (
    isGraphDefinition(graphs[model.root_graph_id]) &&
    Object.values(graphs).every(isGraphDefinition)
  );
}

export function parseProjectJson(raw: string): Project {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error('Project file is not valid JSON.');
  }

  if (!isProject(value)) {
    throw new Error('Project JSON does not match the bpTorch project shape.');
  }
  return value;
}

export function serializeProject(project: Project): string {
  return JSON.stringify(project, null, 2);
}

function getBrowserStorage(): Storage | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readPersistedProject(
  storage: ProjectStorageReader | null = getBrowserStorage()
): PersistedProjectReadResult {
  if (!storage) {
    return { status: 'missing' };
  }

  let raw: string | null;
  try {
    raw = storage.getItem(PROJECT_STORAGE_KEY);
  } catch {
    return { status: 'invalid', error: 'Browser storage could not be read.' };
  }

  if (raw === null) {
    return { status: 'missing' };
  }

  try {
    return { status: 'loaded', project: parseProjectJson(raw) };
  } catch (error) {
    return {
      status: 'invalid',
      error: error instanceof Error ? error.message : 'Saved project data is invalid.',
    };
  }
}

export function persistProject(
  project: Project,
  storage: ProjectStorageWriter | null = getBrowserStorage()
): boolean {
  if (!storage) {
    return false;
  }

  try {
    storage.setItem(PROJECT_STORAGE_KEY, serializeProject(project));
    return true;
  } catch {
    return false;
  }
}

const restoredGraphId = (p: Project): string =>
  p.ui.open_graph_id && p.model.graphs[p.ui.open_graph_id]
    ? p.ui.open_graph_id
    : p.model.root_graph_id;

interface ProjectState {
  project: Project;
  isDirty: boolean;
  openGraphId: string;
  graphHistory: string[];
  historyIndex: number;
  undoStack: Project[];
  redoStack: Project[];

  // Actions
  loadProject: (project: Project) => void;
  openGraph: (graphId: string) => void;
  navigateBack: () => void;
  navigateForward: () => void;
  setGraphViewport: (graphId: string, viewport: Viewport) => void;

  addNode: (node: NodeInstance, position?: NodePosition) => void;
  removeNode: (nodeId: string) => void;
  connectEdge: (edge: Edge) => void;
  removeEdge: (edgeId: string) => void;
  moveNodes: (positions: Record<string, NodePosition>) => void;
  updateNodeProperty: (nodeId: string, key: string, value: unknown) => void;
  updateNodeDisplayName: (nodeId: string, value: string) => void;
  updateModelConfig: (key: string, value: unknown) => void;
  updateTrainingConfig: (key: string, value: unknown) => void;
  setNodeBreakpoint: (nodeId: string, enabled: boolean) => void;
  updateNodeMetadata: (nodeId: string, patch: Partial<NodeMetadata>) => void;
  createEditableModuleCopy: (nodeId: string) => void;
  markClean: () => void;
  batchAddNodesAndEdges: (
    nodes: NodeInstance[],
    edges: Edge[],
    positions: Record<string, NodePosition>
  ) => void;
  extractSubgraph: (nodeIds: string[], moduleName?: string) => string | null;
  toggleNodeCollapsed: (nodeId: string) => void;
  setEdgeWaypoints: (edgeId: string, points: NodePosition[]) => void;
  alignSelected: (axis: 'left' | 'top', ids: string[]) => void;
  undo: () => void;
  redo: () => void;
}

// Initial fallback project state
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  device: 'cpu',
  precision: 'fp32',
  ddp_enabled: false,
  seed: 1337,
  max_epochs: 10,
  max_steps: 100,
  learning_rate: 6e-4,
  weight_decay: 0.1,
  grad_accum_steps: 1,
  grad_clip: 1.0,
  batch_size: 8,
  checkpoint_interval: 50,
  eval_interval: 50,
};

export const createInitialProject = (): Project => ({
  schema_version: 1,
  project: {
    id: 'project_nanogpt',
    name: 'nanoGPT Experiment',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  model: {
    root_graph_id: 'graph_gpt',
    config: {
      block_size: 8,
      vocab_size: 32,
      n_layer: 2,
      n_head: 2,
      n_embd: 16,
      dropout: 0.0,
      bias: true,
      attention_implementation: 'sdpa',
    },
    graphs: {
      graph_gpt: {
        id: 'graph_gpt',
        name: 'nanoGPT',
        kind: 'root',
        interface: { inputs: [], outputs: [] },
        nodes: [],
        edges: [],
      },
    },
    weight_bindings: [],
  },
  ui: {
    graph_viewports: {},
    node_positions: {},
    open_graph_id: 'graph_gpt',
  },
});

export const useProjectStore = create<ProjectState>((set, get) => ({
  project: createInitialProject(),
  isDirty: false,
  openGraphId: 'graph_gpt',
  graphHistory: ['graph_gpt'],
  historyIndex: 0,
  undoStack: [],
  redoStack: [],

  loadProject: (project: Project) => {
    const requestedGraphId = project.ui.open_graph_id;
    const initialGraphId = project.model.graphs[requestedGraphId]
      ? requestedGraphId
      : project.model.root_graph_id;

    set({
      project,
      isDirty: false,
      openGraphId: initialGraphId,
      graphHistory: [initialGraphId],
      historyIndex: 0,
      undoStack: [],
      redoStack: [],
    });
  },

  openGraph: (graphId: string) => {
    const { graphHistory, historyIndex, project } = get();
    if (!project.model.graphs[graphId]) return;

    const newHistory = [...graphHistory.slice(0, historyIndex + 1), graphId];
    set({
      project: {
        ...project,
        ui: { ...project.ui, open_graph_id: graphId },
      },
      openGraphId: graphId,
      graphHistory: newHistory,
      historyIndex: newHistory.length - 1,
    });
  },

  navigateBack: () => {
    const { graphHistory, historyIndex, project } = get();
    if (historyIndex > 0) {
      const prevIdx = historyIndex - 1;
      const graphId = graphHistory[prevIdx];
      set({
        project: {
          ...project,
          ui: { ...project.ui, open_graph_id: graphId },
        },
        openGraphId: graphId,
        historyIndex: prevIdx,
      });
    }
  },

  navigateForward: () => {
    const { graphHistory, historyIndex, project } = get();
    if (historyIndex < graphHistory.length - 1) {
      const nextIdx = historyIndex + 1;
      const graphId = graphHistory[nextIdx];
      set({
        project: {
          ...project,
          ui: { ...project.ui, open_graph_id: graphId },
        },
        openGraphId: graphId,
        historyIndex: nextIdx,
      });
    }
  },

  setGraphViewport: (graphId: string, viewport: Viewport) => {
    const { project } = get();
    set({
      project: {
        ...project,
        ui: {
          ...project.ui,
          graph_viewports: {
            ...project.ui.graph_viewports,
            [graphId]: viewport,
          },
        },
      },
    });
  },

  addNode: (node: NodeInstance, position?: NodePosition) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedGraph: GraphDefinition = {
      ...currentGraph,
      nodes: [...currentGraph.nodes, node],
    };

    const updatedPositions = {
      ...(project.ui.node_positions[openGraphId] || {}),
    };
    if (position) {
      updatedPositions[node.id] = position;
    }

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: updatedGraph,
        },
      },
      ui: {
        ...project.ui,
        node_positions: {
          ...project.ui.node_positions,
          [openGraphId]: updatedPositions,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  removeNode: (nodeId: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedGraph: GraphDefinition = {
      ...currentGraph,
      nodes: currentGraph.nodes.filter((n) => n.id !== nodeId),
      edges: currentGraph.edges.filter(
        (e) => e.source.node_id !== nodeId && e.target.node_id !== nodeId
      ),
    };

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: updatedGraph,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  connectEdge: (edge: Edge) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedGraph: GraphDefinition = {
      ...currentGraph,
      edges: [...currentGraph.edges, edge],
    };

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: updatedGraph,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  removeEdge: (edgeId: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedGraph: GraphDefinition = {
      ...currentGraph,
      edges: currentGraph.edges.filter((e) => e.id !== edgeId),
    };

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: updatedGraph,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  moveNodes: (positions: Record<string, NodePosition>) => {
    const { project, openGraphId, undoStack } = get();
    const currentPositions = project.ui.node_positions[openGraphId] || {};

    const nextProject: Project = {
      ...project,
      ui: {
        ...project.ui,
        node_positions: {
          ...project.ui.node_positions,
          [openGraphId]: {
            ...currentPositions,
            ...positions,
          },
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  updateNodeProperty: (nodeId: string, key: string, value: unknown) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedNodes = currentGraph.nodes.map((node) => {
      if (node.id === nodeId) {
        return {
          ...node,
          properties: {
            ...node.properties,
            [key]: value,
          },
        };
      }
      return node;
    });

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: {
            ...currentGraph,
            nodes: updatedNodes,
          },
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  updateNodeDisplayName: (nodeId: string, value: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;
    const updatedNodes = currentGraph.nodes.map((node) =>
      node.id === nodeId ? { ...node, display_name: value } : node
    );
    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: { ...currentGraph, nodes: updatedNodes },
        },
      },
    };
    set({ project: nextProject, isDirty: true, undoStack: [...undoStack, project], redoStack: [] });
  },

  updateModelConfig: (key: string, value: unknown) => {
    const { project, undoStack } = get();
    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        config: {
          ...project.model.config,
          [key]: value,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },
  updateTrainingConfig: (key: string, value: unknown) => {
    const { project, undoStack } = get();
    const currentTraining: TrainingConfig = project.model.training || DEFAULT_TRAINING_CONFIG;
    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        training: {
          ...currentTraining,
          [key]: value,
        } as TrainingConfig,
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },


  setNodeBreakpoint: (nodeId: string, enabled: boolean) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedNodes = currentGraph.nodes.map((node) => {
      if (node.id === nodeId) {
        return {
          ...node,
          metadata: {
            ...node.metadata,
            breakpoint: enabled,
          },
        };
      }
      return node;
    });

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: {
            ...currentGraph,
            nodes: updatedNodes,
          },
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  updateNodeMetadata: (nodeId: string, patch: Partial<NodeMetadata>) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedNodes = currentGraph.nodes.map((node) => {
      if (node.id === nodeId) {
        return {
          ...node,
          metadata: {
            ...node.metadata,
            ...patch,
          },
        };
      }
      return node;
    });

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: {
            ...currentGraph,
            nodes: updatedNodes,
          },
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  createEditableModuleCopy: (nodeId: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const node = currentGraph.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    let sourceGraphId: string | undefined = COMPOSITE_TYPE_MAP[node.definition_id];
    if (!sourceGraphId || !project.model.graphs[sourceGraphId]) {
      if (node.definition_id.startsWith('custom.')) {
        const suffix = node.definition_id.slice('custom.'.length);
        if (project.model.graphs[suffix]) {
          sourceGraphId = suffix;
        }
      }
    }

    if (!sourceGraphId || !project.model.graphs[sourceGraphId]) {
      useTraceStore.getState().addLog('warn', `Cannot find source graph for ${node.definition_id}`);
      return;
    }

    const orig = project.model.graphs[sourceGraphId];
    const newGraphId = `graph_custom_${Date.now()}`;

    const customGraph: GraphDefinition = {
      ...orig,
      id: newGraphId,
      name: `Custom ${node.display_name}`,
      derived_from: node.definition_id,
      modified: true,
      nodes: orig.nodes.map((n) => ({
        ...n,
        properties: { ...n.properties },
        metadata: { ...n.metadata },
      })),
      edges: orig.edges.map((e) => ({
        ...e,
        source: { ...e.source },
        target: { ...e.target },
      })),
    };

    if (orig.kind === 'repeat') {
      customGraph.target_graph_id = orig.target_graph_id;
    }

    const nextNodePositions = { ...project.ui.node_positions };
    if (project.ui.node_positions[orig.id]) {
      nextNodePositions[newGraphId] = { ...project.ui.node_positions[orig.id] };
    }

    const updatedNodes = currentGraph.nodes.map((n) => {
      if (n.id === nodeId) {
        return {
          ...n,
          definition_id: `custom.${newGraphId}`,
        };
      }
      return n;
    });

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: {
            ...currentGraph,
            nodes: updatedNodes,
          },
          [newGraphId]: customGraph,
        },
      },
      ui: {
        ...project.ui,
        node_positions: nextNodePositions,
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  markClean: () => {
    const { project } = get();
    set({ isDirty: false });
    persistProject(project);
  },

  batchAddNodesAndEdges: (
    newNodes: NodeInstance[],
    newEdges: Edge[],
    newPositions: Record<string, NodePosition>
  ) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const updatedGraph: GraphDefinition = {
      ...currentGraph,
      nodes: [...currentGraph.nodes, ...newNodes],
      edges: [...currentGraph.edges, ...newEdges],
    };

    const updatedPositions = {
      ...(project.ui.node_positions[openGraphId] || {}),
      ...newPositions,
    };

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [openGraphId]: updatedGraph,
        },
      },
      ui: {
        ...project.ui,
        node_positions: {
          ...project.ui.node_positions,
          [openGraphId]: updatedPositions,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  extractSubgraph: (nodeIds: string[], moduleName?: string): string | null => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph || nodeIds.length === 0) return null;

    const selectedNodeSet = new Set(nodeIds);
    const selectedNodes = currentGraph.nodes.filter((n) => selectedNodeSet.has(n.id));
    if (selectedNodes.length === 0) return null;

    const ts = Date.now();
    const newGraphId = `graph_custom_${ts}`;
    const compDefId = `custom.${newGraphId}`;
    const compositeNodeId = `node_comp_${ts}`;
    const compositeDisplayName = moduleName || 'Custom Module';

    const internalEdges: Edge[] = [];
    const incomingEdges: Edge[] = [];
    const outgoingEdges: Edge[] = [];
    const unaffectedEdges: Edge[] = [];

    for (const edge of currentGraph.edges) {
      const srcSelected = selectedNodeSet.has(edge.source.node_id);
      const tgtSelected = selectedNodeSet.has(edge.target.node_id);

      if (srcSelected && tgtSelected) {
        internalEdges.push(edge);
      } else if (!srcSelected && tgtSelected) {
        incomingEdges.push(edge);
      } else if (srcSelected && !tgtSelected) {
        outgoingEdges.push(edge);
      } else {
        unaffectedEdges.push(edge);
      }
    }

    const moduleInputNodes: NodeInstance[] = [];
    const moduleInputPorts: PortDefinition[] = [];
    const newGraphIncomingEdges: Edge[] = [];
    const rewiredIncomingEdges: Edge[] = [];

    incomingEdges.forEach((edge, idx) => {
      const portName = `in_${edge.target.port_id}_${idx}`;
      const inputNodeId = `mod_in_${idx}`;
      moduleInputPorts.push({
        id: portName,
        display_name: portName,
        direction: 'input',
        required: true,
      });
      moduleInputNodes.push({
        id: inputNodeId,
        definition_id: 'builtin.module_input@1',
        display_name: `Input (${portName})`,
        properties: { name: portName },
        metadata: { breakpoint: false, disabled: false },
      });
      newGraphIncomingEdges.push({
        id: `e_mod_in_${idx}`,
        source: { node_id: inputNodeId, port_id: 'output' },
        target: { node_id: edge.target.node_id, port_id: edge.target.port_id },
      });
      rewiredIncomingEdges.push({
        id: edge.id,
        source: edge.source,
        target: { node_id: compositeNodeId, port_id: portName },
      });
    });

    const moduleOutputNodes: NodeInstance[] = [];
    const moduleOutputPorts: PortDefinition[] = [];
    const newGraphOutgoingEdges: Edge[] = [];
    const rewiredOutgoingEdges: Edge[] = [];

    outgoingEdges.forEach((edge, idx) => {
      const portName = `out_${edge.source.port_id}_${idx}`;
      const outputNodeId = `mod_out_${idx}`;
      moduleOutputPorts.push({
        id: portName,
        display_name: portName,
        direction: 'output',
      });
      moduleOutputNodes.push({
        id: outputNodeId,
        definition_id: 'builtin.module_output@1',
        display_name: `Output (${portName})`,
        properties: { name: portName },
        metadata: { breakpoint: false, disabled: false },
      });
      newGraphOutgoingEdges.push({
        id: `e_mod_out_${idx}`,
        source: { node_id: edge.source.node_id, port_id: edge.source.port_id },
        target: { node_id: outputNodeId, port_id: 'input' },
      });
      rewiredOutgoingEdges.push({
        id: edge.id,
        source: { node_id: compositeNodeId, port_id: portName },
        target: edge.target,
      });
    });

    const newGraph: GraphDefinition = {
      id: newGraphId,
      name: compositeDisplayName,
      kind: 'module',
      interface: {
        inputs: moduleInputPorts,
        outputs: moduleOutputPorts,
      },
      nodes: [...moduleInputNodes, ...selectedNodes, ...moduleOutputNodes],
      edges: [...newGraphIncomingEdges, ...internalEdges, ...newGraphOutgoingEdges],
    };

    const oldPositions = project.ui.node_positions[openGraphId] || {};
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    let minX = Infinity;
    let minY = Infinity;
    for (const nid of nodeIds) {
      const pos = oldPositions[nid] || { x: 100, y: 100 };
      sumX += pos.x;
      sumY += pos.y;
      minX = Math.min(minX, pos.x);
      minY = Math.min(minY, pos.y);
      count++;
    }
    const centroid = count > 0 ? { x: Math.round(sumX / count), y: Math.round(sumY / count) } : { x: 150, y: 150 };

    const newGraphPositions: Record<string, NodePosition> = {};
    moduleInputNodes.forEach((n, idx) => {
      newGraphPositions[n.id] = { x: 80, y: 100 + idx * 120 };
    });
    for (const n of selectedNodes) {
      const pos = oldPositions[n.id] || { x: 100, y: 100 };
      newGraphPositions[n.id] = {
        x: Math.max(260, (pos.x - (minX === Infinity ? 0 : minX)) + 260),
        y: Math.max(100, (pos.y - (minY === Infinity ? 0 : minY)) + 100),
      };
    }
    moduleOutputNodes.forEach((n, idx) => {
      newGraphPositions[n.id] = { x: 600, y: 100 + idx * 120 };
    });

    const compositeNode: NodeInstance = {
      id: compositeNodeId,
      definition_id: compDefId,
      display_name: compositeDisplayName,
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    };

    const remainingNodes = currentGraph.nodes.filter((n) => !selectedNodeSet.has(n.id));
    const updatedCurrentGraph: GraphDefinition = {
      ...currentGraph,
      nodes: [...remainingNodes, compositeNode],
      edges: [...unaffectedEdges, ...rewiredIncomingEdges, ...rewiredOutgoingEdges],
    };

    const updatedCurrentPositions = { ...oldPositions };
    for (const nid of nodeIds) {
      delete updatedCurrentPositions[nid];
    }
    updatedCurrentPositions[compositeNodeId] = centroid;

    const nextProject: Project = {
      ...project,
      model: {
        ...project.model,
        graphs: {
          ...project.model.graphs,
          [newGraphId]: newGraph,
          [openGraphId]: updatedCurrentGraph,
        },
      },
      ui: {
        ...project.ui,
        node_positions: {
          ...project.ui.node_positions,
          [newGraphId]: newGraphPositions,
          [openGraphId]: updatedCurrentPositions,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });

    return newGraphId;
  },

  toggleNodeCollapsed: (nodeId: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentList = project.ui.collapsed_node_ids?.[openGraphId] ?? [];
    const nextList = currentList.includes(nodeId)
      ? currentList.filter((id) => id !== nodeId)
      : [...currentList, nodeId];

    const nextProject: Project = {
      ...project,
      ui: {
        ...project.ui,
        collapsed_node_ids: {
          ...(project.ui.collapsed_node_ids || {}),
          [openGraphId]: nextList,
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  setEdgeWaypoints: (edgeId: string, points: NodePosition[]) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraphWaypoints = project.ui.edge_waypoints?.[openGraphId] ?? {};

    const nextProject: Project = {
      ...project,
      ui: {
        ...project.ui,
        edge_waypoints: {
          ...(project.ui.edge_waypoints || {}),
          [openGraphId]: {
            ...currentGraphWaypoints,
            [edgeId]: points,
          },
        },
      },
    };

    set({
      project: nextProject,
      isDirty: true,
      undoStack: [...undoStack, project],
      redoStack: [],
    });
  },

  alignSelected: (axis: 'left' | 'top', ids: string[]) => {
    const { project, openGraphId } = get();
    if (!ids || ids.length <= 1) return;
    const positions = project.ui.node_positions[openGraphId] || {};
    const relevantPositions = ids.map((id) => positions[id] || { x: 0, y: 0 });

    if (axis === 'left') {
      const minX = Math.min(...relevantPositions.map((p) => p.x));
      const nextPositions: Record<string, NodePosition> = {};
      for (const id of ids) {
        const cur = positions[id] || { x: 0, y: 0 };
        nextPositions[id] = { x: minX, y: cur.y };
      }
      get().moveNodes(nextPositions);
    } else {
      const minY = Math.min(...relevantPositions.map((p) => p.y));
      const nextPositions: Record<string, NodePosition> = {};
      for (const id of ids) {
        const cur = positions[id] || { x: 0, y: 0 };
        nextPositions[id] = { x: cur.x, y: minY };
      }
      get().moveNodes(nextPositions);
    }
  },

  undo: () => {
    const { project, undoStack, redoStack, openGraphId } = get();
    if (undoStack.length === 0) return;

    const previous = undoStack[undoStack.length - 1];
    const nextOpenGraphId = previous.model.graphs[openGraphId]
      ? openGraphId
      : restoredGraphId(previous);
    const restoredProject: Project = {
      ...previous,
      ui: { ...previous.ui, open_graph_id: nextOpenGraphId },
    };

    set({
      project: restoredProject,
      isDirty: true,
      openGraphId: nextOpenGraphId,
      graphHistory: [nextOpenGraphId],
      historyIndex: 0,
      undoStack: undoStack.slice(0, -1),
      redoStack: [project, ...redoStack],
    });
  },

  redo: () => {
    const { project, undoStack, redoStack } = get();
    if (redoStack.length === 0) return;

    const next = redoStack[0];
    set({
      project: next,
      isDirty: true,
      openGraphId: restoredGraphId(next),
      graphHistory: [restoredGraphId(next)],
      historyIndex: 0,
      undoStack: [...undoStack, project],
      redoStack: redoStack.slice(1),
    });
  },
}));

useProjectStore.subscribe((state, previousState) => {
  if (state.project !== previousState.project) {
    persistProject(state.project);
  }
});
