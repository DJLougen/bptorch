/**
 * Canonical Project & Architecture IR Zustand Store.
 */

import { create } from 'zustand';
import type {
  Edge,
  GraphDefinition,
  NodeInstance,
  NodePosition,
  Project,
  Viewport,
} from '../api/contracts';

export const PROJECT_STORAGE_KEY = 'neural-blueprint-studio.project.v1';

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
    throw new Error('Project JSON does not match the Neural Blueprint project shape.');
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
  setNodeBreakpoint: (nodeId: string, enabled: boolean) => void;
  createEditableModuleCopy: (nodeId: string) => void;

  undo: () => void;
  redo: () => void;
}

// Initial fallback project state
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

  setNodeBreakpoint: (nodeId: string, enabled: boolean) => {
    const { project, openGraphId } = get();
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

    set({
      project: {
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
      },
    });
  },

  createEditableModuleCopy: (nodeId: string) => {
    const { project, openGraphId, undoStack } = get();
    const currentGraph = project.model.graphs[openGraphId];
    if (!currentGraph) return;

    const node = currentGraph.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Create custom definition
    const customType = `custom.${node.id}_${Date.now()}`;
    const customGraphId = `graph_custom_${node.id}`;

    // Find original subgraph
    const origSubgraph = Object.values(project.model.graphs).find(
      (g) => g.id.includes(node.id) || g.name.toLowerCase().includes(node.display_name.toLowerCase())
    );

    if (!origSubgraph) return;

    const customGraph: GraphDefinition = {
      ...origSubgraph,
      id: customGraphId,
      name: `Custom ${node.display_name}`,
    };

    const updatedNodes = currentGraph.nodes.map((n) => {
      if (n.id === nodeId) {
        return {
          ...n,
          definition_id: customType,
          display_name: `${n.display_name}*`,
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
          [customGraphId]: customGraph,
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
