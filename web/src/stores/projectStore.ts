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

[Showing lines 1-300 of 802. Use :301 to continue]