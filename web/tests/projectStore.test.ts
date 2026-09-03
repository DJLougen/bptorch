import { beforeEach, describe, expect, it } from 'vitest';
import {
  PROJECT_STORAGE_KEY,
  createInitialProject,
  parseProjectJson,
  readPersistedProject,
  serializeProject,
  useProjectStore,
} from '../src/stores/projectStore';

describe('project browser persistence', () => {
  beforeEach(() => {
    useProjectStore.getState().loadProject(createInitialProject());
    localStorage.clear();
  });

  it('persists a project mutation and restores the edited value after a reload boundary', () => {
    useProjectStore.getState().updateModelConfig('n_embd', 96);

    const savedJson = localStorage.getItem(PROJECT_STORAGE_KEY);
    expect(savedJson).not.toBeNull();

    useProjectStore.getState().loadProject(createInitialProject());
    localStorage.setItem(PROJECT_STORAGE_KEY, savedJson as string);

    const restored = readPersistedProject();
    if (restored.status !== 'loaded') {
      throw new Error(`Expected a saved project, received ${restored.status}`);
    }

    useProjectStore.getState().loadProject(restored.project);
    expect(useProjectStore.getState().project.model.config.n_embd).toBe(96);
  });

  it('round-trips exported project JSON through the import shape guard', () => {
    const project = createInitialProject();
    project.project.id = 'round_trip_project';
    project.project.name = 'Round Trip Project';
    project.model.config.n_embd = 48;

    const imported = parseProjectJson(serializeProject(project));

    expect(imported).toEqual(project);
    expect(imported.project.name).toBe('Round Trip Project');
  });

  it('reports malformed saved state without replacing the current project', () => {
    const currentProject = useProjectStore.getState().project;
    localStorage.setItem(PROJECT_STORAGE_KEY, '{not-json');

    const restored = readPersistedProject();

    expect(restored.status).toBe('invalid');
    expect(useProjectStore.getState().project).toBe(currentProject);
    expect(() => parseProjectJson(JSON.stringify({ schema_version: 1 }))).toThrow(
      /project shape/i
    );
  });

  it('undo restores openGraphId after cross-graph navigation', () => {
    const project = createInitialProject();
    const graphMlp = project.model.graphs.graph_gpt;
    project.model.graphs.graph_block = {
      ...graphMlp,
      id: 'graph_block',
      name: 'Block Graph',
    };
    project.ui.open_graph_id = 'graph_gpt';

    useProjectStore.getState().loadProject(project);
    useProjectStore.getState().openGraph('graph_block');
    useProjectStore.getState().updateModelConfig('n_embd', 99);
    useProjectStore.getState().navigateBack();
    useProjectStore.getState().undo();

    expect(useProjectStore.getState().openGraphId).toBe(project.ui.open_graph_id);
  });

  it('creates an editable module copy with custom graph and lineage', () => {
    const project = createInitialProject();
    project.model.graphs.graph_mlp = {
      id: 'graph_mlp',
      name: 'MLP',
      kind: 'module',
      interface: { inputs: [], outputs: [] },
      nodes: [],
      edges: [],
    };
    const rootId = project.model.root_graph_id;
    project.model.graphs[rootId].nodes.push({
      id: 'node_mlp_1',
      definition_id: 'builtin.nanogpt_mlp@1',
      display_name: 'MLP Block',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });
    project.ui.open_graph_id = rootId;

    useProjectStore.getState().loadProject(project);
    useProjectStore.getState().createEditableModuleCopy('node_mlp_1');

    const updated = useProjectStore.getState().project;
    const copiedNode = updated.model.graphs[rootId].nodes.find((n) => n.id === 'node_mlp_1');
    expect(copiedNode).toBeDefined();
    expect(copiedNode?.definition_id).toMatch(/^custom\.graph_custom_/);

    const suffix = copiedNode!.definition_id.slice('custom.'.length);
    expect(updated.model.graphs[suffix]).toBeDefined();
    expect(updated.model.graphs[suffix].derived_from).toBe('builtin.nanogpt_mlp@1');
    expect(updated.model.graphs[suffix].modified).toBe(true);
    expect(updated.model.graphs.graph_mlp).toBeDefined();
  });

  it('toggleNodeCollapsed persists and restores ui.collapsed_node_ids', () => {
    const project = createInitialProject();
    const rootId = project.model.root_graph_id;
    useProjectStore.getState().loadProject(project);

    useProjectStore.getState().toggleNodeCollapsed('node_to_collapse');
    expect(useProjectStore.getState().project.ui.collapsed_node_ids?.[rootId]).toEqual(['node_to_collapse']);

    const savedJson = localStorage.getItem(PROJECT_STORAGE_KEY);
    expect(savedJson).not.toBeNull();

    useProjectStore.getState().loadProject(createInitialProject());
    localStorage.setItem(PROJECT_STORAGE_KEY, savedJson as string);

    const restored = readPersistedProject();
    if (restored.status !== 'loaded') {
      throw new Error(`Expected a saved project, received ${restored.status}`);
    }

    useProjectStore.getState().loadProject(restored.project);
    expect(useProjectStore.getState().project.ui.collapsed_node_ids?.[rootId]).toEqual(['node_to_collapse']);
  });
});