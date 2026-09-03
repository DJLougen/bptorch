import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Project } from '../src/api/contracts';
import { TopBar } from '../src/components/TopBar';
import {
  createInitialProject,
  readPersistedProject,
  serializeProject,
  useProjectStore,
} from '../src/stores/projectStore';
import { useTraceStore } from '../src/stores/traceStore';
import { useValidationStore } from '../src/stores/validationStore';

describe('TopBar project JSON controls', () => {
  let originalRunValidation: (project: Project) => Promise<void>;

  beforeEach(() => {
    localStorage.clear();
    useProjectStore.getState().loadProject(createInitialProject());
    localStorage.clear();
    useTraceStore.setState({ logs: [], status: 'idle' });

    originalRunValidation = useValidationStore.getState().runValidation;
    const runValidation = vi.fn<(project: Project) => Promise<void>>();
    runValidation.mockResolvedValue(undefined);
    useValidationStore.setState({ diagnostics: [], runValidation });
  });

  afterEach(() => {
    useValidationStore.setState({ runValidation: originalRunValidation });
  });

  it('imports guarded project JSON and persists the imported project', async () => {
    const importedProject = createInitialProject();
    importedProject.project.id = 'imported_project';
    importedProject.project.name = 'Imported Project';
    const file = new File(['project'], 'imported.nbp.json', {
      type: 'application/json',
    });
    Object.defineProperty(file, 'text', {
      value: () => Promise.resolve(serializeProject(importedProject)),
    });

    render(<TopBar />);
    expect(screen.getByRole('button', { name: 'Import project JSON' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Export project JSON' })).toBeVisible();

    fireEvent.change(screen.getByLabelText('Import project JSON file'), {
      target: { files: [file] },
    });

    await waitFor(() => {
      expect(useProjectStore.getState().project.project.name).toBe('Imported Project');
    });

    const persisted = readPersistedProject();
    expect(persisted.status).toBe('loaded');
    if (persisted.status !== 'loaded') {
      throw new Error('Expected the imported project to be persisted');
    }
    expect(persisted.project).toEqual(importedProject);
  });

  it('leaves the current project intact when imported JSON fails the shape guard', async () => {
    const originalProject = useProjectStore.getState().project;
    const file = new File(['invalid'], 'invalid.json', { type: 'application/json' });
    Object.defineProperty(file, 'text', {
      value: () => Promise.resolve(JSON.stringify({ schema_version: 1 })),
    });

    render(<TopBar />);
    fireEvent.change(screen.getByLabelText('Import project JSON file'), {
      target: { files: [file] },
    });

    await waitFor(() => {
      expect(useTraceStore.getState().logs.at(-1)?.message).toMatch(/import rejected/i);
    });
    expect(useProjectStore.getState().project).toBe(originalProject);
  });

  it('labels parity as belonging only to the bundled nanoGPT baseline', () => {
    render(<TopBar />);

    const parityBadge = screen.getByRole('button', {
      name: 'Bundled nanoGPT baseline parity is verified',
    });
    expect(parityBadge).toHaveAttribute('title');
    expect(parityBadge.getAttribute('title')).toMatch(/not the current project/i);
  });

  it('renders Compile model and Save project buttons', () => {
    render(<TopBar />);
    expect(screen.getByLabelText('Compile model')).toBeInTheDocument();
    expect(screen.getByLabelText('Save project')).toBeInTheDocument();
    expect(screen.getByLabelText('Import PyTorch module')).toBeInTheDocument();
  });
});

describe('TopBar validation badge', () => {
  it('does not show Graph Valid while validation is in flight', () => {
    useValidationStore.setState({
      isValidating: true,
      lastValidatedTimestamp: 0,
      diagnostics: [],
    });
    render(<TopBar />);
    expect(screen.queryByText('Graph Valid')).toBeNull();
    expect(screen.getByText('Validating…')).toBeVisible();
  });
});
describe('TopBar auto-layout', () => {
  it('has Auto-layout button and clicking it separates overlapping node positions', () => {
    const project = createInitialProject();
    project.model.root_graph_id = 'graph_test';
    project.model.graphs = {
      graph_test: {
        id: 'graph_test',
        name: 'Test Graph',
        kind: 'root',
        interface: { inputs: [], outputs: [] },
        nodes: [
          {
            id: 'n1',
            definition_id: 'builtin.linear@1',
            display_name: 'Node 1',
            properties: {},
            metadata: { breakpoint: false, disabled: false },
          },
          {
            id: 'n2',
            definition_id: 'builtin.linear@1',
            display_name: 'Node 2',
            properties: {},
            metadata: { breakpoint: false, disabled: false },
          },
        ],
        edges: [
          {
            id: 'e1',
            source: { node_id: 'n1', port_id: 'output' },
            target: { node_id: 'n2', port_id: 'input' },
          },
        ],
      },
    };
    project.ui.open_graph_id = 'graph_test';
    project.ui.node_positions = {
      graph_test: {
        n1: { x: 100, y: 100 },
        n2: { x: 100, y: 100 },
      },
    };

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<TopBar />);
    const autoLayoutBtn = screen.getByRole('button', { name: 'Auto-layout graph' });
    expect(autoLayoutBtn).toBeVisible();

    fireEvent.click(autoLayoutBtn);

    const positions = useProjectStore.getState().project.ui.node_positions.graph_test;
    expect(positions.n1.x).not.toEqual(positions.n2.x);
    expect(positions.n1.x).toBeLessThan(positions.n2.x);
  });
});

