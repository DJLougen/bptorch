import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { BreadcrumbBar } from '../src/components/BreadcrumbBar';
import { createInitialProject, useProjectStore } from '../src/stores/projectStore';

describe('BreadcrumbBar Component', () => {
  it('renders root breadcrumb', () => {
    // Ground state
    useProjectStore.setState({
      openGraphId: 'graph_gpt',
      project: {
        schema_version: 1,
        project: {
          id: 'p1',
          name: 'nanoGPT Experiment',
          created_at: '',
          updated_at: '',
        },
        model: {
          root_graph_id: 'graph_gpt',
          config: { n_layer: 2 },
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
      },
    });

    render(<BreadcrumbBar />);
    expect(screen.getByText('nanoGPT')).toBeInTheDocument();
  });

  it('enables the block instance switcher', () => {
    const project = createInitialProject();
    project.model.graphs.graph_block = {
      id: 'graph_block',
      name: 'Repeat Graph',
      kind: 'root',
      interface: { inputs: [], outputs: [] },
      nodes: [],
      edges: [],
    };
    project.model.config.n_layer = 2;
    useProjectStore.setState({
      project,
      openGraphId: 'graph_block',
      graphHistory: ['graph_block'],
      historyIndex: 0,
    });

    render(<BreadcrumbBar />);
    expect(screen.getByTitle('Inspect activations for this repeated block')).not.toBeDisabled();
  });

  it('resolves dynamic breadcrumbs from composite parent map without attention naming', () => {
    const project = createInitialProject();
    project.model.root_graph_id = 'graph_root';
    project.model.graphs = {
      graph_root: {
        id: 'graph_root',
        name: 'Root Graph',
        kind: 'root',
        interface: { inputs: [], outputs: [] },
        nodes: [
          {
            id: 'n_mid',
            definition_id: 'custom.mid',
            display_name: 'Mid Subgraph',
            properties: {},
            metadata: { breakpoint: false, disabled: false },
          },
        ],
        edges: [],
      },
      mid: {
        id: 'mid',
        name: 'Middle Subgraph',
        kind: 'module',
        interface: { inputs: [], outputs: [] },
        nodes: [
          {
            id: 'n_block',
            definition_id: 'builtin.nanogpt_block@1',
            display_name: 'Block Node',
            properties: {},
            metadata: { breakpoint: false, disabled: false },
          },
        ],
        edges: [],
      },
      graph_block: {
        id: 'graph_block',
        name: 'Transformer Block',
        kind: 'module',
        interface: { inputs: [], outputs: [] },
        nodes: [],
        edges: [],
      },
    };

    useProjectStore.setState({
      project,
      openGraphId: 'graph_block',
      graphHistory: ['graph_block'],
      historyIndex: 0,
    });

    render(<BreadcrumbBar />);
    expect(screen.getByText('Root Graph')).toBeInTheDocument();
    expect(screen.getByText('Transformer Block')).toBeInTheDocument();
  });

  it('enables the Architecture button with show-architecture title', () => {
    useProjectStore.setState({
      openGraphId: 'graph_gpt',
      project: createInitialProject(),
    });

    render(<BreadcrumbBar />);
    const archBtn = screen.getByRole('button', { name: /Architecture/i });
    expect(archBtn).not.toBeDisabled();
    expect(archBtn).toHaveAttribute('title', 'Show architecture graph');
  });
});