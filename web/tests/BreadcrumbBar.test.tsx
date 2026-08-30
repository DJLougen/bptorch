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

  it('disables the block instance switcher', () => {
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
    expect(screen.getByTitle('Block instance switching is not yet implemented')).toBeDisabled();
  });

});