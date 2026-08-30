import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { NodeDefinitionSummary } from '../src/api/contracts';
import { NodePalette } from '../src/palette/NodePalette';

const mockCatalog: NodeDefinitionSummary[] = [
  {
    type_id: 'builtin.linear@1',
    version: 1,
    display_name: 'Linear',
    category: 'Layers',
    description: 'Linear transformation',
    icon: 'GitCommit',
    is_composite: false,
    property_schema: {},
    default_inputs: [],
    default_outputs: [],
  },
  {
    type_id: 'builtin.gelu@1',
    version: 1,
    display_name: 'GELU',
    category: 'Layers',
    description: 'GELU activation',
    icon: 'Activity',
    is_composite: false,
    property_schema: {},
    default_inputs: [],
    default_outputs: [],
  },
];

describe('NodePalette Component', () => {
  it('renders categories and filters nodes with search', () => {
    render(<NodePalette catalog={mockCatalog} />);

    expect(screen.getByText(/Layers/i)).toBeInTheDocument();
    expect(screen.getByText('Linear')).toBeInTheDocument();
    expect(screen.getByText('GELU')).toBeInTheDocument();

    // Search filter
    const searchInput = screen.getByPlaceholderText('Search nodes...');
    fireEvent.change(searchInput, { target: { value: 'Linear' } });

    expect(screen.getByText('Linear')).toBeInTheDocument();
    expect(screen.queryByText('GELU')).not.toBeInTheDocument();
  });
});
