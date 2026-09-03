import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { ApiClient } from '../src/api/client';
import { BottomDrawer } from '../src/components/BottomDrawer';
import { useProjectStore } from '../src/stores/projectStore';
import { useUIStore } from '../src/stores/uiStore';

describe('BottomDrawer tester tab', () => {
  beforeEach(() => {
    useUIStore.setState({ isDrawerOpen: true, activeDrawerTab: 'tester' });
    useProjectStore.getState().loadProject(useProjectStore.getState().project);
  });

  it('shows an error when the test suite request fails', async () => {
    vi.spyOn(ApiClient, 'runTestSuite').mockRejectedValue(new Error('network down'));
    render(<BottomDrawer />);
    fireEvent.click(screen.getByRole('button', { name: /Run Test Suite/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('network down');
  });
});
describe('BottomDrawer playground tab', () => {
  beforeEach(() => {
    useUIStore.setState({ isDrawerOpen: true, activeDrawerTab: 'loss' });
    useProjectStore.getState().loadProject(useProjectStore.getState().project);
  });

  it('has playground tab button and displays playground controls when selected', () => {
    render(<BottomDrawer />);
    const playgroundTabBtn = screen.getByRole('button', { name: /Playground/i });
    expect(playgroundTabBtn).toBeInTheDocument();

    fireEvent.click(playgroundTabBtn);
    expect(screen.getByRole('button', { name: /Generate/i })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Prompt template' })).toBeInTheDocument();
    expect(screen.getByLabelText(/KV Cache/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Top-k')).toBeInTheDocument();
    expect(screen.getByLabelText('Top-p')).toBeInTheDocument();
  });
});

describe('BottomDrawer cook tab', () => {
  beforeEach(() => {
    useUIStore.setState({ isDrawerOpen: true, activeDrawerTab: 'code' });
    useProjectStore.getState().loadProject(useProjectStore.getState().project);
  });

  it('shows an alert when cook export fails', async () => {
    vi.spyOn(ApiClient, 'cookExport').mockRejectedValue(
      new Error('Cook export does not support dual-flow training pipelines')
    );
    render(<BottomDrawer />);
    fireEvent.click(screen.getByRole('button', { name: /Export PyTorch Code/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Cook export does not support dual-flow training pipelines'
    );
  });
});
describe('BottomDrawer loss tab controls', () => {
  beforeEach(() => {
    useUIStore.setState({ isDrawerOpen: true, activeDrawerTab: 'loss' });
    useProjectStore.getState().loadProject(useProjectStore.getState().project);
  });

  it('renders dataset select and save checkpoint button when drawer open on loss tab', () => {
    render(<BottomDrawer />);
    expect(screen.getByRole('combobox', { name: 'Training dataset' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Upload .txt' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save checkpoint' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Load checkpoint' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Validation' })).toBeInTheDocument();
    expect(screen.getByLabelText('Batch size')).toBeInTheDocument();
    expect(screen.getByLabelText('Validation fraction')).toBeInTheDocument();
  });
});
