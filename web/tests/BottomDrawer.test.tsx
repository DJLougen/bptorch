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
