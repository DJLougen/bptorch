/**
 * UI layout and presentation Zustand store with responsive sidebar defaults.
 */

import { create } from 'zustand';

export type DrawerTab = 'tensor' | 'diagnostics' | 'parameters' | 'parity' | 'logs' | 'loss' | 'metrics' | 'tester';
export type TraceSpeed = 'instant' | 'fast' | 'normal' | 'step';

interface UIState {
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  activeDrawerTab: DrawerTab | null;
  isDrawerOpen: boolean;
  isPaletteOpen: boolean;
  isInspectorOpen: boolean;
  repeatInstanceIndex: number;
  traceSpeed: TraceSpeed;
  paletteSearchQuery: string;
  selectedPaletteCategory: string | null;
  paletteWidth: number;
  inspectorWidth: number;
  drawerHeight: number;

  // Actions
  selectNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  openDrawerTab: (tab: DrawerTab) => void;
  closeDrawer: () => void;
  toggleDrawer: () => void;
  togglePalette: () => void;
  toggleInspector: () => void;
  setRepeatInstanceIndex: (index: number) => void;
  setTraceSpeed: (speed: TraceSpeed) => void;
  setPaletteSearchQuery: (query: string) => void;
  setSelectedPaletteCategory: (category: string | null) => void;
  adjustPaletteWidth: (delta: number) => void;
  adjustInspectorWidth: (delta: number) => void;
  adjustDrawerHeight: (delta: number) => void;
}

export const useUIStore = create<UIState>((set) => ({
  selectedNodeId: null,
  selectedEdgeId: null,
  activeDrawerTab: null,
  isDrawerOpen: false,
  isPaletteOpen: true,
  isInspectorOpen: true,
  repeatInstanceIndex: 0,
  traceSpeed: 'normal',
  paletteSearchQuery: '',
  selectedPaletteCategory: null,
  paletteWidth: 260,
  inspectorWidth: 320,
  drawerHeight: 260,

  selectNode: (nodeId: string | null) => {
    set({
      selectedNodeId: nodeId,
      selectedEdgeId: null,
      activeDrawerTab: nodeId ? 'tensor' : null,
      isDrawerOpen: Boolean(nodeId),
      isInspectorOpen: true,
    });
  },

  selectEdge: (edgeId: string | null) => {
    set({
      selectedEdgeId: edgeId,
      selectedNodeId: null,
      activeDrawerTab: edgeId ? 'tensor' : null,
      isDrawerOpen: Boolean(edgeId),
      isInspectorOpen: true,
    });
  },

  openDrawerTab: (tab: DrawerTab) => {
    set({
      activeDrawerTab: tab,
      isDrawerOpen: true,
    });
  },

  closeDrawer: () => {
    set({
      isDrawerOpen: false,
      activeDrawerTab: null,
    });
  },

  toggleDrawer: () => {
    set((state) => ({
      isDrawerOpen: !state.isDrawerOpen,
      activeDrawerTab: !state.isDrawerOpen ? 'loss' : null,
    }));
  },

  togglePalette: () => {
    set((state) => ({ isPaletteOpen: !state.isPaletteOpen }));
  },

  toggleInspector: () => {
    set((state) => ({ isInspectorOpen: !state.isInspectorOpen }));
  },

  setRepeatInstanceIndex: (index: number) => {
    set({ repeatInstanceIndex: index });
  },

  setTraceSpeed: (speed: TraceSpeed) => {
    set({ traceSpeed: speed });
  },

  setPaletteSearchQuery: (query: string) => {
    set({ paletteSearchQuery: query });
  },

  setSelectedPaletteCategory: (category: string | null) => {
    set({ selectedPaletteCategory: category });
  },
  adjustPaletteWidth: (delta: number) => {
    set((state) => ({
      paletteWidth: Math.min(480, Math.max(180, state.paletteWidth + delta)),
    }));
  },

  adjustInspectorWidth: (delta: number) => {
    set((state) => ({
      inspectorWidth: Math.min(520, Math.max(220, state.inspectorWidth - delta)),
    }));
  },

  adjustDrawerHeight: (delta: number) => {
    set((state) => ({
      drawerHeight: Math.min(520, Math.max(140, state.drawerHeight - delta)),
    }));
  },
}));
