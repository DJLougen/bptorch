/**
 * Dynamic bento-box workspace shell with draggable panel sizing.
 */

import React from 'react';
import { BentoPanel } from './BentoPanel';
import { ResizeHandle } from './ResizeHandle';
import { useUIStore } from '../../stores/uiStore';

const COLLAPSED_PALETTE_WIDTH = 40;
const COLLAPSED_INSPECTOR_WIDTH = 40;
const COLLAPSED_DRAWER_HEIGHT = 32;

interface BentoShellProps {
  palette: React.ReactNode;
  canvas: React.ReactNode;
  inspector: React.ReactNode;
  drawer: React.ReactNode;
}

export const BentoShell: React.FC<BentoShellProps> = ({ palette, canvas, inspector, drawer }) => {
  const {
    isPaletteOpen,
    isInspectorOpen,
    isDrawerOpen,
    paletteWidth,
    inspectorWidth,
    drawerHeight,
    adjustPaletteWidth,
    adjustInspectorWidth,
    adjustDrawerHeight,
  } = useUIStore();

  const effectivePaletteWidth = isPaletteOpen ? paletteWidth : COLLAPSED_PALETTE_WIDTH;
  const effectiveInspectorWidth = isInspectorOpen ? inspectorWidth : COLLAPSED_INSPECTOR_WIDTH;
  const effectiveDrawerHeight = isDrawerOpen ? drawerHeight : COLLAPSED_DRAWER_HEIGHT;

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: 8,
        background: '#0c0d12',
      }}
    >
      <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 0 }}>
        <BentoPanel title={isPaletteOpen ? 'Node Palette' : undefined} style={{ width: effectivePaletteWidth, flexShrink: 0, borderRadius: isPaletteOpen ? 12 : 8 }}>
          {palette}
        </BentoPanel>

        <ResizeHandle axis="x" enabled={isPaletteOpen} onDelta={adjustPaletteWidth} />

        <BentoPanel title="Blueprint Canvas" style={{ flex: 1 }}>
          {canvas}
        </BentoPanel>

        <ResizeHandle axis="x" enabled={isInspectorOpen} onDelta={adjustInspectorWidth} />

        <BentoPanel title={isInspectorOpen ? 'Inspector' : undefined} style={{ width: effectiveInspectorWidth, flexShrink: 0, borderRadius: isInspectorOpen ? 12 : 8 }}>
          {inspector}
        </BentoPanel>
      </div>

      <ResizeHandle axis="y" enabled={isDrawerOpen} onDelta={adjustDrawerHeight} />

      <BentoPanel title={isDrawerOpen ? 'Diagnostics & Trace' : undefined} style={{ height: effectiveDrawerHeight, flexShrink: 0, borderRadius: isDrawerOpen ? 12 : 8 }}>
        {drawer}
      </BentoPanel>
    </div>
  );
};
