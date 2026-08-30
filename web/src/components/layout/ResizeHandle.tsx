/**
 * Draggable separator between bento panels.
 */

import React from 'react';
import { useResizeDrag } from '../../hooks/useResizeDrag';

interface ResizeHandleProps {
  axis: 'x' | 'y';
  onDelta: (delta: number) => void;
  enabled?: boolean;
}

export const ResizeHandle: React.FC<ResizeHandleProps> = ({ axis, onDelta, enabled = true }) => {
  const { onPointerDown } = useResizeDrag({ axis, onDelta, enabled });

  const isVertical = axis === 'x';

  return (
    <div
      role="separator"
      aria-orientation={isVertical ? 'vertical' : 'horizontal'}
      onPointerDown={onPointerDown}
      style={{
        flexShrink: 0,
        width: isVertical ? 6 : '100%',
        height: isVertical ? '100%' : 6,
        cursor: isVertical ? 'col-resize' : 'row-resize',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        touchAction: 'none',
        opacity: enabled ? 1 : 0.35,
      }}
    >
      <div
        style={{
          width: isVertical ? 2 : '48px',
          height: isVertical ? '48px' : 2,
          borderRadius: 999,
          background: '#334155',
        }}
      />
    </div>
  );
};
