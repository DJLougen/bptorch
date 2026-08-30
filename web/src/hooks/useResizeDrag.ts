/**
 * Pointer-driven resize hook for bento panel handles.
 */

import { useCallback, useEffect, useRef } from 'react';

type ResizeAxis = 'x' | 'y';

interface UseResizeDragOptions {
  axis: ResizeAxis;
  onDelta: (delta: number) => void;
  enabled?: boolean;
}

export function useResizeDrag({ axis, onDelta, enabled = true }: UseResizeDragOptions) {
  const draggingRef = useRef(false);
  const lastPosRef = useRef(0);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!enabled) return;
      draggingRef.current = true;
      lastPosRef.current = axis === 'x' ? event.clientX : event.clientY;
      event.currentTarget.setPointerCapture(event.pointerId);
      document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    },
    [axis, enabled],
  );

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      if (!draggingRef.current) return;
      const current = axis === 'x' ? event.clientX : event.clientY;
      const delta = current - lastPosRef.current;
      if (delta !== 0) {
        onDelta(delta);
        lastPosRef.current = current;
      }
    };

    const onPointerUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [axis, onDelta]);

  return { onPointerDown };
}
