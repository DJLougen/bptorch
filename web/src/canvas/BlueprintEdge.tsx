/**
 * Custom Blueprint edge component rendering Exec wires (white with chevron pulse)
 * and Data wires (colored bezier curves with shape labels and flow-direction markers).
 */

import { memo, useId } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  getBezierPath,
} from '@xyflow/react';
import { useUIStore } from '../stores/uiStore';

function flowAngleDeg(sourceX: number, sourceY: number, targetX: number, targetY: number): number {
  return (Math.atan2(targetY - sourceY, targetX - sourceX) * 180) / Math.PI;
}

export const BlueprintEdge = memo(
  ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    selected,
    data,
  }: EdgeProps) => {
    const { selectedEdgeId, selectEdge } = useUIStore();
    const isSelected = selected || selectedEdgeId === id;
    const markerId = useId();

    const [edgePath, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });

    const isExec = Boolean(data?.isExec);
    const label = (data?.shapeLabel as string) || undefined;
    const angle = flowAngleDeg(sourceX, sourceY, targetX, targetY);

    const strokeColor = isSelected
      ? '#38bdf8'
      : isExec
        ? '#ffffff'
        : '#475569';

    const flowAccent = isExec ? '#e2e8f0' : isSelected ? '#7dd3fc' : '#64748b';
    const strokeWidth = isSelected ? 3 : isExec ? 2.5 : 2;

    return (
      <>
        <defs>
          <marker
            id={markerId}
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="6"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M0,0 L0,12 L12,6 z" fill={strokeColor} />
          </marker>
        </defs>
        <BaseEdge
          id={id}
          path={edgePath}
          markerEnd={`url(#${markerId})`}
          interactionWidth={20}
          onClick={() => selectEdge(id)}
          style={{
            ...style,
            stroke: strokeColor,
            strokeWidth,
            strokeDasharray: isExec ? '6 4' : undefined,
            transition: 'stroke 0.15s ease',
          }}
        />

        {/* Animated dashes show direction of flow along the wire */}
        <path
          d={edgePath}
          fill="none"
          pointerEvents="none"
          stroke={flowAccent}
          strokeWidth={Math.max(1.5, strokeWidth - 0.5)}
          strokeDasharray={isExec ? '5 9' : '7 11'}
          strokeLinecap="round"
          className="blueprint-edge-flow"
          style={{ opacity: isSelected ? 0.95 : 0.7 }}
        />

        {/* Mid-edge chevron reinforces source → target direction */}
        <g transform={`translate(${labelX}, ${labelY}) rotate(${angle})`} pointerEvents="none">
          <polygon
            points="-5,-4 6,0 -5,4"
            fill={flowAccent}
            opacity={isSelected ? 0.95 : 0.75}
          />
        </g>

        {label && !isExec && (
          <EdgeLabelRenderer>
            <div
              style={{
                position: 'absolute',
                transform: `translate(-50%, -50%) translate(${labelX}px,${labelY - 14}px)`,
                fontSize: 10,
                fontFamily: 'monospace',
                pointerEvents: 'all',
                background: '#0f172a',
                border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                color: isSelected ? '#38bdf8' : '#94a3b8',
                padding: '2px 6px',
                borderRadius: 4,
                cursor: 'pointer',
                boxShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
                zIndex: 10,
              }}
              onClick={(e) => {
                e.stopPropagation();
                selectEdge(id);
              }}
            >
              {label}
            </div>
          </EdgeLabelRenderer>
        )}
      </>
    );
  }
);
