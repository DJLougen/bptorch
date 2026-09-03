/**
 * Custom Blueprint edge component rendering Exec wires (white with chevron pulse)
 * and Data wires (colored bezier curves with shape labels and flow-direction markers).
 */

import { memo, useId } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  Position,
  getBezierPath,
  useReactFlow,
  useStore,
} from '@xyflow/react';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import type { NodePosition } from '../api/contracts';
import { compactEdgeLabel } from './edgeLabels';

const EMPTY_WAYPOINTS: NodePosition[] = [];

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
    const waypoints =
      useProjectStore((s) => s.project.ui.edge_waypoints?.[s.openGraphId]?.[id]) ??
      EMPTY_WAYPOINTS;
    const setEdgeWaypoints = useProjectStore((s) => s.setEdgeWaypoints);
    const reactFlowInstance = useReactFlow();
    const zoom = useStore((s) => s.transform[2]);

    const isExec = Boolean(data?.isExec);
    const rawLabel = (data?.shapeLabel as string) || undefined;
    const displayLabel = compactEdgeLabel(rawLabel, zoom);
    const strokeColor = isSelected ? '#38bdf8' : isExec ? '#ffffff' : '#475569';
    const flowAccent = isExec ? '#e2e8f0' : isSelected ? '#7dd3fc' : '#64748b';
    const strokeWidth = isSelected ? 3 : isExec ? 2.5 : 2;

    const allPoints = [{ x: sourceX, y: sourceY }, ...waypoints, { x: targetX, y: targetY }];
    const segments: Array<{ path: string; labelX: number; labelY: number }> = [];
    for (let i = 0; i < allPoints.length - 1; i++) {
      const pStart = allPoints[i];
      const pEnd = allPoints[i + 1];
      const sPos = i === 0 ? sourcePosition : Position.Right;
      const tPos = i === allPoints.length - 2 ? targetPosition : Position.Left;
      const [segPath, segLabelX, segLabelY] = getBezierPath({
        sourceX: pStart.x,
        sourceY: pStart.y,
        sourcePosition: sPos,
        targetX: pEnd.x,
        targetY: pEnd.y,
        targetPosition: tPos,
      });
      segments.push({ path: segPath, labelX: segLabelX, labelY: segLabelY });
    }

    const midSegIdx = Math.floor(segments.length / 2);
    const midSegment = segments[midSegIdx];
    const labelX = midSegment.labelX;
    const labelY = midSegment.labelY;
    const angle = flowAngleDeg(sourceX, sourceY, targetX, targetY);

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
        {segments.map((seg, i) => (
          <BaseEdge
            key={`${id}_seg_${i}`}
            id={`${id}_seg_${i}`}
            path={seg.path}
            markerEnd={i === segments.length - 1 ? `url(#${markerId})` : undefined}
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
        ))}

        {/* Animated dashes show direction of flow along the wire */}
        {segments.map((seg, i) => (
          <path
            key={`flow_${id}_seg_${i}`}
            d={seg.path}
            fill="none"
            pointerEvents="none"
            stroke={flowAccent}
            strokeWidth={Math.max(1.5, strokeWidth - 0.5)}
            strokeDasharray={isExec ? '5 9' : '7 11'}
            strokeLinecap="round"
            className="blueprint-edge-flow"
            style={{ opacity: isSelected ? 0.95 : 0.7 }}
          />
        ))}

        {/* Mid-edge chevron reinforces source → target direction */}
        <g transform={`translate(${labelX}, ${labelY}) rotate(${angle})`} pointerEvents="none">
          <polygon
            points="-5,-4 6,0 -5,4"
            fill={flowAccent}
            opacity={isSelected ? 0.95 : 0.75}
          />
        </g>

        {waypoints.length > 0 && (
          <EdgeLabelRenderer>
            {waypoints.map((pt, wpIdx) => (
              <div
                key={`wp_${id}_${wpIdx}`}
                style={{
                  position: 'absolute',
                  transform: `translate(-50%, -50%) translate(${pt.x}px, ${pt.y}px)`,
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: isSelected ? '#38bdf8' : '#94a3b8',
                  border: '1px solid #0f172a',
                  cursor: 'grab',
                  pointerEvents: 'all',
                  zIndex: 20,
                }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  const onPointerMove = (moveEvt: PointerEvent) => {
                    const newFlowPos = reactFlowInstance.screenToFlowPosition({
                      x: moveEvt.clientX,
                      y: moveEvt.clientY,
                    });
                    const updated = [...waypoints];
                    updated[wpIdx] = newFlowPos;
                    setEdgeWaypoints(id, updated);
                  };
                  const onPointerUp = () => {
                    window.removeEventListener('pointermove', onPointerMove);
                    window.removeEventListener('pointerup', onPointerUp);
                  };
                  window.addEventListener('pointermove', onPointerMove);
                  window.addEventListener('pointerup', onPointerUp);
                }}
              />
            ))}
          </EdgeLabelRenderer>
        )}

        {displayLabel && !isExec && (
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
              {displayLabel}
            </div>
          </EdgeLabelRenderer>
        )}
      </>
    );
  }
);
