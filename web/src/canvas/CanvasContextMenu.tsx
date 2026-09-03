/**
 * Context menu for ReactFlow canvas pane, nodes, and edges.
 */

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import {
  AlignHorizontalJustifyStart,
  AlignVerticalJustifyStart,
  CircleDot,
  Copy,
  FolderInput,
  FolderPlus,
  Maximize2,
  Minimize2,
  Plus,
  Scissors,
  Sliders,
  Trash2,
  Wand2,
  XCircle,
} from 'lucide-react';

export interface CanvasContextMenuProps {
  kind: 'pane' | 'node' | 'edge';
  x: number;
  y: number;
  nodeId?: string;
  edgeId?: string;
  onClose: () => void;
  // Pane actions
  onPaste?: () => void;
  canPaste?: boolean;
  onQuickAdd?: () => void;
  onAutoLayout?: () => void;
  // Node actions
  onCopy?: () => void;
  onDuplicate?: () => void;
  onDeleteNode?: () => void;
  onGroupModule?: () => void;
  onCreateEditableCopy?: () => void;
  canCreateEditableCopy?: boolean;
  onToggleBreakpoint?: () => void;
  isBreakpoint?: boolean;
  onToggleDisabled?: () => void;
  isDisabled?: boolean;
  onOpenSubgraph?: () => void;
  canOpenSubgraph?: boolean;
  onToggleCollapse?: () => void;
  isCollapsed?: boolean;
  onAlignLeft?: () => void;
  onAlignTop?: () => void;
  onZoomToSelection?: () => void;
  hasMultiSelection?: boolean;
  // Edge actions
  onDeleteEdge?: () => void;
  onRemoveWaypoints?: () => void;
  hasWaypoints?: boolean;
}

const menuPanelStyle: React.CSSProperties = {
  position: 'fixed',
  minWidth: 200,
  background: '#10121a',
  border: '1px solid #272c3b',
  borderRadius: 8,
  boxShadow: '0 12px 32px rgba(0, 0, 0, 0.45)',
  padding: '6px 0',
  zIndex: 5000,
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
};

interface MenuItemProps {
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}

const MenuItem: React.FC<MenuItemProps> = ({
  icon,
  label,
  shortcut,
  onClick,
  disabled = false,
  danger = false,
}) => {
  const [hovered, setHovered] = React.useState(false);

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        if (!disabled) {
          onClick();
        }
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      disabled={disabled}
      style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
        background: hovered && !disabled ? '#1a202c' : 'transparent',
        border: 'none',
        color: disabled ? '#475569' : danger ? '#f87171' : '#e2e8f0',
        padding: '6px 12px',
        fontSize: 12,
        cursor: disabled ? 'not-allowed' : 'pointer',
        textAlign: 'left',
        transition: 'background 0.1s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {icon}
        <span>{label}</span>
      </div>
      {shortcut && <span style={{ fontSize: 10, color: '#64748b' }}>{shortcut}</span>}
    </button>
  );
};

export const CanvasContextMenu: React.FC<CanvasContextMenuProps> = (props) => {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handlePointerDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        props.onClose();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        props.onClose();
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [props]);

  const left = Math.min(props.x, window.innerWidth - 220);
  const top = Math.min(props.y, window.innerHeight - 300);

  return createPortal(
    <div
      ref={menuRef}
      role="menu"
      aria-label="Canvas context menu"
      style={{
        ...menuPanelStyle,
        left: Math.max(8, left),
        top: Math.max(8, top),
      }}
      onClick={(e) => e.stopPropagation()}
    >
      {props.kind === 'pane' && (
        <>
          <MenuItem
            icon={<Plus size={14} />}
            label="Quick Add"
            shortcut="Space"
            onClick={() => {
              props.onQuickAdd?.();
              props.onClose();
            }}
          />
          <MenuItem
            icon={<Scissors size={14} />}
            label="Paste"
            shortcut="Ctrl+V"
            disabled={!props.canPaste}
            onClick={() => {
              props.onPaste?.();
              props.onClose();
            }}
          />
          <MenuItem
            icon={<Wand2 size={14} />}
            label="Auto-layout"
            onClick={() => {
              props.onAutoLayout?.();
              props.onClose();
            }}
          />
        </>
      )}

      {props.kind === 'node' && (
        <>
          <MenuItem
            icon={<Copy size={14} />}
            label="Copy"
            shortcut="Ctrl+C"
            onClick={() => {
              props.onCopy?.();
              props.onClose();
            }}
          />
          <MenuItem
            icon={<Copy size={14} />}
            label="Duplicate"
            shortcut="Ctrl+D"
            onClick={() => {
              props.onDuplicate?.();
              props.onClose();
            }}
          />
          <MenuItem
            icon={<FolderPlus size={14} />}
            label="Group into Module"
            shortcut="Ctrl+G"
            onClick={() => {
              props.onGroupModule?.();
              props.onClose();
            }}
          />
          {props.canCreateEditableCopy && (
            <MenuItem
              icon={<Copy size={14} />}
              label="Create Editable Copy"
              onClick={() => {
                props.onCreateEditableCopy?.();
                props.onClose();
              }}
            />
          )}
          <MenuItem
            icon={<CircleDot size={14} />}
            label={props.isBreakpoint ? 'Remove Breakpoint' : 'Set Breakpoint'}
            onClick={() => {
              props.onToggleBreakpoint?.();
              props.onClose();
            }}
          />
          <MenuItem
            icon={<Sliders size={14} />}
            label={props.isDisabled ? 'Enable Node' : 'Disable Node'}
            onClick={() => {
              props.onToggleDisabled?.();
              props.onClose();
            }}
          />
          {props.canOpenSubgraph && (
            <MenuItem
              icon={<FolderInput size={14} />}
              label="Open Subgraph"
              onClick={() => {
                props.onOpenSubgraph?.();
                props.onClose();
              }}
            />
          )}
          {props.onToggleCollapse && (
            <MenuItem
              icon={props.isCollapsed ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
              label={props.isCollapsed ? 'Expand Node' : 'Collapse Node'}
              onClick={() => {
                props.onToggleCollapse?.();
                props.onClose();
              }}
            />
          )}
          {props.hasMultiSelection && props.onAlignLeft && (
            <MenuItem
              icon={<AlignHorizontalJustifyStart size={14} />}
              label="Align Left"
              onClick={() => {
                props.onAlignLeft?.();
                props.onClose();
              }}
            />
          )}
          {props.hasMultiSelection && props.onAlignTop && (
            <MenuItem
              icon={<AlignVerticalJustifyStart size={14} />}
              label="Align Top"
              onClick={() => {
                props.onAlignTop?.();
                props.onClose();
              }}
            />
          )}
          {props.onZoomToSelection && (
            <MenuItem
              icon={<Maximize2 size={14} />}
              label="Zoom to Selection"
              shortcut="Shift+2"
              onClick={() => {
                props.onZoomToSelection?.();
                props.onClose();
              }}
            />
          )}
          <MenuItem
            icon={<Trash2 size={14} />}
            label="Delete"
            shortcut="Del"
            danger
            onClick={() => {
              props.onDeleteNode?.();
              props.onClose();
            }}
          />
        </>
      )}

      {props.kind === 'edge' && (
        <>
          {props.hasWaypoints && props.onRemoveWaypoints && (
            <MenuItem
              icon={<XCircle size={14} />}
              label="Remove Waypoints"
              onClick={() => {
                props.onRemoveWaypoints?.();
                props.onClose();
              }}
            />
          )}
          <MenuItem
            icon={<Trash2 size={14} />}
            label="Delete"
            shortcut="Del"
            danger
            onClick={() => {
              props.onDeleteEdge?.();
              props.onClose();
            }}
          />
        </>
      )}
    </div>,
    document.body
  );
};
