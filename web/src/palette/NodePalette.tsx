/**
 * Left-side Node Palette supporting search, category grouping, and drag-and-drop onto the canvas.
 */

import React, { useState } from 'react';
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Box,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Columns,
  Combine,
  Crosshair,
  Divide,
  Flame,
  GitCommit,
  GitFork,
  Key,
  Layers,
  ListOrdered,
  LogIn,
  LogOut,
  Maximize2,
  Minimize2,
  PanelLeft,
  Percent,
  PieChart,
  Plus,
  RefreshCw,
  Scissors,
  Search,
  Settings,
  Shield,
  Sliders,
  Target,
  Terminal,
  X,
  Zap,
} from 'lucide-react';
import { NodeDefinitionSummary, NodeInstance } from '../api/contracts';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';

const ICON_MAP: Record<string, React.FC<{ size?: number; className?: string }>> = {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Box,
  Columns,
  Combine,
  Crosshair,
  Divide,
  Flame,
  GitCommit,
  GitFork,
  Key,
  Layers,
  ListOrdered,
  LogIn,
  LogOut,
  Maximize2,
  Minimize2,
  Percent,
  PieChart,
  Plus,
  RefreshCw,
  Scissors,
  Settings,
  Shield,
  Sliders,
  Target,
  Terminal,
  X,
  Zap,
};

const CATEGORIES = [
  'Flow Control',
  'Events',
  'Variables',
  'Data Pipelines',
  'Optimization',
  'LR Schedulers',
  'Metrics & Evaluation',
  'Persistence',
  'Inputs',
  'Layers',
  'Tensor Operations',
  'Attention',
  'Composite Modules',
  'Loss & Outputs',
];

export const NodePalette: React.FC<{ catalog?: NodeDefinitionSummary[] }> = ({ catalog = [] }) => {
  const { addNode } = useProjectStore();
  const { paletteSearchQuery, setPaletteSearchQuery, isPaletteOpen, togglePalette } = useUIStore();
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({});

  const toggleCategory = (cat: string) => {
    setCollapsedCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  const safeCatalog = Array.isArray(catalog) ? catalog : [];
  const filteredCatalog = safeCatalog.filter((item) => {
    if (!paletteSearchQuery) return true;
    const q = paletteSearchQuery.toLowerCase();
    return (
      item.display_name.toLowerCase().includes(q) ||
      item.description.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q)
    );
  });

  const grouped = CATEGORIES.map((category) => ({
    category,
    items: filteredCatalog.filter((item) => item.category === category),
  }));

  const handleDragStart = (event: React.DragEvent, typeId: string) => {
    event.dataTransfer.setData('application/neural-blueprint-node', typeId);
    event.dataTransfer.effectAllowed = 'move';
  };

  const handleQuickAdd = (defn: NodeDefinitionSummary) => {
    const rawType = defn.type_id.includes('.') ? defn.type_id.split('.')[1].split('@')[0] : 'node';
    const newNodeId = `node_${rawType}_${Date.now()}`;
    const newNode: NodeInstance = {
      id: newNodeId,
      definition_id: defn.type_id,
      display_name: defn.display_name,
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    };
    addNode(newNode);
  };

  if (!isPaletteOpen) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          background: '#12141c',
          borderRight: '1px solid #1f2430',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: 10,
        }}
      >
        <button
          onClick={togglePalette}
          title="Expand Palette"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            padding: 6,
          }}
        >
          <PanelLeft size={16} />
        </button>
      </div>
    );
  }

  return (
    <aside
      style={{
        width: '100%',
        height: '100%',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Search Input & Collapse Button */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #1f2430', display: 'flex', alignItems: 'center', gap: 6 }}>
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            background: '#181b24',
            borderRadius: 6,
            padding: '5px 8px',
            border: '1px solid #272c3b',
          }}
        >
          <Search size={13} color="#64748b" style={{ marginRight: 6 }} />
          <input
            type="text"
            placeholder="Search nodes..."
            value={paletteSearchQuery}
            onChange={(e) => setPaletteSearchQuery(e.target.value)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#e2e8f0',
              fontSize: 11,
              outline: 'none',
              width: '100%',
            }}
          />
        </div>
        <button
          onClick={togglePalette}
          title="Collapse Palette"
          style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', padding: 2 }}
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* Categories List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 0' }}>
        {grouped.map(({ category, items }) => {
          if (items.length === 0) return null;
          const isCollapsed = collapsedCategories[category];

          return (
            <div key={category} style={{ marginBottom: 4 }}>
              {/* Category Header */}
              <div
                onClick={() => toggleCategory(category)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '5px 12px',
                  cursor: 'pointer',
                  fontSize: 10,
                  fontWeight: 600,
                  color: '#94a3b8',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  background: '#151822',
                }}
              >
                <span>{category}</span>
                {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
              </div>

              {/* Category Items */}
              {!isCollapsed && (
                <div style={{ padding: '3px 6px' }}>
                  {items.map((defn) => {
                    const IconComponent = defn.icon ? ICON_MAP[defn.icon] || Box : Box;

                    return (
                      <div
                        key={defn.type_id}
                        draggable
                        onDragStart={(e) => handleDragStart(e, defn.type_id)}
                        onClick={() => handleQuickAdd(defn)}
                        title={defn.description}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '5px 8px',
                          margin: '2px 0',
                          borderRadius: 4,
                          background: '#181b26',
                          border: '1px solid #232838',
                          color: '#e2e8f0',
                          fontSize: 11,
                          cursor: 'grab',
                          transition: 'all 0.1s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <IconComponent size={13} className="text-slate-400" />
                          <span>{defn.display_name}</span>
                        </div>
                        <Plus size={11} color="#64748b" />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};
