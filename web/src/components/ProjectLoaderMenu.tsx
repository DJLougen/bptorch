/**
 * Hierarchical blueprint loader with portaled menus (never clipped by toolbar overflow).
 */

import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, ChevronRight, Layers } from 'lucide-react';
import {
  fetchProjectFromPath,
  fetchSamplesManifest,
  groupSamplesByCategory,
  type SampleManifestEntry,
} from '../lib/samplesManifest';
import { useProjectStore } from '../stores/projectStore';
import { useTraceStore } from '../stores/traceStore';

const STARTER_TEMPLATES = [
  {
    id: 'nanogpt',
    label: 'nanoGPT Architecture',
    path: '/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json',
  },
  {
    id: 'mlp',
    label: 'Two-Layer MLP',
    path: '/examples/arch_4_twolayer_mlp/arch_4_twolayer_mlp.nbp.json',
  },
] as const;

const menuPanelStyle: React.CSSProperties = {
  position: 'fixed',
  minWidth: 240,
  background: '#10121a',
  border: '1px solid #272c3b',
  borderRadius: 8,
  boxShadow: '0 12px 32px rgba(0,0,0,0.45)',
  padding: '6px 0',
  zIndex: 5000,
};

const menuItemStyle: React.CSSProperties = {
  width: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
  background: 'transparent',
  border: 'none',
  color: '#e2e8f0',
  fontSize: 12,
  textAlign: 'left',
  padding: '8px 12px',
  cursor: 'pointer',
};

const categoryHeaderStyle: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: '#64748b',
  padding: '8px 12px 4px',
};

interface AnchorRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export const ProjectLoaderMenu: React.FC = () => {
  const { loadProject } = useProjectStore();
  const { addLog } = useTraceStore();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const samplesItemRef = useRef<HTMLButtonElement>(null);
  const samplesPinnedRef = useRef(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [samplesOpen, setSamplesOpen] = useState(false);
  const [samples, setSamples] = useState<SampleManifestEntry[]>([]);
  const [manifestError, setManifestError] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<AnchorRect | null>(null);
  const [samplesAnchor, setSamplesAnchor] = useState<AnchorRect | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSamplesManifest()
      .then((manifest) => {
        if (!cancelled) {
          setSamples(manifest.samples ?? []);
          setManifestError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setManifestError(error instanceof Error ? error.message : String(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const updateAnchors = () => {
    const buttonRect = buttonRef.current?.getBoundingClientRect();
    if (buttonRect) {
      setMenuAnchor({
        top: buttonRect.bottom + 4,
        left: buttonRect.left,
        width: buttonRect.width,
        height: buttonRect.height,
      });
    }
    const samplesRect = samplesItemRef.current?.getBoundingClientRect();
    if (samplesRect) {
      setSamplesAnchor({
        top: samplesRect.top,
        left: samplesRect.right,
        width: samplesRect.width,
        height: samplesRect.height,
      });
    }
  };

  useLayoutEffect(() => {
    if (!menuOpen) {
      setMenuAnchor(null);
      setSamplesAnchor(null);
      return;
    }
    updateAnchors();
    const onLayout = () => updateAnchors();
    window.addEventListener('resize', onLayout);
    window.addEventListener('scroll', onLayout, true);
    return () => {
      window.removeEventListener('resize', onLayout);
      window.removeEventListener('scroll', onLayout, true);
    };
  }, [menuOpen, samplesOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (buttonRef.current?.contains(target)) return;
      if ((event.target as HTMLElement | null)?.closest('[data-template-menu-root="true"]')) return;
      samplesPinnedRef.current = false;
      setMenuOpen(false);
      setSamplesOpen(false);
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        samplesPinnedRef.current = false;
        setMenuOpen(false);
        setSamplesOpen(false);
      }
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [menuOpen]);

  const groupedSamples = useMemo(() => groupSamplesByCategory(samples), [samples]);

  const closeMenus = () => {
    samplesPinnedRef.current = false;
    setMenuOpen(false);
    setSamplesOpen(false);
  };

  const closeSamplesSubmenu = () => {
    samplesPinnedRef.current = false;
    setSamplesOpen(false);
  };

  const openSamplesSubmenu = () => {
    setSamplesOpen(true);
    requestAnimationFrame(updateAnchors);
  };

  const toggleSamplesSubmenu = () => {
    setSamplesOpen((open) => {
      const next = !open;
      samplesPinnedRef.current = next;
      requestAnimationFrame(updateAnchors);
      return next;
    });
  };

  const loadBlueprint = async (label: string, path: string) => {
    setLoadingPath(path);
    try {
      const project = await fetchProjectFromPath(path);
      loadProject(project);
      addLog('info', `Loaded blueprint: ${label}`);
      closeMenus();
    } catch (error) {
      addLog(
        'error',
        `Failed to load ${label}: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      setLoadingPath(null);
    }
  };

  const menuPortal =
    menuOpen && menuAnchor
      ? createPortal(
          <div
            data-template-menu-root="true"
            role="menu"
            aria-label="Blueprint templates"
            style={{
              ...menuPanelStyle,
              top: menuAnchor.top,
              left: menuAnchor.left,
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: '#64748b',
                padding: '6px 12px 4px',
              }}
            >
              Starter Blueprints
            </div>
            {STARTER_TEMPLATES.map((template) => (
              <button
                key={template.id}
                type="button"
                role="menuitem"
                disabled={loadingPath === template.path}
                onClick={() => void loadBlueprint(template.label, template.path)}
                style={{
                  ...menuItemStyle,
                  opacity: loadingPath === template.path ? 0.6 : 1,
                }}
              >
                <span>{template.label}</span>
              </button>
            ))}

            <div style={{ height: 1, background: '#1f2430', margin: '6px 0' }} />

            <button
              ref={samplesItemRef}
              type="button"
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={samplesOpen}
              onMouseEnter={openSamplesSubmenu}
              onClick={toggleSamplesSubmenu}
              style={{
                ...menuItemStyle,
                background: samplesOpen ? '#1e293b' : 'transparent',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Layers size={13} color="#38bdf8" />
                <span>
                  Architecture Samples
                  {samples.length > 0 ? ` (${samples.length})` : ''}
                </span>
              </span>
              <ChevronRight size={13} color="#94a3b8" />
            </button>
          </div>,
          document.body,
        )
      : null;

  const samplesPortal =
    menuOpen && samplesOpen && samplesAnchor
      ? createPortal(
          <div
            data-template-menu-root="true"
            role="menu"
            aria-label="Architecture samples"
            style={{
              ...menuPanelStyle,
              top: samplesAnchor.top,
              left: samplesAnchor.left + 4,
              minWidth: 300,
              maxHeight: 'min(420px, calc(100vh - 16px))',
              overflowY: 'auto',
            }}
            onMouseEnter={openSamplesSubmenu}
            onMouseLeave={() => {
              if (!samplesPinnedRef.current) {
                closeSamplesSubmenu();
              }
            }}
          >
            {manifestError ? (
              <div style={{ padding: '8px 12px', color: '#f87171', fontSize: 11 }}>{manifestError}</div>
            ) : groupedSamples.length === 0 ? (
              <div style={{ padding: '8px 12px', color: '#94a3b8', fontSize: 11 }}>Loading samples…</div>
            ) : (
              groupedSamples.map(({ category, samples: categorySamples }) => (
                <div key={category}>
                  <div style={categoryHeaderStyle}>{category}</div>
                  {categorySamples.map((sample) => (
                    <button
                      key={sample.id}
                      type="button"
                      role="menuitem"
                      title={sample.description}
                      disabled={loadingPath === sample.path}
                      onClick={() => void loadBlueprint(sample.name, sample.path)}
                      style={{
                        ...menuItemStyle,
                        paddingLeft: 16,
                        opacity: loadingPath === sample.path ? 0.6 : 1,
                      }}
                    >
                      <span>{sample.name}</span>
                    </button>
                  ))}
                </div>
              ))
            )}
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        aria-label="Load blueprint template"
        onClick={() => {
          setMenuOpen((open) => {
            if (open) {
              closeSamplesSubmenu();
            }
            return !open;
          });
        }}
        style={{
          background: '#181b24',
          border: '1px solid #272c3b',
          color: '#cbd5e1',
          padding: '4px 10px',
          borderRadius: 4,
          fontSize: 11,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <Layers size={13} />
        <span>Templates</span>
        <ChevronDown size={12} />
      </button>
      {menuPortal}
      {samplesPortal}
    </>
  );
};
