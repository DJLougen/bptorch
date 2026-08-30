/**
 * Sample Gallery — browse and load 25 architecture demonstrations.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { BookOpen, Layers, Sparkles, X } from 'lucide-react';
import {
  fetchProjectFromPath,
  fetchSamplesManifest,
  type SampleManifestEntry,
  type SamplesManifest,
} from '../lib/samplesManifest';
import { useProjectStore } from '../stores/projectStore';
import { useTraceStore } from '../stores/traceStore';

export type { SampleManifestEntry };

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#22c55e',
  intermediate: '#38bdf8',
  advanced: '#f59e0b',
};

interface SamplesGalleryProps {
  open: boolean;
  onClose: () => void;
}

export const SamplesGallery: React.FC<SamplesGalleryProps> = ({ open, onClose }) => {
  const { loadProject } = useProjectStore();
  const { addLog } = useTraceStore();
  const [manifest, setManifest] = useState<SamplesManifest | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    fetchSamplesManifest()
      .then((data: SamplesManifest) => setManifest(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [open]);

  const categories = useMemo(() => {
    if (!manifest) return ['All'];
    const cats = [...new Set(manifest.samples.map((s) => s.category))].sort();
    return ['All', ...cats];
  }, [manifest]);

  const filtered = useMemo(() => {
    if (!manifest) return [];
    if (activeCategory === 'All') return manifest.samples;
    return manifest.samples.filter((s) => s.category === activeCategory);
  }, [manifest, activeCategory]);

  const loadSample = async (sample: SampleManifestEntry) => {
    setLoadingId(sample.id);
    try {
      const data = await fetchProjectFromPath(sample.path);
      loadProject(data);
      addLog('info', `Loaded sample: ${sample.name}`);
      onClose();
    } catch (err) {
      addLog('error', `Failed to load ${sample.name}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingId(null);
    }
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-label="Sample Gallery"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.65)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#10121a',
          border: '1px solid #272c3b',
          borderRadius: 12,
          width: 'min(960px, 100%)',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #1f2430', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <BookOpen size={18} color="#38bdf8" />
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Architecture Sample Gallery</h2>
              {manifest && (
                <span style={{ fontSize: 11, color: '#64748b' }}>{manifest.count} samples</span>
              )}
            </div>
            <p style={{ margin: 0, fontSize: 12, color: '#94a3b8', maxWidth: 640 }}>
              {manifest?.description || 'Explore 25 trainable blueprints — transformers, MLPs, training pipelines, attention mechanics, and more.'}
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '10px 20px', borderBottom: '1px solid #1f2430', display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              style={{
                background: activeCategory === cat ? '#1e3a5f' : '#181b24',
                border: `1px solid ${activeCategory === cat ? '#38bdf8' : '#272c3b'}`,
                color: activeCategory === cat ? '#38bdf8' : '#94a3b8',
                padding: '4px 10px',
                borderRadius: 16,
                fontSize: 11,
                cursor: 'pointer',
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
          {error && (
            <div style={{ gridColumn: '1 / -1', color: '#ef4444', fontSize: 12 }}>
              Failed to load gallery manifest: {error}. Run <code>make examples</code> to generate samples.
            </div>
          )}
          {!manifest && !error && (
            <div style={{ gridColumn: '1 / -1', color: '#64748b', fontSize: 12 }}>Loading samples...</div>
          )}
          {filtered.map((sample) => (
            <button
              key={sample.id}
              onClick={() => loadSample(sample)}
              disabled={loadingId === sample.id}
              style={{
                textAlign: 'left',
                background: '#181b24',
                border: '1px solid #272c3b',
                borderRadius: 8,
                padding: 14,
                cursor: loadingId === sample.id ? 'wait' : 'pointer',
                color: '#e2e8f0',
                opacity: loadingId && loadingId !== sample.id ? 0.6 : 1,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 12 }}>{sample.name}</span>
                <span
                  style={{
                    fontSize: 9,
                    textTransform: 'uppercase',
                    color: DIFFICULTY_COLORS[sample.difficulty] || '#94a3b8',
                    border: `1px solid ${DIFFICULTY_COLORS[sample.difficulty] || '#94a3b8'}`,
                    padding: '1px 5px',
                    borderRadius: 3,
                  }}
                >
                  {sample.difficulty}
                </span>
              </div>
              <p style={{ margin: '0 0 8px', fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>{sample.description}</p>
              {sample.highlight && (
                <div style={{ display: 'flex', gap: 4, alignItems: 'flex-start', marginBottom: 8 }}>
                  <Sparkles size={11} color="#f59e0b" style={{ flexShrink: 0, marginTop: 1 }} />
                  <span style={{ fontSize: 10, color: '#fbbf24', lineHeight: 1.3 }}>{sample.highlight}</span>
                </div>
              )}
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                <Layers size={10} color="#64748b" />
                {sample.node_count != null && (
                  <span style={{ fontSize: 10, color: '#64748b' }}>{sample.node_count} nodes</span>
                )}
                {sample.tags.slice(0, 3).map((tag) => (
                  <span key={tag} style={{ fontSize: 9, background: '#1e293b', color: '#64748b', padding: '1px 5px', borderRadius: 3 }}>
                    {tag}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
