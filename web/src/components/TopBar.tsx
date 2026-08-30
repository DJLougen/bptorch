/**
 * Top application toolbar containing project title, validation status,
 * compilation, execution controls, trace speed, parity status, and undo/redo.
 */
import React, { useEffect, useRef } from 'react';
import {
  AlertTriangle,
  Activity,
  CheckCircle2,
  Cpu,
  Download,
  Play,
  Redo2,
  Square,
  StepForward,
  Undo2,
  Upload,
  Zap,
} from 'lucide-react';
import { ProjectLoaderMenu } from './ProjectLoaderMenu';
import {
  parseProjectJson,
  serializeProject,
  useProjectStore,
} from '../stores/projectStore';
import type { TraceSpeed } from '../stores/uiStore';
import { useUIStore } from '../stores/uiStore';
import { useTraceStore } from '../stores/traceStore';
import { useValidationStore } from '../stores/validationStore';

const projectFileButtonStyle: React.CSSProperties = {
  background: '#181b24',
  border: '1px solid #272c3b',
  color: '#cbd5e1',
  padding: '4px 8px',
  borderRadius: 4,
  fontSize: 11,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
};

export const TopBar: React.FC = () => {
  const { project, isDirty, undo, redo, undoStack, redoStack, loadProject } = useProjectStore();
  const { traceSpeed, setTraceSpeed, openDrawerTab } = useUIStore();
  const { diagnostics, runValidation, isValidating, lastValidatedTimestamp } = useValidationStore();
  const { status, compileAndRun, step, continueRun, stop, addLog } = useTraceStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const timeout = setTimeout(() => {
      runValidation(project);
    }, 150);
    return () => clearTimeout(timeout);
  }, [project, runValidation]);

  const handleImportProject = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    try {
      const importedProject = parseProjectJson(await file.text());
      loadProject(importedProject);
      addLog('info', `Imported project "${importedProject.project.name}" from ${file.name}.`);
    } catch (error) {
      addLog(
        'error',
        `Project import rejected: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    } finally {
      input.value = '';
    }
  };

  const handleExportProject = () => {
    let downloadUrl: string | null = null;
    try {
      const filenameStem =
        project.project.name
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9_-]+/g, '-')
          .replace(/^-+|-+$/g, '') || 'project';
      const blob = new Blob([serializeProject(project)], {
        type: 'application/json',
      });
      downloadUrl = URL.createObjectURL(blob);

      const anchor = document.createElement('a');
      anchor.href = downloadUrl;
      anchor.download = `${filenameStem}.nbp.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      addLog('info', `Exported project as ${anchor.download}.`);
    } catch (error) {
      addLog(
        'error',
        `Project export failed: ${error instanceof Error ? error.message : String(error)}`
      );
    } finally {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
    }
  };

  const hasErrors = diagnostics.some((d) => d.severity === 'error');
  const hasValidated = !isValidating && lastValidatedTimestamp > 0;

  return (
    <header
      style={{
        height: 52,
        background: '#10121a',
        borderBottom: '1px solid #1f2430',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
        overflow: 'visible',
        whiteSpace: 'nowrap',
        flexShrink: 0,
        padding: '0 16px',
        color: '#e2e8f0',
        userSelect: 'none',
        zIndex: 100,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Zap size={18} color="#38bdf8" />
          <span style={{ fontWeight: 700, fontSize: 13, letterSpacing: '0.06em', color: '#60a5fa' }}>
            NEURAL BLUEPRINT
          </span>
          <span style={{ fontSize: 10, background: '#1e293b', color: '#94a3b8', padding: '1px 5px', borderRadius: 3 }}>
            v0.1
          </span>
        </div>

        <div style={{ width: 1, height: 20, background: '#1f2430' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, fontSize: 12 }}>
            {project.project.name}
            {isDirty && <span style={{ color: '#f59e0b' }}> *</span>}
          </span>
          <ProjectLoaderMenu />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            title="Undo (Ctrl+Z)"
            disabled={undoStack.length === 0}
            onClick={undo}
            style={{
              background: 'transparent',
              border: 'none',
              color: undoStack.length > 0 ? '#e2e8f0' : '#475569',
              cursor: undoStack.length > 0 ? 'pointer' : 'default',
              padding: 4,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Undo2 size={14} />
          </button>
          <button
            title="Redo (Ctrl+Shift+Z)"
            disabled={redoStack.length === 0}
            onClick={redo}
            style={{
              background: 'transparent',
              border: 'none',
              color: redoStack.length > 0 ? '#e2e8f0' : '#475569',
              cursor: redoStack.length > 0 ? 'pointer' : 'default',
              padding: 4,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Redo2 size={14} />
          </button>
        </div>

        <div aria-label="Project JSON controls" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.nbp.json,application/json"
            aria-label="Import project JSON file"
            onChange={handleImportProject}
            style={{ display: 'none' }}
          />
          <button
            type="button"
            title="Import project JSON"
            aria-label="Import project JSON"
            onClick={() => fileInputRef.current?.click()}
            style={projectFileButtonStyle}
          >
            <Upload size={12} />
            Import
          </button>
          <button
            type="button"
            title="Export project JSON"
            aria-label="Export project JSON"
            onClick={handleExportProject}
            style={projectFileButtonStyle}
          >
            <Download size={12} />
            Export
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {status === 'running' ? (
          <button
            onClick={stop}
            style={{
              background: '#ef4444',
              border: 'none',
              color: '#fff',
              padding: '6px 12px',
              borderRadius: 4,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <Square size={13} fill="#fff" />
            Stop
          </button>
        ) : status === 'paused' ? (
          <>
            <button
              onClick={continueRun}
              style={{
                background: '#22c55e',
                border: 'none',
                color: '#fff',
                padding: '6px 12px',
                borderRadius: 4,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <Play size={13} fill="#fff" />
              Continue
            </button>
            <button
              onClick={step}
              style={{
                background: '#38bdf8',
                border: 'none',
                color: '#0f172a',
                padding: '6px 12px',
                borderRadius: 4,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <StepForward size={13} />
              Step
            </button>
            <button
              onClick={stop}
              style={{
                background: '#334155',
                border: 'none',
                color: '#cbd5e1',
                padding: '6px 12px',
                borderRadius: 4,
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              Stop
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => compileAndRun(project, traceSpeed)}
              disabled={hasErrors}
              style={{
                background: hasErrors ? '#334155' : '#0284c7',
                border: 'none',
                color: hasErrors ? '#64748b' : '#fff',
                padding: '6px 14px',
                borderRadius: 4,
                fontSize: 12,
                fontWeight: 600,
                cursor: hasErrors ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                boxShadow: hasErrors ? 'none' : '0 0 12px rgba(2, 132, 199, 0.4)',
              }}
            >
              <Play size={13} fill={hasErrors ? '#64748b' : '#fff'} />
              Run Batch
            </button>
            <button
              onClick={() => compileAndRun(project, 'step')}
              disabled={hasErrors}
              style={{
                background: '#1e293b',
                border: '1px solid #334155',
                color: '#93c5fd',
                padding: '6px 10px',
                borderRadius: 4,
                fontSize: 12,
                cursor: hasErrors ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
              }}
            >
              <StepForward size={13} />
              Step Mode
            </button>
          </>
        )}

        <select
          value={traceSpeed}
          onChange={(e) => setTraceSpeed(e.target.value as TraceSpeed)}
          style={{
            background: '#181b24',
            border: '1px solid #272c3b',
            color: '#94a3b8',
            padding: '4px 8px',
            borderRadius: 4,
            fontSize: 11,
            outline: 'none',
            cursor: 'pointer',
          }}
        >
          <option value="instant">Speed: Instant</option>
          <option value="fast">Speed: Fast</option>
          <option value="normal">Speed: Normal</option>
          <option value="step">Speed: Step-by-Step</option>
        </select>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div
          onClick={() => openDrawerTab('diagnostics')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: hasErrors
              ? 'rgba(239, 68, 68, 0.15)'
              : hasValidated
                ? 'rgba(34, 197, 94, 0.15)'
                : 'rgba(100, 116, 139, 0.15)',
            border: `1px solid ${hasErrors ? '#ef4444' : hasValidated ? '#22c55e' : '#64748b'}`,
            color: hasErrors ? '#f87171' : hasValidated ? '#4ade80' : '#94a3b8',
            padding: '4px 8px',
            borderRadius: 4,
            fontSize: 11,
            cursor: 'pointer',
            fontWeight: 500,
          }}
        >
          {hasErrors ? (
            <AlertTriangle size={13} />
          ) : hasValidated ? (
            <CheckCircle2 size={13} />
          ) : (
            <Activity size={13} />
          )}
          <span>
            {hasErrors
              ? `${diagnostics.filter((d) => d.severity === 'error').length} Errors`
              : hasValidated
                ? 'Graph Valid'
                : isValidating
                  ? 'Validating…'
                  : 'Not validated'}
          </span>
        </div>

        <button
          type="button"
          onClick={() => openDrawerTab('parity')}
          title="Parity is verified only for the bundled nanoGPT baseline, not the current project."
          aria-label="Bundled nanoGPT baseline parity is verified"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid #38bdf8',
            color: '#38bdf8',
            padding: '4px 8px',
            borderRadius: 4,
            fontSize: 11,
            cursor: 'pointer',
            fontWeight: 500,
          }}
        >
          <Cpu size={13} />
          <span>Bundled nanoGPT baseline parity: Verified</span>
        </button>
      </div>
    </header>
  );
};
