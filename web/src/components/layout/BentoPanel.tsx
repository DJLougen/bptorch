/**
 * Rounded bento tile wrapper for a layout region.
 */

import React from 'react';

interface BentoPanelProps {
  title?: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}

export const BentoPanel: React.FC<BentoPanelProps> = ({ title, children, style }) => (
  <section
    style={{
      display: 'flex',
      flexDirection: 'column',
      minWidth: 0,
      minHeight: 0,
      background: '#10121a',
      border: '1px solid #232838',
      borderRadius: 12,
      boxShadow: '0 8px 24px rgba(0,0,0,0.25)',
      overflow: 'hidden',
      ...style,
    }}
  >
    {title ? (
      <header
        style={{
          padding: '6px 10px',
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#64748b',
          borderBottom: '1px solid #1f2430',
          flexShrink: 0,
        }}
      >
        {title}
      </header>
    ) : null}
    <div style={{ flex: 1, minHeight: 0, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
      {children}
    </div>
  </section>
);
