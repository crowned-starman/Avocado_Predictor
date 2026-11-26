import { CalendarRange } from 'lucide-react';
import type { TemporalBlock } from '../data/orchards';

interface TemporalBlocksCardProps {
  blocks: TemporalBlock[];
}

export function TemporalBlocksCard({ blocks }: TemporalBlocksCardProps) {
  return (
    <section className="card" aria-label="Bloques temporales">
      <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <CalendarRange size={20} color="#28572b" />
        Bloques temporales (B1–B4 · ejemplo)
      </h2>

      <div className="blocks-grid">
        {blocks.map((block) => (
          <div key={block.code} className="block-item">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <strong>{block.label}</strong>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--avocado-dark)' }}>
                {block.window}
              </span>
            </div>
            <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)' }}>
              NDVI ejemplo: <strong>{block.ndviIndex.toFixed(2)}</strong>
            </div>
            <div style={{ fontSize: '0.85rem', marginTop: '4px', color: 'var(--text-main)' }}>
              Estado: {block.status}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}