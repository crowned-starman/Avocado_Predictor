import { AlertTriangle, Info } from 'lucide-react';

interface NoticesCardProps {
  coverageNotice: string;
  qualityNotice: string;
}

export function NoticesCard({ coverageNotice, qualityNotice }: NoticesCardProps) {
  return (
    <section className="card" aria-label="Avisos de cobertura y calidad">
      <h2 className="card-title">Avisos de cobertura y calidad (datos de ejemplo)</h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div
          className="notice-banner"
          style={{
            backgroundColor: 'var(--avocado-soft)',
            borderColor: '#cde2ce',
            color: 'var(--avocado-dark)',
          }}
        >
          <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>{coverageNotice}</span>
        </div>

        <div className="notice-banner">
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>{qualityNotice}</span>
        </div>
      </div>
    </section>
  );
}