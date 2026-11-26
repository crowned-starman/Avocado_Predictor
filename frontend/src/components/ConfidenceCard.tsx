import { ShieldCheck } from 'lucide-react';

interface ConfidenceCardProps {
  confidencePct: number;
  orchardName: string;
}

export function ConfidenceCard({ confidencePct, orchardName }: ConfidenceCardProps) {
  const clampedPct = Math.max(0, Math.min(100, confidencePct));

  return (
    <section className="card" aria-label="Nivel de confianza">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
        <h2 className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldCheck size={20} color="#28572b" />
          Confianza de la estimación — {orchardName}
        </h2>
        <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--avocado-dark)' }}>
          {clampedPct}%
        </span>
      </div>

      <div className="confidence-track" role="progressbar" aria-valuenow={clampedPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="confidence-fill" style={{ width: `${clampedPct}%` }} />
      </div>

      <p style={{ marginTop: '10px', fontSize: '0.88rem', color: 'var(--text-muted)' }}>
        Valor de ejemplo calculado para la vista previa local sin conexión externa ni envío de datos.
      </p>
    </section>
  );
}