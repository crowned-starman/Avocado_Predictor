import { BarChart3, TrendingUp, TrendingDown } from 'lucide-react';

interface YieldComparisonCardProps {
  historicalYieldTha: number;
  estimatedYieldTha: number;
}

export function YieldComparisonCard({
  historicalYieldTha,
  estimatedYieldTha,
}: YieldComparisonCardProps) {
  const diffTha = estimatedYieldTha - historicalYieldTha;
  const diffPct =
    historicalYieldTha > 0 ? ((diffTha / historicalYieldTha) * 100).toFixed(1) : '0.0';
  const isPositive = diffTha >= 0;

  return (
    <section className="card" aria-label="Comparativa de rendimiento">
      <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <BarChart3 size={20} color="#28572b" />
        Comparativa histórica vs. estimada (datos de prueba)
      </h2>

      <div className="comparison-grid">
        <div className="metric-box">
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Rendimiento histórico
          </div>
          <div className="metric-value">{historicalYieldTha.toFixed(1)} t/ha</div>
        </div>

        <div className="metric-box">
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Rendimiento estimado
          </div>
          <div className="metric-value">{estimatedYieldTha.toFixed(1)} t/ha</div>
        </div>

        <div className="metric-box">
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Variación proyectada
          </div>
          <div
            className="metric-value"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            {isPositive ? (
              <TrendingUp size={20} color="#28572b" />
            ) : (
              <TrendingDown size={20} color="#7a5808" />
            )}
            <span>
              {isPositive ? `+${diffPct}%` : `${diffPct}%`}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}