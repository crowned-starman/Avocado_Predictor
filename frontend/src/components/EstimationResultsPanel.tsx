import { Leaf, Droplets, CloudRain, History, TrendingUp } from 'lucide-react';
import type { ConsultaParametros, ResultadoSimulacion } from '../data/mockSimulation';

interface EstimationResultsPanelProps {
  consulta: ConsultaParametros;
  resultado: ResultadoSimulacion | null;
  errorMsg: string | null;
  isLoading: boolean;
}

export function EstimationResultsPanel({
  consulta,
  resultado,
  errorMsg,
  isLoading,
}: EstimationResultsPanelProps) {
  const periodoTexto =
    consulta.modo === 'temporada' ? consulta.temporada : `Año ${consulta.anio}`;

  const diferenciaTon = resultado
    ? Number(
        (
          resultado.produccion_estimada_toneladas -
          resultado.historico_produccion_toneladas
        ).toFixed(1)
      )
    : 0;

  return (
    <section className="main-area" aria-label="Resultados de estimación">
      <h1 className="top-title">
        Estimación de Producción • {consulta.municipio}
      </h1>

      {isLoading && (
        <div className="card">
          <p className="info-text">
            Procesando consulta local simulada para {consulta.municipio} ({periodoTexto})...
          </p>
        </div>
      )}

      {!isLoading && errorMsg && (
        <div className="card">
          <p className="info-text" style={{ color: '#7a5808', fontWeight: 600 }}>
            {errorMsg}
          </p>
        </div>
      )}

      {!isLoading && !errorMsg && !resultado && (
        <div className="card">
          <p className="info-text">
            Selecciona un municipio, indica la superficie en hectáreas y elige consulta por año o por temporada; luego presiona <strong>Consultar Estimación</strong>.
          </p>
        </div>
      )}

      {!isLoading && !errorMsg && resultado && (
        <>
          <div className="card">
            <div className="prod-label">Producción Estimada</div>
            <div className="prod-big">
              {resultado.produccion_estimada_toneladas.toFixed(1)} Toneladas
            </div>
            <div className="prod-sub">
              {resultado.toneladas_por_hectarea.toFixed(1)} ton/ha • Rango estimado:{' '}
              {resultado.rango_estimado_toneladas[0].toFixed(1)} -{' '}
              {resultado.rango_estimado_toneladas[1].toFixed(1)} ton
            </div>

            <div className="conf-row">
              <span>Confidence bar ({resultado.confianza_pct}%)</span>
              <span>
                {resultado.rango_estimado_toneladas[0].toFixed(1)} -{' '}
                {resultado.rango_estimado_toneladas[1].toFixed(1)}
              </span>
            </div>
            <div
              className="conf-track"
              role="progressbar"
              aria-valuenow={resultado.confianza_pct}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="conf-fill"
                style={{ width: `${resultado.confianza_pct}%` }}
              />
            </div>
          </div>

          <div className="context-grid">
            <div className="context-card">
              <Leaf size={22} color="#3b8735" />
              <div className="context-title">Vigor de Vegetación:</div>
              <div className="context-val">
                {resultado.indicadores_contexto.vigor_vegetacion}
              </div>
            </div>

            <div className="context-card">
              <Droplets size={22} color="#2b82c9" />
              <div className="context-title">Humedad del Suelo:</div>
              <div className="context-val">
                {resultado.indicadores_contexto.humedad_suelo}
              </div>
            </div>

            <div className="context-card">
              <CloudRain size={22} color="#5e87b8" />
              <div className="context-title">Precipitación:</div>
              <div className="context-val">
                {resultado.indicadores_contexto.precipitacion_mm} mm
              </div>
            </div>
          </div>

          <div className="card">
            <div className="section-header">
              <div className="section-title">
                <History size={20} color="#2c6227" />
                <span>Histórico vs. Estimación actual</span>
              </div>
              <span className="diff-badge">
                <TrendingUp size={14} color="#2c6227" />
                {diferenciaTon >= 0 ? `+${diferenciaTon.toFixed(1)}` : diferenciaTon.toFixed(1)} Toneladas
              </span>
            </div>

            <div className="context-grid" style={{ marginTop: '12px' }}>
              <div className="context-card" style={{ background: '#f8faf7' }}>
                <div className="context-title" style={{ marginTop: 0 }}>
                  Referencia Histórica ({consulta.hectareas} ha)
                </div>
                <div className="context-val">
                  {resultado.historico_produccion_toneladas.toFixed(1)} Toneladas
                </div>
                <div className="prod-sub" style={{ marginTop: '4px', marginBottom: 0 }}>
                  {resultado.historico_toneladas_por_hectarea.toFixed(1)} ton/ha histórico
                </div>
              </div>

              <div className="context-card" style={{ background: '#f8faf7' }}>
                <div className="context-title" style={{ marginTop: 0 }}>
                  Estimación Actual ({consulta.hectareas} ha)
                </div>
                <div className="context-val" style={{ color: 'var(--avocado-dark)' }}>
                  {resultado.produccion_estimada_toneladas.toFixed(1)} Toneladas
                </div>
                <div className="prod-sub" style={{ marginTop: '4px', marginBottom: 0 }}>
                  {resultado.toneladas_por_hectarea.toFixed(1)} ton/ha estimado
                </div>
              </div>

              <div className="context-card" style={{ background: '#f8faf7' }}>
                <div className="context-title" style={{ marginTop: 0 }}>
                  Contexto de Consulta
                </div>
                <div className="context-val" style={{ fontSize: '16px' }}>
                  {periodoTexto}
                </div>
                <div className="prod-sub" style={{ marginTop: '4px', marginBottom: 0 }}>
                  NDVI {resultado.indicadores_contexto.ndvi_referencia.toFixed(2)} • NDMI{' '}
                  {resultado.indicadores_contexto.ndmi_referencia.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="meta-strip">
              <span>Municipio: <strong>{consulta.municipio}</strong></span>
              <span>Superficie: <strong>{consulta.hectareas} ha</strong></span>
              <span>Origen: <strong>{resultado.origen_dato}</strong></span>
            </div>
          </div>
        </>
      )}
    </section>
  );
}