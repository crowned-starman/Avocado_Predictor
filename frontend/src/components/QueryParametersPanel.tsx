import { Sprout } from 'lucide-react';
import {
  ANIOS_DISPONIBLES,
  MUNICIPIOS_DISPONIBLES,
  TEMPORADAS_DISPONIBLES,
  type ConsultaModo,
} from '../data/mockSimulation';

interface QueryParametersPanelProps {
  municipio: string;
  hectareas: number;
  modo: ConsultaModo;
  anio: number;
  temporada: string;
  isLoading: boolean;
  onMunicipioChange: (val: string) => void;
  onHectareasChange: (val: number) => void;
  onModoChange: (val: ConsultaModo) => void;
  onAnioChange: (val: number) => void;
  onTemporadaChange: (val: string) => void;
  onConsultar: () => void;
}

export function QueryParametersPanel({
  municipio,
  hectareas,
  modo,
  anio,
  temporada,
  isLoading,
  onMunicipioChange,
  onHectareasChange,
  onModoChange,
  onAnioChange,
  onTemporadaChange,
  onConsultar,
}: QueryParametersPanelProps) {
  return (
    <aside className="sidebar" aria-label="Parámetros de Consulta">
      <div className="brand-box">
        <Sprout size={24} color="#86d66d" />
        <div>
          <div className="brand-title">AVOCADO PREDICTOR</div>
          <div className="brand-sub">Analytics</div>
        </div>
      </div>

      <h2 className="sidebar-heading">Parámetros de Consulta</h2>

      <div className="field-group">
        <label htmlFor="select-municipio">Municipio</label>
        <select
          id="select-municipio"
          value={municipio}
          onChange={(e) => onMunicipioChange(e.target.value)}
        >
          {MUNICIPIOS_DISPONIBLES.map((m) => (
            <option key={m.id} value={m.nombre}>
              {m.nombre}
            </option>
          ))}
        </select>
      </div>

      <div className="field-group">
        <label htmlFor="input-hectareas">Superficie (Hectáreas)</label>
        <input
          id="input-hectareas"
          type="number"
          min={0.5}
          step={0.5}
          value={Number.isNaN(hectareas) ? '' : hectareas}
          onChange={(e) => onHectareasChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="field-group">
        <div className="mode-switch" role="group" aria-label="Tipo de predicción">
          <button
            type="button"
            className={`mode-btn ${modo === 'anio' ? 'active' : ''}`}
            onClick={() => onModoChange('anio')}
          >
            Por año
          </button>
          <button
            type="button"
            className={`mode-btn ${modo === 'temporada' ? 'active' : ''}`}
            onClick={() => onModoChange('temporada')}
          >
            Por temporada
          </button>
        </div>
      </div>

      {modo === 'temporada' ? (
        <div className="field-group">
          <label htmlFor="select-temporada">Temporada</label>
          <select
            id="select-temporada"
            value={temporada}
            onChange={(e) => onTemporadaChange(e.target.value)}
          >
            {TEMPORADAS_DISPONIBLES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="field-group">
          <label htmlFor="select-anio">Año</label>
          <select
            id="select-anio"
            value={anio}
            onChange={(e) => onAnioChange(Number(e.target.value))}
          >
            {ANIOS_DISPONIBLES.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
      )}

      <button
        type="button"
        className="submit-btn"
        disabled={isLoading}
        onClick={onConsultar}
      >
        {isLoading ? 'Consultando...' : 'Consultar Estimación'}
      </button>
    </aside>
  );
}