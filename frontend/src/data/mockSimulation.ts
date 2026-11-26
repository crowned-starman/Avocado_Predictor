export type ConsultaModo = 'anio' | 'temporada';

export interface ConsultaParametros {
  municipio: string;
  hectareas: number;
  modo: ConsultaModo;
  anio: number;
  temporada: string;
}

export interface IndicadoresContexto {
  vigor_vegetacion: string;
  humedad_suelo: string;
  precipitacion_mm: number;
  ndvi_referencia: number;
  ndmi_referencia: number;
}

export interface ResultadoSimulacion {
  produccion_estimada_toneladas: number;
  toneladas_por_hectarea: number;
  historico_produccion_toneladas: number;
  historico_toneladas_por_hectarea: number;
  rango_estimado_toneladas: [number, number];
  confianza_pct: number;
  indicadores_contexto: IndicadoresContexto;
  origen_dato: 'simulacion_propuesta_ui';
}

export interface MunicipioBase {
  id: string;
  nombre: string;
  cvegeo: string;
  baseTonHa: number;
  historicoTonHa: number;
  vigor: string;
  humedad: string;
  precipitacionBase: number;
  ndvi: number;
  ndmi: number;
  confianzaBase: number;
}

export const MUNICIPIOS_DISPONIBLES: MunicipioBase[] = [
  {
    id: 'uruapan',
    nombre: 'Uruapan',
    cvegeo: '16102',
    baseTonHa: 11.5,
    historicoTonHa: 10.9,
    vigor: 'Alto',
    humedad: 'Moderada-Alta',
    precipitacionBase: 395,
    ndvi: 0.76,
    ndmi: 0.39,
    confianzaBase: 88,
  },
  {
    id: 'tancitaro',
    nombre: 'Tancítaro',
    cvegeo: '16083',
    baseTonHa: 12.0,
    historicoTonHa: 11.4,
    vigor: 'Muy Alto',
    humedad: 'Alta',
    precipitacionBase: 420,
    ndvi: 0.79,
    ndmi: 0.42,
    confianzaBase: 90,
  },
  {
    id: 'ario',
    nombre: 'Ario',
    cvegeo: '16009',
    baseTonHa: 9.8,
    historicoTonHa: 9.6,
    vigor: 'Moderado-Alto',
    humedad: 'Moderada',
    precipitacionBase: 310,
    ndvi: 0.69,
    ndmi: 0.31,
    confianzaBase: 82,
  },
  {
    id: 'periban',
    nombre: 'Peribán',
    cvegeo: '16069',
    baseTonHa: 11.2,
    historicoTonHa: 10.8,
    vigor: 'Alto',
    humedad: 'Moderada-Alta',
    precipitacionBase: 375,
    ndvi: 0.75,
    ndmi: 0.37,
    confianzaBase: 86,
  },
];

export const ANIOS_DISPONIBLES = [2026, 2025, 2024];

export const TEMPORADAS_DISPONIBLES = [
  'Temporada 1 (predefinida)',
  'Temporada 2 (predefinida)',
  'Temporada Floración-Cuajado',
];

export function simularEstimacion(params: ConsultaParametros): ResultadoSimulacion {
  const mun =
    MUNICIPIOS_DISPONIBLES.find((m) => m.nombre === params.municipio) ??
    MUNICIPIOS_DISPONIBLES[0];

  const factorPeriodo =
    params.modo === 'temporada'
      ? params.temporada.includes('Temporada 1')
        ? 1.0
        : params.temporada.includes('Temporada 2')
          ? 0.94
          : 1.04
      : params.anio === 2026
        ? 1.02
        : params.anio === 2025
          ? 0.99
          : 0.96;

  const tonHa = Number((mun.baseTonHa * factorPeriodo).toFixed(1));
  const produccionTotal = Number((tonHa * params.hectareas).toFixed(1));

  const histTonHa = Number(mun.historicoTonHa.toFixed(1));
  const histTotal = Number((histTonHa * params.hectareas).toFixed(1));

  const rangoMin = Number((produccionTotal * 0.887).toFixed(1));
  const rangoMax = Number((produccionTotal * 1.113).toFixed(1));
  const precipitacion = Math.round(mun.precipitacionBase * factorPeriodo);

  return {
    produccion_estimada_toneladas: produccionTotal,
    toneladas_por_hectarea: tonHa,
    historico_produccion_toneladas: histTotal,
    historico_toneladas_por_hectarea: histTonHa,
    rango_estimado_toneladas: [rangoMin, rangoMax],
    confianza_pct: mun.confianzaBase,
    indicadores_contexto: {
      vigor_vegetacion: mun.vigor,
      humedad_suelo: mun.humedad,
      precipitacion_mm: precipitacion,
      ndvi_referencia: mun.ndvi,
      ndmi_referencia: mun.ndmi,
    },
    origen_dato: 'simulacion_propuesta_ui',
  };
}