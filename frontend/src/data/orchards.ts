export interface TemporalBlock {
  code: 'B1' | 'B2' | 'B3' | 'B4';
  label: string;
  window: string;
  ndviIndex: number;
  status: string;
}

export interface OrchardRecord {
  id: string;
  name: string;
  municipality: string;
  areaHa: number;
  historicalYieldTha: number;
  estimatedYieldTha: number;
  confidencePct: number;
  coverageNotice: string;
  qualityNotice: string;
  blocks: TemporalBlock[];
}

export const ORCHARDS: OrchardRecord[] = [
  {
    id: 'huerta-norte-01',
    name: 'Huerta El Mirador',
    municipality: 'Uruapan (Ejemplo)',
    areaHa: 18.5,
    historicalYieldTha: 11.4,
    estimatedYieldTha: 12.8,
    confidencePct: 86,
    coverageNotice: 'Cobertura satelital válida en el 94 % de la superficie de prueba.',
    qualityNotice: 'Calidad de observación moderada-alta; sin nubes críticas en los últimos cortes.',
    blocks: [
      { code: 'B1', label: 'Bloque B1', window: 'Q1', ndviIndex: 0.72, status: 'Estable' },
      { code: 'B2', label: 'Bloque B2', window: 'Q2', ndviIndex: 0.76, status: 'Vigoroso' },
      { code: 'B3', label: 'Bloque B3', window: 'Q3', ndviIndex: 0.74, status: 'Estable' },
      { code: 'B4', label: 'Bloque B4', window: 'Q4', ndviIndex: 0.79, status: 'Óptimo' },
    ],
  },
  {
    id: 'huerta-sur-02',
    name: 'Parcela Loma Verde',
    municipality: 'Tancítaro (Ejemplo)',
    areaHa: 12.0,
    historicalYieldTha: 10.9,
    estimatedYieldTha: 10.2,
    confidencePct: 71,
    coverageNotice: 'Cobertura parcial de prueba (78 %); verificar linderos en la esquina sur.',
    qualityNotice: 'Aviso de calidad: presencia moderada de nubosidad en uno de los periodos.',
    blocks: [
      { code: 'B1', label: 'Bloque B1', window: 'Q1', ndviIndex: 0.68, status: 'Moderado' },
      { code: 'B2', label: 'Bloque B2', window: 'Q2', ndviIndex: 0.65, status: 'Atención' },
      { code: 'B3', label: 'Bloque B3', window: 'Q3', ndviIndex: 0.70, status: 'Estable' },
      { code: 'B4', label: 'Bloque B4', window: 'Q4', ndviIndex: 0.71, status: 'Estable' },
    ],
  },
];