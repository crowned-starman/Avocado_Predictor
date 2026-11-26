import { useState } from 'react';
import {
  ANIOS_DISPONIBLES,
  MUNICIPIOS_DISPONIBLES,
  TEMPORADAS_DISPONIBLES,
  simularEstimacion,
  type ConsultaModo,
  type ConsultaParametros,
  type ResultadoSimulacion,
} from './data/mockSimulation';
import { QueryParametersPanel } from './components/QueryParametersPanel';
import { EstimationResultsPanel } from './components/EstimationResultsPanel';

export function App() {
  const [municipio, setMunicipio] = useState<string>(MUNICIPIOS_DISPONIBLES[0].nombre);
  const [hectareas, setHectareas] = useState<number>(10);
  const [modo, setModo] = useState<ConsultaModo>('temporada');
  const [anio, setAnio] = useState<number>(ANIOS_DISPONIBLES[0]);
  const [temporada, setTemporada] = useState<string>(TEMPORADAS_DISPONIBLES[0]);

  const [consultaActiva, setConsultaActiva] = useState<ConsultaParametros>({
    municipio: MUNICIPIOS_DISPONIBLES[0].nombre,
    hectareas: 10,
    modo: 'temporada',
    anio: ANIOS_DISPONIBLES[0],
    temporada: TEMPORADAS_DISPONIBLES[0],
  });

  const [resultado, setResultado] = useState<ResultadoSimulacion | null>(() =>
    simularEstimacion({
      municipio: MUNICIPIOS_DISPONIBLES[0].nombre,
      hectareas: 10,
      modo: 'temporada',
      anio: ANIOS_DISPONIBLES[0],
      temporada: TEMPORADAS_DISPONIBLES[0],
    })
  );

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  function handleConsultar() {
    const params: ConsultaParametros = {
      municipio,
      hectareas,
      modo,
      anio,
      temporada,
    };

    setConsultaActiva(params);

    if (Number.isNaN(hectareas) || hectareas <= 0) {
      setResultado(null);
      setErrorMsg('Ingresa una superficie válida mayor a 0 hectáreas para consultar.');
      return;
    }

    setErrorMsg(null);
    setIsLoading(true);

    window.setTimeout(() => {
      const simulado = simularEstimacion(params);
      setResultado(simulado);
      setIsLoading(false);
    }, 250);
  }

  return (
    <div className="layout">
      <QueryParametersPanel
        municipio={municipio}
        hectareas={hectareas}
        modo={modo}
        anio={anio}
        temporada={temporada}
        isLoading={isLoading}
        onMunicipioChange={setMunicipio}
        onHectareasChange={setHectareas}
        onModoChange={setModo}
        onAnioChange={setAnio}
        onTemporadaChange={setTemporada}
        onConsultar={handleConsultar}
      />

      <EstimationResultsPanel
        consulta={consultaActiva}
        resultado={resultado}
        errorMsg={errorMsg}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;