# Predicción de Retrasos e Incidencias en la Red Ferroviaria de Renfe (Big Data)

Sistema distribuido *end-to-end* desarrollado con **Apache Spark (PySpark, SparkML y Structured Streaming)** para la ingesta, enriquecimiento meteorológico, entrenamiento de modelos predictivos e inferencia continua en tiempo real sobre la red de Cercanías Renfe.

El sistema combina telemetría en tiempo real (**GTFS-RT v2.0**), planificación ferroviaria estática (**GTFS**) y observaciones climatológicas de **AEMET OpenData** para resolver simultáneamente dos tareas operativas:
1. **Regresión (`arrival_delay_seconds`):** Predicción continua del retraso de llegada en segundos.
2. **Clasificación Multiclase (`severity_label`):** Semáforo operativo del estado del servicio:
   - **Clase `0` — VERDE (`Puntual <=3m`):** $\le 180\text{ s}$
   - **Clase `1` — AMARILLO (`Moderado 3-10m`):** $181\text{ s} - 600\text{ s}$
   - **Clase `2` — ROJO (`Grave >10m`):** $> 600\text{ s}$

---

## Arquitectura Completa del Proyecto

```text
================================================================================
                        FASE BATCH Y MODELADO (HITOS 1 - 2)
================================================================================
[GTFS-RT Histórico]           [GTFS Estático]             [API AEMET OpenData]
(trip_updates, position,   (stop_times, trips,         (Inventario estaciones +
 incidents)                 stops, routes)              observaciones diarias)
         │                         │                              │
         └────────────┬────────────┘                              │
                      ▼                                           │
         1. hito2_build_dataset.py                                │
            • Aplanado JSON (explode) + Desduplicación            │
            • Distancia Haversine GPS + Alertas por línea         │
                      │                                           │
                      ▼                                           │
       data/processed/master_trains.parquet                       │
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                               2. hito2_enrich_aemet.py
                                  • Conversión DMS -> Decimal
                                  • Observatorio AEMET más cercano
                                            │
                                            ▼
                         data/processed/master_trains_weather.parquet
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        3a. hito2_train_sparkml.py                  3b. hito2_train_sparkml_tuned.py
            (Modelos Baseline)                          (Congestión móvil 30m +
                                                         Class Weights + 3-Fold CV)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                         models/delay_regressor_pipeline
                         models/severity_classifier_pipeline
                         models/model_metadata.json

================================================================================
                    FASE STREAMING EN TIEMPO REAL (HITO 3)
================================================================================
[Renfe OpenData API (cada 20s)]           [Histórico Local (Simulación 6s)]
    hito3_live_producer.py                    hito3_stream_simulator.py
              │                                           │
              └─────────────────────┬─────────────────────┘
                        (Escritura atómica .tmp -> .json)
                                    ▼
                     data/stream_in/trip_updates/
                                    │
                                    ▼
                     hito3_streaming_inference.py
                     (Spark Structured Streaming + foreachBatch)
                       • Estado móvil 30m en corredor por línea
                       • Extracción de línea y cruce GTFS/AEMET
                       • Inferencia dual (Regresor + Clasificador)
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
         [Alertas en Consola en Vivo]   [data/stream_out/predictions/*.parquet]
```

---

## Estructura del Repositorio

```text
big_data_renfe/
├── .github/
│   └── workflows/
│       └── ci.yml                           # Integración continua con GitHub Actions
├── data/
│   ├── raw/
│   │   ├── trip_updates/                    # Capturas JSON GTFS-RT de retrasos por parada
│   │   ├── position/                        # Capturas JSON GTFS-RT de posición GPS
│   │   ├── incidents/                       # Capturas JSON GTFS-RT de alertas activas
│   │   └── aemet/                           # Caché local de estaciones y clima AEMET
│   └── processed/
│       ├── gtfs/                            # Tablas estáticas GTFS (trips, stops, routes...)
│       ├── master_trains.parquet/           # Dataset ferroviario consolidado (Hito 2)
│       └── master_trains_weather.parquet/   # Dataset enriquecido con clima (Hito 2)
├── models/
│   ├── delay_regressor_pipeline/            # PipelineModel ganador de regresión
│   ├── severity_classifier_pipeline/        # PipelineModel ganador de clasificación
│   ├── delay_regressor_tuned_pipeline/      # Artefacto serializado afinado
│   ├── severity_classifier_tuned_pipeline/  # Artefacto serializado afinado
│   └── model_metadata.json                  # Registro de métricas y parámetros
├── tests/
│   └── test_pipeline.py                     # Pruebas unitarias PySpark (pytest)
├── hito2_build_dataset.py                   # ETL batch: unión GTFS-RT + GTFS estático
├── hito2_enrich_aemet.py                    # Enriquecimiento geoespacial con AEMET OpenData
├── hito2_train_sparkml.py                   # Entrenamiento y evaluación Baseline (v1)
├── hito2_train_sparkml_tuned.py             # Entrenamiento afinado con congestión móvil (v2)
├── hito3_streaming_inference.py             # Motor de inferencia Spark Structured Streaming
├── hito3_live_producer.py                   # Productor en vivo contra API Renfe (cada 20 s)
├── hito3_stream_simulator.py                # Simulador offline de flujo GTFS-RT
├── requirements.txt                         # Dependencias Python del proyecto
├── AGENTS.md                                # Contratos técnicos y guardrails del sistema
└── README.md                                # Documentación general
```

---

## Resultados del Modelado Predictivo (Hito 2)

Sobre el conjunto de prueba temporal estrictamente posterior al entrenamiento (`20%` final), la segunda iteración con variables de congestión móvil (`route_rolling_delay_30m`), balanceo de clases y validación cruzada mejora todas las métricas respecto al modelo base:

| Tarea | Algoritmo Ganador | Métrica | Baseline (v1) | Afinado (v2) | Mejora |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Regresión** | `RandomForestRegressor` | **$R^2$** | `0.7985` | **`0.8958`** | **`+0.0973`** |
| **Regresión** | `RandomForestRegressor` | **MAE** | `507.61 s` (8.46 min) | **`388.71 s` (6.48 min)** | **`-23.4%`** |
| **Regresión** | `RandomForestRegressor` | **RMSE** | `773.80 s` (12.90 min) | **`599.18 s` (9.99 min)** | **`-22.6%`** |
| **Clasificación** | `RandomForestClassifier` | **F1-Score (ponderado)** | `0.7166` | **`0.7773`** | **`+0.0607`** |
| **Clasificación** | `RandomForestClassifier` | **Accuracy global** | `72.36%` | **`77.44%`** | **`+5.08` pts** |

- **Propagación de retrasos:** `route_rolling_delay_30m` y `route_rolling_trains_30m` se posicionan en el Top 5 de variables más influyentes, capturando el efecto dominó en cada corredor ferroviario.
- **Detección de incidencias graves:** El clasificador identifica correctamente el **88.4% de los retrasos graves (`ROJO`)** y eleva la sensibilidad en retrasos moderados (`AMARILLO`) del **45.8% al 70.4%**.

---

## Pipeline en Tiempo Real y Calidad (Hito 3)

### 1. Inferencia Continua (`hito3_streaming_inference.py`)
- **Ingesta desacoplada:** Vigila `data/stream_in/trip_updates/` mediante `readStream.json` con esquema explícito `STREAM_SCHEMA`.
- **Estado móvil en vivo:** Calcula en cada micro-batch el retraso medio acumulado de los últimos 30 minutos por línea (`congestion_linea_30m_s`) combinando el historial reciente con el paquete entrante.
- **Baja latencia operativa:** Procesa ~260–430 paradas por captura en ~6–7 segundos en entorno local, operando holgadamente dentro de la ventana de actualización de **20 segundos** del feed oficial de Renfe.

### 2. Pruebas Unitarias e Integración Continua (`tests/test_pipeline.py` & CI)
La suite `pytest` valida automáticamente en cada `push` mediante **GitHub Actions**:
- Conversión de horas GTFS estándar y servicios nocturnos posteriores a medianoche (`>24:00:00`).
- Limpieza de espacios de relleno en cabeceras y valores de los ficheros estáticos de Renfe.
- Cálculo geodésico de distancias mediante la fórmula de Haversine.
- Umbrales de clasificación operativa (`VERDE`, `AMARILLO`, `ROJO`).
- Contrato de parseo JSON GTFS-RT y esquema de salida del streaming.

---

## Guía de Ejecución

### 1. Instalación de dependencias
Requiere **Python 3.10+** y **Java JDK 17**.
```bash
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline Batch (Hito 2)
```powershell
# 1. Consolidar dataset maestro ferroviario
python -u hito2_build_dataset.py

# 2. Enriquecer con observaciones meteorológicas de AEMET
$env:AEMET_API_KEY="TU_CLAVE_AEMET"
python -u hito2_enrich_aemet.py

# 3. Entrenar modelos Baseline y Afinados
python -u hito2_train_sparkml.py
python -u hito2_train_sparkml_tuned.py
```

### 3. Ejecución en Tiempo Real (Hito 3)
Abrir dos terminales en la raíz del proyecto:

- **Terminal 1 — Motor Spark Structured Streaming:**
  ```powershell
  python -u hito3_streaming_inference.py
  ```
- **Terminal 2 — Ingesta en vivo desde Renfe OpenData (cada 20 s):**
  ```powershell
  python -u hito3_live_producer.py
  ```
  *(Alternativa sin conexión utilizando capturas históricas cada 6 s: `python -u hito3_stream_simulator.py`)*.

### 4. Ejecución de Pruebas Unitarias
```bash
pytest -v
```