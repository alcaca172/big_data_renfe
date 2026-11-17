# Predicción de Retrasos e Incidencias en la Red Ferroviaria de Renfe (Big Data)

Pipeline de ingeniería de datos y aprendizaje automático distribuido desarrollado con **Apache Spark (PySpark & SparkML)** para la monitorización, enriquecimiento meteorológico y predicción de retrasos en trenes de Cercanías Renfe.

El proyecto integra telemetría en tiempo real en formato **GTFS-RT v2.0**, planificación estática **GTFS** y observaciones climatológicas de **AEMET OpenData** para resolver dos tareas predictivas:
1. **Regresión (`arrival_delay_seconds`):** Estimación continua del retraso exacto de llegada en segundos.
2. **Clasificación Multiclase (`severity_label`):** Categorización operativa del estado del servicio en tres niveles de gravedad:
   - **Clase `0` — Puntual / Leve:** $\le 3\text{ minutos}$ ($\le 180\text{ s}$)
   - **Clase `1` — Retraso Moderado:** entre $3\text{ y }10\text{ minutos}$ ($181\text{ s} - 600\text{ s}$)
   - **Clase `2` — Incidencia / Retraso Grave:** $> 10\text{ minutos}$ ($> 600\text{ s}$)

---

## Arquitectura del Proyecto (Hitos 1 y 2)

```text
[GTFS-RT JSON Feeds]          [GTFS Estático]             [API AEMET OpenData]
(trip_updates, position,   (stop_times, trips,         (Inventario estaciones +
 incidents)                 stops, routes)              observaciones diarias)
         │                         │                              │
         └────────────┬────────────┘                              │
                      ▼                                           │
         1. hito2_build_dataset.py                                │
            • Aplanado JSON (explode)                             │
            • Desduplicación por ventana                          │
            • Distancia Haversine GPS                             │
            • Agregación de alertas                               │
                      │                                           │
                      ▼                                           │
       data/processed/master_trains.parquet                       │
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                               2. hito2_enrich_aemet.py
                                  • Conversión DMS -> Decimal
                                  • Estación AEMET más cercana
                                            │
                                            ▼
                         data/processed/master_trains_weather.parquet
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        3a. hito2_train_sparkml.py                  3b. hito2_train_sparkml_tuned.py
            (Modelos Baseline)                          (Feature Eng. + Class Weights
                                                         + 3-Fold CrossValidator)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                         models/delay_regressor_pipeline
                         models/severity_classifier_pipeline
                         models/model_metadata.json
```

---

## Estructura del Repositorio

```text
proyecto_renfe/
├── data/
│   ├── raw/
│   │   ├── trip_updates/          # Capturas JSON GTFS-RT de retrasos por parada
│   │   ├── position/              # Capturas JSON GTFS-RT de posición GPS y estado
│   │   ├── incidents/             # Capturas JSON GTFS-RT de alertas activas
│   │   └── aemet/                 # Caché local de estaciones y clima de AEMET
│   └── processed/
│       ├── gtfs/                  # Tablas estáticas GTFS (stop_times, trips, stops, routes)
│       ├── master_trains.parquet/ # Dataset consolidado ferroviario
│       └── master_trains_weather.parquet/ # Dataset final enriquecido con meteorología
├── models/
│   ├── delay_regressor_pipeline/          # PipelineModel ganador de regresión
│   ├── severity_classifier_pipeline/      # PipelineModel ganador de clasificación
│   ├── delay_regressor_tuned_pipeline/    # Artefacto de regresión afinado
│   ├── severity_classifier_tuned_pipeline/# Artefacto de clasificación afinado
│   └── model_metadata.json                # Registro de métricas y clases
├── hito2_build_dataset.py         # ETL batch: unión GTFS-RT + GTFS estático
├── hito2_enrich_aemet.py          # Enriquecimiento geoespacial con AEMET OpenData
├── hito2_train_sparkml.py         # Entrenamiento y evaluación Baseline en SparkML
├── hito2_train_sparkml_tuned.py   # Iteración v2: ingeniería de dominio, pesos y CV
└── README.md
```

---

## Descripción de los Módulos (Hito 2)

### 1. Consolidación del Dataset Maestro (`build_dataset_hito2.py`)
- **Lectura y desempaquetado GTFS-RT:** Procesa ficheros JSON multilínea (`multiLine=true`), desplegando las listas anidadas de entidades (`entity`), paradas (`stopTimeUpdate`) y líneas afectadas (`informedEntity`).
- **Desduplicación temporal:** Mediante funciones de ventana (`Window.partitionBy("trip_id", "stop_id", "event_date").orderBy(F.col("tu_ts").desc())`), conserva la última actualización consolidada de cada tren en cada estación.
- **Integración con maestros GTFS:** Limpia los espacios de relleno de los ficheros oficiales de Renfe, filtra `stop_times.txt` mediante `left_semi` join para reducir memoria y cruza `trips.txt`, `stops.txt` y `routes.txt` mediante *broadcast joins*.
- **Cálculo geoespacial causal:** Asocia cada parada con la última posición GPS registrada del tren en los **15 minutos anteriores** (`pos_ts <= tu_ts`) y calcula la distancia en kilómetros a la estación mediante la fórmula de **Haversine** (`dist_to_stop_km`), además de contabilizar las incidencias activas por línea y hora (`active_incidents_count`).

### 2. Enriquecimiento Meteorológico (`enrich_aemet_hito2.py`)
- **Integración con AEMET OpenData:** Gestiona el protocolo de doble petición de la API oficial de AEMET y normaliza tanto coordenadas sexagesimales en formato texto (`DMS`, ej. `402443N`) a grados decimales como valores numéricos en formato español (comas decimales y registros inapreciables `Ip`).
- **Vinculación por proximidad:** Empareja cada estación de Renfe con su observatorio meteorológico más cercano mediante distancia Haversine, incorporando precipitación diaria (`precip_mm`), temperatura media (`temp_avg_c`), viento medio (`wind_avg_ms`) y racha máxima (`wind_gust_ms`).

### 3. Modelado Predictivo en SparkML (`hito2_train_sparkml.py` y `hito2_train_sparkml_tuned.py`)
- **Particionado temporal estricto:** Divide los datos respetando el orden cronológico (`80%` inicial para entrenamiento y `20%` posterior para evaluación) para evitar fugas de información entre trenes consecutivos.
- **Ingeniería de características (v2):** Añade identificadores de estación (`stop_id`), indicadores de hora punta (`is_rush_hour`), fin de semana (`is_weekend`), tiempo de parada programado (`scheduled_dwell_sec`) y métricas de **congestión móvil en el corredor** durante los 30 minutos estrictamente anteriores (`route_rolling_delay_30m` y `route_rolling_trains_30m`, usando `rangeBetween(-1800, -1)`).
- **Balanceo y optimización:** Aplica pesos inversos a la frecuencia de clase (`weightCol`) para corregir el desbalanceo de la clase intermedia y ajusta hiperparámetros mediante validación cruzada de 3 pliegues (`CrossValidator`).

---

## Resultados Experimentales

### Comparativa: Baseline (v1) vs. Modelo Afinado (v2)

| Tarea | Algoritmo Ganador | Métrica | Baseline (v1) | Afinado (v2) | Mejora |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Regresión** | `RandomForestRegressor` | **$R^2$** | `0.7985` | **`0.8958`** | **`+0.0973`** |
| **Regresión** | `RandomForestRegressor` | **MAE** | `507.61 s` (8.46 min) | **`388.71 s` (6.48 min)** | **`-23.4%`** |
| **Regresión** | `RandomForestRegressor` | **RMSE** | `773.80 s` (12.90 min) | **`599.18 s` (9.99 min)** | **`-22.6%`** |
| **Clasificación** | `RandomForestClassifier` | **F1-Score (ponderado)** | `0.7166` | **`0.7773`** | **`+0.0607`** |
| **Clasificación** | `RandomForestClassifier` | **Accuracy global** | `72.36%` | **`77.44%`** | **`+5.08` pts** |

### Principales Hallazgos
- **Impacto de la congestión del corredor:** Las variables móviles `route_rolling_delay_30m` y `route_rolling_trains_30m` se sitúan entre las cinco más influyentes del modelo, confirmando el efecto propagación de los retrasos a lo largo de una misma línea.
- **Aporte meteorológico:** La temperatura media (`temp_avg_c`), la precipitación (`precip_mm`) y la velocidad del viento (`wind_avg_ms`) entran en el Top 8 de importancia del clasificador de gravedad.
- **Sensibilidad en alertas graves:** El clasificador afinado detecta el **88.4% de los retrasos graves** (`Clase 2`, 61 de 69 casos en test) y eleva el acierto en retrasos moderados (`Clase 1`) del **45.8% al 70.4%** tras aplicar balanceo de clases.

---

## Requisitos y Ejecución

### Requisitos previos
- **Python 3.10+** y **Java JDK 17** (requerido por Apache Spark).
- **Apache Spark / PySpark 4.x** (`pip install pyspark`).
- Clave gratuita de la API de **AEMET OpenData**.

### Pasos de ejecución

1. **Construir el dataset maestro ferroviario:**
   ```bash
   python -u build_dataset_hito2.py
   ```

2. **Enriquecer con datos meteorológicos de AEMET:**
   ```powershell
   # En Windows PowerShell:
   $env:AEMET_API_KEY="TU_CLAVE_DE_AEMET"
   python -u enrich_aemet_hito2.py
   ```

3. **Entrenar modelos Baseline y Afinados:**
   ```bash
   python -u hito2_train_sparkml.py
   python -u hito2_train_sparkml_tuned.py
   ```

---

## Próximos Pasos (Hito 3)
- Despliegue del pipeline de inferencia en tiempo real mediante **Spark Structured Streaming** consumiendo los flujos JSON en vivo y cargando los artefactos `PipelineModel` almacenados en `models/`.
- Incorporación de pruebas unitarias, flujo de integración continua (CI) y documentación técnica del proyecto.