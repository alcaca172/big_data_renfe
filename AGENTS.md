# AGENTS.md — Guía Técnica y Contratos del Proyecto Big Data Renfe

Este documento define la arquitectura, los contratos de esquemas, las reglas de integridad de datos y los comandos de verificación del proyecto **Predicción de Retrasos e Incidencias en la Red Ferroviaria de Renfe**.

---

## 1. Arquitectura General del Sistema

El repositorio implementa un pipeline híbrido **Batch + Streaming** sobre **Apache Spark (PySpark 4.x / SparkML)**:

1. **Capa Batch y Feature Engineering (Hito 2):**
   - `hito2_build_dataset.py`: Ingesta capturas históricas JSON **GTFS-RT v2.0** (`trip_updates`, `position`, `incidents`), aplica desduplicación temporal por ventana, cruza maestros estáticos **GTFS** y calcula distancia geodésica previa (`dist_to_stop_km`, ventana causal de 15 min) e incidencias activas por línea. Genera `data/processed/master_trains.parquet`.
   - `hito2_enrich_aemet.py`: Vincula cada estación ferroviaria con su observatorio meteorológico más cercano de **AEMET OpenData** mediante distancia Haversine, normaliza coordenadas DMS y valores españoles, y genera `data/processed/master_trains_weather.parquet`.
   - `hito2_train_sparkml.py`: Entrena y evalúa modelos *baseline* (`RandomForest`, `GBTRegressor`, `LogisticRegression`) con corte temporal estricto 80/20.
   - `hito2_train_sparkml_tuned.py`: Entrena la iteración final con variables de congestión móvil de 30 minutos (`route_rolling_delay_30m`, `route_rolling_trains_30m`), balanceo por pesos inversos de clase (`weightCol="class_weight"`) y ajuste mediante `CrossValidator` (3 pliegues). Serializa los modelos ganadores en `models/`.

2. **Capa Tiempo Real / Structured Streaming (Hito 3):**
   - `hito3_live_producer.py`: Consulta cada **20 segundos** el feed público `https://gtfsrt.renfe.com/trip_updates.json`, filtra capturas duplicadas por `header.timestamp` y escribe cada JSON de forma atómica en `data/stream_in/trip_updates/`.
   - `hito3_stream_simulator.py`: Productor determinista offline que inyecta capturas históricas en `data/stream_in/trip_updates/` cada 6 segundos para demostraciones y pruebas sin conexión.
   - `hito3_streaming_inference.py`: Motor **Spark Structured Streaming** (`foreachBatch`, trigger de 5 s) que aplana mensajes GTFS-RT entrantes, mantiene estado móvil en memoria de los últimos 30 minutos por corredor, ejecuta inferencia encadenada con ambos `PipelineModel` de `models/` y persiste predicciones en Parquet (`data/stream_out/predictions/`).

3. **Calidad y CI/CD (Hito 3):**
   - `tests/test_pipeline.py`: Suite unitaria con `pytest` ejecutada sobre una sesión local de PySpark (`local[1]`).
   - `.github/workflows/ci.yml`: Workflow de GitHub Actions (Ubuntu + Java 17 Temurin + Python 3.12) que ejecuta `pytest -v` en cada `push` y `pull_request` sobre `main`.

---

## 2. Contratos de Datos e Invariantes Críticos

### A. Esquema de Entrada Streaming (`STREAM_SCHEMA`)
Todo fichero JSON depositado en `data/stream_in/trip_updates/` debe respetar el formato GTFS-RT v2.0 multilínea (`multiLine=true`):
- `header.timestamp` (string numérico Unix epoch en segundos).
- `entity[].tripUpdate.trip.tripId` y `routeId`.
- `entity[].tripUpdate.delay` (double opcional a nivel de viaje).
- `entity[].tripUpdate.stopTimeUpdate[].stopId` y `arrival.delay` (double en segundos).

### B. Contrato de Features de Entrada a los Modelos (`models/*_pipeline`)
Cualquier DataFrame pasado a `delay_regressor_pipeline` o `severity_classifier_pipeline` debe contener exactamente estas **19 columnas** sin valores nulos:
- **Categóricas (`string`):** `route_short_name`, `stop_id`, `train_status`.
- **Numéricas (`int`/`double`):** `stop_sequence`, `scheduled_arrival_sec`, `scheduled_dwell_sec`, `dist_to_stop_km`, `active_incidents_count`, `has_active_incident`, `hour_of_day`, `day_of_week`, `is_rush_hour`, `is_weekend`, `route_rolling_delay_30m`, `route_rolling_trains_30m`, `precip_mm`, `temp_avg_c`, `wind_avg_ms`, `wind_gust_ms`.

### C. Contrato de Salida en Tiempo Real (`data/stream_out/predictions/`)
Cada micro-batch escribe registros Parquet con las columnas:
`tu_ts`, `linea`, `trip_id`, `stop_name`, `retraso_actual_s`, `pred_retraso_s`, `severity_alert`, `congestion_linea_30m_s`.

---

## 3. Reglas de Desarrollo (Guardrails)

1. **Prevención de Fuga de Datos (*No Data Leakage*):**
   - Las divisiones *train/test* deben ser estrictamente cronológicas (`80%` inicial vs. `20%` final).
   - Todas las ventanas temporales deben ser causales: `pos_ts <= tu_ts` para telemetría GPS y `rangeBetween(-1800, -1)` para métricas móviles de congestión en el corredor.
2. **Aislamiento de Columnas entre Pipelines SparkML:**
   - Al encadenar `reg_model.transform()` y `clf_model.transform()` sobre el mismo DataFrame, es obligatorio eliminar previamente las columnas intermedias (`*_idx`, `*_vec`, `features`, `rawPrediction`, `probability`) para evitar colisiones de nombres de columna en `StringIndexer` y `OneHotEncoder`.
3. **Escritura Atómica en el Buzón de Streaming:**
   - Ningún productor debe escribir directamente sobre un archivo `.json` en `data/stream_in/trip_updates/`. Debe escribirse primero en `<nombre>.json.tmp` y renombrarse mediante `os.replace()` para evitar lecturas truncadas en `FileStreamSource`.
4. **Saneamiento de Ficheros GTFS de Renfe:**
   - Los CSV estáticos de Renfe incluyen espacios en blanco tanto en los nombres de cabecera como en los valores y horas superiores a `24:00:00` para servicios nocturnos. Usar siempre `read_gtfs_clean()` y `gtfs_time_to_seconds()`.
5. **Gestión de Artefactos Pesados y Rutas en Windows:**
   - `data/processed/gtfs/stop_times.txt` supera los 100 MB y está excluido en `.gitignore`.
   - En entornos Windows dentro de OneDrive, mantener `git config core.longpaths true` debido a la longitud de ruta de los metadatos internos de `RandomForest` en SparkML.

---

## 4. Comandos de Verificación

```bash
# Ejecutar suite de pruebas unitarias
pytest -v

# Reconstruir dataset maestro y modelos desde cero (Batch Hito 2)
python -u hito2_build_dataset.py
python -u hito2_enrich_aemet.py
python -u hito2_train_sparkml.py
python -u hito2_train_sparkml_tuned.py

# Ejecutar inferencia en tiempo real (Terminal 1) + Ingesta en vivo (Terminal 2)
python -u hito3_streaming_inference.py
python -u hito3_live_producer.py
```