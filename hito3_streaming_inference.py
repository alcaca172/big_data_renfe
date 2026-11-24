import os
import shutil
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    ArrayType,
)
from pyspark.ml import PipelineModel


# Esquema explícito obligatorio para Spark Structured Streaming sobre GTFS-RT
STREAM_SCHEMA = StructType(
    [
        StructField(
            "header",
            StructType(
                [
                    StructField("gtfsRealtimeVersion", StringType(), True),
                    StructField("timestamp", StringType(), True),
                ]
            ),
            True,
        ),
        StructField(
            "entity",
            ArrayType(
                StructType(
                    [
                        StructField("id", StringType(), True),
                        StructField(
                            "tripUpdate",
                            StructType(
                                [
                                    StructField(
                                        "trip",
                                        StructType(
                                            [
                                                StructField("tripId", StringType(), True),
                                                StructField("routeId", StringType(), True),
                                            ]
                                        ),
                                        True,
                                    ),
                                    StructField("delay", DoubleType(), True),
                                    StructField(
                                        "stopTimeUpdate",
                                        ArrayType(
                                            StructType(
                                                [
                                                    StructField("stopId", StringType(), True),
                                                    StructField(
                                                        "arrival",
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "delay", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "time", StringType(), True
                                                                ),
                                                            ]
                                                        ),
                                                        True,
                                                    ),
                                                ]
                                            )
                                        ),
                                        True,
                                    ),
                                ]
                            ),
                            True,
                        ),
                    ]
                )
            ),
            True,
        ),
    ]
)


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("Renfe_Hito3_Structured_Streaming")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def read_gtfs_clean(spark: SparkSession, path: str) -> DataFrame:
    """Lee un fichero GTFS estático eliminando los espacios de relleno de Renfe."""
    df = spark.read.csv(path, header=True, inferSchema=False)
    clean_cols = [c.strip() for c in df.columns]
    df = df.toDF(*clean_cols)
    return df.select([F.trim(F.col(c)).alias(c) for c in clean_cols])


def gtfs_time_to_seconds(col_name: str):
    parts = F.split(F.trim(F.col(col_name)), ":")
    return (
        parts.getItem(0).cast("int") * 3600
        + parts.getItem(1).cast("int") * 60
        + parts.getItem(2).cast("int")
    )


def load_static_reference_tables(
    spark: SparkSession,
    gtfs_dir: str = "data/processed/gtfs",
    weather_parquet: str = "data/processed/master_trains_weather.parquet",
):
    """Carga y cachea en memoria los maestros GTFS y el último contexto meteorológico."""
    stop_times_df = (
        read_gtfs_clean(spark, os.path.join(gtfs_dir, "stop_times.txt"))
        .select(
            "trip_id",
            "stop_id",
            F.col("stop_sequence").cast("int").alias("stop_sequence"),
            gtfs_time_to_seconds("arrival_time").alias("scheduled_arrival_sec"),
            gtfs_time_to_seconds("departure_time").alias("scheduled_departure_sec"),
        )
        .cache()
    )

    trips_df = (
        read_gtfs_clean(spark, os.path.join(gtfs_dir, "trips.txt"))
        .select("trip_id", "route_id")
        .cache()
    )

    stops_df = (
        read_gtfs_clean(spark, os.path.join(gtfs_dir, "stops.txt"))
        .select("stop_id", "stop_name")
        .cache()
    )

    routes_df = (
        read_gtfs_clean(spark, os.path.join(gtfs_dir, "routes.txt"))
        .select("route_id", "route_short_name")
        .cache()
    )

    if os.path.exists(weather_parquet):
        w_latest = Window.partitionBy("stop_id").orderBy(F.col("event_date").desc())
        weather_lookup_df = (
            spark.read.parquet(weather_parquet)
            .select(
                "stop_id",
                "event_date",
                "precip_mm",
                "temp_avg_c",
                "wind_avg_ms",
                "wind_gust_ms",
            )
            .withColumn("rn", F.row_number().over(w_latest))
            .filter(F.col("rn") == 1)
            .drop("rn", "event_date")
            .cache()
        )
    else:
        weather_lookup_df = spark.createDataFrame(
            [],
            "stop_id string, precip_mm double, temp_avg_c double, wind_avg_ms double, wind_gust_ms double",
        )

    return stop_times_df, trips_df, stops_df, routes_df, weather_lookup_df


def run_streaming_pipeline(
    stream_in_dir: str = "data/stream_in/trip_updates",
    stream_out_dir: str = "data/stream_out/predictions",
    checkpoint_dir: str = "data/stream_out/checkpoints",
    models_dir: str = "models",
    reset_state: bool = True,
) -> None:
    if reset_state:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(stream_out_dir, ignore_errors=True)

    os.makedirs(stream_in_dir, exist_ok=True)
    os.makedirs(stream_out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("[INFO] Cargando maestros GTFS y contexto meteorológico...")
    stop_times_df, trips_df, stops_df, routes_df, weather_lookup_df = load_static_reference_tables(spark)

    print("[INFO] Cargando modelos serializados de SparkML desde 'models/'...")
    reg_model = PipelineModel.load(os.path.join(models_dir, "delay_regressor_pipeline"))
    clf_model = PipelineModel.load(os.path.join(models_dir, "severity_classifier_pipeline"))

    # Estado en memoria con los últimos 30 min de eventos por línea para calcular congestión móvil
    corridor_history_pd = pd.DataFrame(
        columns=["route_id", "trip_id", "stop_id", "ts_epoch", "arrival_delay_seconds"]
    )

    # Lectura continua de ficheros JSON multilínea en la carpeta de entrada
    raw_stream = (
        spark.readStream.schema(STREAM_SCHEMA)
        .option("multiLine", "true")
        .option("maxFilesPerTrigger", 1)
        .json(Path(stream_in_dir).as_posix())
    )

    def process_microbatch(batch_df: DataFrame, batch_id: int) -> None:
        nonlocal corridor_history_pd

        if batch_df.isEmpty():
            return

        print(f"\n==================== MICRO-BATCH #{batch_id} ====================")

        # 1. Aplanar el mensaje GTFS-RT entrante
        exploded_entities = batch_df.select(
            F.to_timestamp(F.from_unixtime(F.col("header.timestamp").cast("long"))).alias("tu_ts"),
            F.explode("entity").alias("ent"),
        )

        flat_tu = (
            exploded_entities.select(
                "tu_ts",
                F.trim(F.col("ent.tripUpdate.trip.tripId")).alias("trip_id"),
                F.trim(F.col("ent.tripUpdate.trip.routeId")).alias("raw_route_id"),
                F.col("ent.tripUpdate.delay").cast("double").alias("trip_delay_sec"),
                F.explode("ent.tripUpdate.stopTimeUpdate").alias("stu"),
            )
            .select(
                "tu_ts",
                "trip_id",
                "raw_route_id",
                F.trim(F.col("stu.stopId")).alias("stop_id"),
                F.coalesce(
                    F.col("stu.arrival.delay").cast("double"),
                    F.col("trip_delay_sec"),
                ).alias("arrival_delay_seconds"),
            )
            .filter(
                F.col("trip_id").isNotNull()
                & F.col("stop_id").isNotNull()
                & F.col("arrival_delay_seconds").isNotNull()
            )
            .withColumn("ts_epoch", F.unix_timestamp("tu_ts"))
        )

        if flat_tu.isEmpty():
            print("[INFO] El archivo recibido no contiene actualizaciones de parada válidas.")
            return

        # 2. Enriquecer con horarios GTFS, estaciones, líneas y clima
        enriched = (
            flat_tu.join(stop_times_df, on=["trip_id", "stop_id"], how="left")
            .join(F.broadcast(trips_df), on="trip_id", how="left")
            .withColumn(
                "route_id",
                F.coalesce(F.col("route_id"), F.col("raw_route_id"), F.lit("DESCONOCIDA")),
            )
            .drop("raw_route_id")
            .join(F.broadcast(stops_df), on="stop_id", how="left")
            .join(F.broadcast(routes_df), on="route_id", how="left")
            .withColumn(
                "route_short_name",
                F.coalesce(
                    F.col("route_short_name"),
                    F.nullif(F.regexp_extract(F.col("trip_id"), r"^\d+M\d+(.+)$", 1), F.lit("")),
                    F.lit("DESCONOCIDA"),
                ),
            )
            .join(F.broadcast(weather_lookup_df), on="stop_id", how="left")
        )

        # 3. Calcular congestión móvil del corredor (últimos 30 min) combinando historial + lote actual
        curr_events_pd = enriched.select(
            "route_id", "trip_id", "stop_id", "ts_epoch", "arrival_delay_seconds"
        ).toPandas()

        if corridor_history_pd.empty:
            combined_pd = curr_events_pd.copy()
        else:
            combined_pd = pd.concat([corridor_history_pd, curr_events_pd], ignore_index=True)

        max_epoch = combined_pd["ts_epoch"].max()
        corridor_history_pd = combined_pd[combined_pd["ts_epoch"] >= max_epoch - 1800].drop_duplicates(
            subset=["route_id", "trip_id", "stop_id", "ts_epoch"], keep="last"
        )

        hist_spark = spark.createDataFrame(
            corridor_history_pd[["route_id", "trip_id", "stop_id", "ts_epoch", "arrival_delay_seconds"]]
        )
        w_corridor_30m = (
            Window.partitionBy("route_id").orderBy("ts_epoch").rangeBetween(-1800, -1)
        )
        rolling_stats = (
            hist_spark.withColumn(
                "route_rolling_delay_30m",
                F.coalesce(F.avg("arrival_delay_seconds").over(w_corridor_30m), F.lit(0.0)),
            )
            .withColumn(
                "route_rolling_trains_30m",
                F.count("trip_id").over(w_corridor_30m).cast("double"),
            )
            .select(
                "route_id",
                "trip_id",
                "stop_id",
                "ts_epoch",
                "route_rolling_delay_30m",
                "route_rolling_trains_30m",
            )
            .dropDuplicates(["route_id", "trip_id", "stop_id", "ts_epoch"])
        )

        # 4. Construir todas las variables que esperan los PipelineModel del Hito 2
        features_df = (
            enriched.join(
                rolling_stats, on=["route_id", "trip_id", "stop_id", "ts_epoch"], how="left"
            )
            .withColumn("hour_of_day", F.hour("tu_ts"))
            .withColumn("day_of_week", F.dayofweek("tu_ts"))
            .withColumn(
                "is_rush_hour",
                F.when(F.col("hour_of_day").isin([7, 8, 9, 18, 19, 20]), 1.0).otherwise(0.0),
            )
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin([1, 7]), 1.0).otherwise(0.0),
            )
            .withColumn(
                "scheduled_dwell_sec",
                F.greatest(
                    F.lit(0.0),
                    (F.col("scheduled_departure_sec") - F.col("scheduled_arrival_sec")).cast(
                        "double"
                    ),
                ),
            )
            # Columnas de telemetría/incidencias no presentes en el stream de trip_updates
            .withColumn("train_status", F.lit("SIN_TELEMETRIA"))
            .withColumn("dist_to_stop_km", F.lit(-1.0))
            .withColumn("active_incidents_count", F.lit(0))
            .withColumn("has_active_incident", F.lit(0))
            .fillna(
                {
                    "route_short_name": "DESCONOCIDA",
                    "stop_id": "DESCONOCIDA",
                    "stop_name": "Estación desconocida",
                    "stop_sequence": 1,
                    "scheduled_arrival_sec": 43200,
                    "scheduled_dwell_sec": 0.0,
                    "route_rolling_delay_30m": 0.0,
                    "route_rolling_trains_30m": 0.0,
                    "precip_mm": 0.0,
                    "temp_avg_c": 15.0,
                    "wind_avg_ms": 0.0,
                    "wind_gust_ms": 0.0,
                }
            )
        )

        # 5. Inferencia con ambos modelos SparkML (limpiando columnas intermedias entre pipelines)
        reg_out = reg_model.transform(features_df).withColumnRenamed(
            "prediction", "pred_delay_sec"
        )
        intermediate_cols = [
            c
            for c in reg_out.columns
            if c.endswith("_idx")
            or c.endswith("_vec")
            or c in ("features", "rawPrediction", "probability")
        ]
        reg_clean = reg_out.drop(*intermediate_cols)

        clf_out = clf_model.transform(reg_clean).withColumnRenamed(
            "prediction", "pred_severity_code"
        )

        final_preds = (
            clf_out.withColumn(
                "severity_alert",
                F.when(F.col("pred_severity_code") == 0.0, F.lit("VERDE (Puntual <=3m)"))
                .when(F.col("pred_severity_code") == 1.0, F.lit("AMARILLO (Moderado 3-10m)"))
                .otherwise(F.lit("ROJO (Grave >10m)")),
            )
            .select(
                "tu_ts",
                F.col("route_short_name").alias("linea"),
                "trip_id",
                "stop_name",
                F.round("arrival_delay_seconds", 1).alias("retraso_actual_s"),
                F.round("pred_delay_sec", 1).alias("pred_retraso_s"),
                "severity_alert",
                F.round("route_rolling_delay_30m", 1).alias("congestion_linea_30m_s"),
            )
        )

        total_paradas = final_preds.count()
        print(f"[OK] Predicciones generadas en tiempo real ({total_paradas} paradas):")
        final_preds.groupBy("severity_alert").count().orderBy("severity_alert").show(truncate=False)
        final_preds.show(15, truncate=False)

        # Persistir predicciones del micro-batch en Parquet
        final_preds.write.mode("append").parquet(stream_out_dir)

    query = (
        raw_stream.writeStream.foreachBatch(process_microbatch)
        .option("checkpointLocation", Path(checkpoint_dir).as_posix())
        .trigger(processingTime="5 seconds")
        .start()
    )

    print(f"[ESCUCHANDO] Esperando ficheros JSON en '{stream_in_dir}' (Ctrl+C para detener)...")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[INFO] Deteniendo Spark Structured Streaming...")
        query.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    run_streaming_pipeline()