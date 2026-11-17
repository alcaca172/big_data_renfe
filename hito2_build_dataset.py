import glob
import os
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from pyspark.sql.window import Window


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("Renfe_Hito2_Build_Master_Dataset")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def find_gtfs_dir(base_dir: str = ".") -> str:
    """Localiza automáticamente la carpeta GTFS (data/gtfs o gtfs)."""
    candidates = [
        os.path.join(base_dir, "data", "gtfs"),
        os.path.join(base_dir, "gtfs"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    for root, _, files in os.walk(base_dir):
        if "stop_times.txt" in files or "trips.txt" in files:
            return root
    raise FileNotFoundError("No se encontró la carpeta GTFS ('data/gtfs' o 'gtfs').")


def find_json_files(folder: str) -> list[str]:
    """Busca archivos .json y .jsonl en la carpeta y normaliza rutas para Spark en Windows."""
    files = glob.glob(os.path.join(folder, "*.json")) + glob.glob(os.path.join(folder, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .json ni .jsonl en: {folder}")
    return [Path(f).as_posix() for f in sorted(files)]


def read_gtfs_csv(spark: SparkSession, path: str) -> DataFrame | None:
    """Lee un fichero GTFS eliminando los espacios de relleno en cabeceras y valores."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        df = spark.read.csv(path, header=True, inferSchema=False)
        clean_cols = [c.strip() for c in df.columns]
        df = df.toDF(*clean_cols)
        return df.select([F.trim(F.col(c)).alias(c) for c in clean_cols])
    return None


def has_field(schema: StructType, path: str) -> bool:
    """Comprueba de forma segura si una ruta anidada existe en el esquema."""
    curr = schema
    for part in path.split("."):
        if not isinstance(curr, StructType):
            return False
        fields = {f.name: f.dataType for f in curr.fields}
        if part not in fields:
            return False
        curr = fields[part]
    return True


def safe_col(df: DataFrame, path: str, cast_type: str | None = None):
    """Devuelve la columna si existe en el esquema o NULL tipado si es opcional y no viene."""
    if has_field(df.schema, path):
        col = F.col(path)
        return col.cast(cast_type) if cast_type else col
    return F.lit(None).cast(cast_type) if cast_type else F.lit(None)


def unix_str_to_ts(col_expr):
    """Convierte timestamps Unix en formato texto (ej. '1794333320') a Timestamp real."""
    return F.to_timestamp(F.from_unixtime(col_expr.cast("long")))


def gtfs_time_to_seconds(col_name: str):
    """Convierte horas GTFS (incluyendo >24:00:00) a segundos desde medianoche."""
    parts = F.split(F.trim(F.col(col_name)), ":")
    return (
        parts.getItem(0).cast("int") * 3600
        + parts.getItem(1).cast("int") * 60
        + parts.getItem(2).cast("int")
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula la distancia en kilómetros entre dos coordenadas GPS en Spark SQL."""
    r = 6371.0
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (
        F.sin(dlat / 2) ** 2
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    )
    return 2 * r * F.asin(F.sqrt(a))


def build_master_dataset(
    spark: SparkSession,
    gtfs_dir: str | None = None,
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/master_trains.parquet",
) -> None:
    if gtfs_dir is None:
        gtfs_dir = find_gtfs_dir()
    print(f"[INFO] Usando carpeta GTFS: {gtfs_dir}")

    # =========================================================================
    # 1. APLANAR Y DESDUPLICAR TRIP UPDATES (Retrasos por tren y parada)
    # =========================================================================
    tu_files = find_json_files(os.path.join(raw_dir, "trip_updates"))
    tu_raw = spark.read.option("multiLine", "true").json(tu_files)

    tu_entities = tu_raw.select(
        unix_str_to_ts(F.col("header.timestamp")).alias("tu_ts"),
        F.explode("entity").alias("ent"),
    )

    tu_exploded = tu_entities.select(
        "tu_ts",
        safe_col(tu_entities, "ent.tripUpdate.trip.tripId", "string").alias("trip_id"),
        safe_col(tu_entities, "ent.tripUpdate.delay", "double").alias("trip_delay_sec"),
        F.explode("ent.tripUpdate.stopTimeUpdate").alias("stu"),
    )

    tu_flat = (
        tu_exploded.select(
            "tu_ts",
            F.trim(F.col("trip_id")).alias("trip_id"),
            F.trim(safe_col(tu_exploded, "stu.stopId", "string")).alias("stop_id"),
            F.coalesce(
                safe_col(tu_exploded, "stu.arrival.delay", "double"),
                safe_col(tu_exploded, "stu.departure.delay", "double"),
                F.col("trip_delay_sec"),
            ).alias("arrival_delay_seconds"),
        )
        .filter(
            F.col("trip_id").isNotNull()
            & F.col("stop_id").isNotNull()
            & F.col("arrival_delay_seconds").isNotNull()
        )
        .withColumn("event_date", F.to_date("tu_ts"))
    )

    w_latest = Window.partitionBy("trip_id", "stop_id", "event_date").orderBy(F.col("tu_ts").desc())
    tu_dedup = (
        tu_flat.withColumn("rn", F.row_number().over(w_latest))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # =========================================================================
    # 2. CRUZAR CON LOS MAESTROS ESTÁTICOS DE GTFS (Horarios y estaciones)
    # =========================================================================
    base_df = tu_dedup
    observed_trips = tu_dedup.select("trip_id").distinct()

    stop_times_raw = read_gtfs_csv(spark, os.path.join(gtfs_dir, "stop_times.txt"))
    if stop_times_raw is not None:
        stop_times_df = (
            stop_times_raw.join(observed_trips, on="trip_id", how="left_semi")
            .select(
                "trip_id",
                "stop_id",
                safe_col(stop_times_raw, "stop_sequence", "int").alias("stop_sequence"),
                gtfs_time_to_seconds("arrival_time").alias("scheduled_arrival_sec"),
                gtfs_time_to_seconds("departure_time").alias("scheduled_departure_sec"),
            )
        )
        base_df = base_df.join(stop_times_df, on=["trip_id", "stop_id"], how="left")

    trips_raw = read_gtfs_csv(spark, os.path.join(gtfs_dir, "trips.txt"))
    if trips_raw is not None:
        trips_df = trips_raw.select(
            safe_col(trips_raw, "trip_id", "string").alias("trip_id"),
            safe_col(trips_raw, "route_id", "string").alias("route_id"),
            safe_col(trips_raw, "service_id", "string").alias("service_id"),
            safe_col(trips_raw, "shape_id", "string").alias("shape_id"),
        )
        base_df = base_df.join(F.broadcast(trips_df), on="trip_id", how="left")
    else:
        base_df = base_df.withColumn("route_id", F.lit(None).cast("string"))

    stops_raw = read_gtfs_csv(spark, os.path.join(gtfs_dir, "stops.txt"))
    if stops_raw is not None:
        stops_df = stops_raw.select(
            safe_col(stops_raw, "stop_id", "string").alias("stop_id"),
            safe_col(stops_raw, "stop_name", "string").alias("stop_name"),
            safe_col(stops_raw, "stop_lat", "double").alias("stop_lat"),
            safe_col(stops_raw, "stop_lon", "double").alias("stop_lon"),
        )
        base_df = base_df.join(F.broadcast(stops_df), on="stop_id", how="left")
    else:
        base_df = base_df.withColumn("stop_lat", F.lit(None).cast("double")).withColumn(
            "stop_lon", F.lit(None).cast("double")
        )

    routes_raw = read_gtfs_csv(spark, os.path.join(gtfs_dir, "routes.txt"))
    if routes_raw is not None and "route_id" in base_df.columns:
        routes_df = routes_raw.select(
            safe_col(routes_raw, "route_id", "string").alias("route_id"),
            safe_col(routes_raw, "route_short_name", "string").alias("route_short_name"),
            safe_col(routes_raw, "route_long_name", "string").alias("route_long_name"),
        )
        base_df = base_df.join(F.broadcast(routes_df), on="route_id", how="left")

    # =========================================================================
    # 3. APLANAR POSITION Y CALCULAR DISTANCIA DEL TREN A LA ESTACIÓN
    # =========================================================================
    pos_files = find_json_files(os.path.join(raw_dir, "positions"))
    pos_raw = spark.read.option("multiLine", "true").json(pos_files)

    pos_entities = pos_raw.select(
        F.col("header.timestamp").alias("hdr_ts"),
        F.explode("entity").alias("ent"),
    )

    pos_flat = (
        pos_entities.select(
            unix_str_to_ts(
                F.coalesce(
                    safe_col(pos_entities, "ent.vehicle.timestamp", "string"),
                    F.col("hdr_ts"),
                )
            ).alias("pos_ts"),
            F.trim(safe_col(pos_entities, "ent.vehicle.trip.tripId", "string")).alias("pos_trip_id"),
            safe_col(pos_entities, "ent.vehicle.position.latitude", "double").alias("train_lat"),
            safe_col(pos_entities, "ent.vehicle.position.longitude", "double").alias("train_lon"),
            safe_col(pos_entities, "ent.vehicle.currentStatus", "string").alias("train_status"),
        )
        .filter(F.col("pos_trip_id").isNotNull() & F.col("pos_ts").isNotNull())
    )

    joined_pos = base_df.join(
        pos_flat,
        (base_df["trip_id"] == pos_flat["pos_trip_id"])
        & (pos_flat["pos_ts"] <= base_df["tu_ts"])
        & (pos_flat["pos_ts"] >= base_df["tu_ts"] - F.expr("INTERVAL 15 MINUTES")),
        how="left",
    ).drop("pos_trip_id")

    w_nearest_pos = Window.partitionBy("trip_id", "stop_id", "event_date").orderBy(
        F.col("pos_ts").desc_nulls_last()
    )
    master_df = (
        joined_pos.withColumn("rn_pos", F.row_number().over(w_nearest_pos))
        .filter(F.col("rn_pos") == 1)
        .withColumn(
            "dist_to_stop_km",
            haversine_km(
                F.col("train_lat"), F.col("train_lon"), F.col("stop_lat"), F.col("stop_lon")
            ),
        )
        .drop("rn_pos", "pos_ts")
    )

    # =========================================================================
    # 4. APLANAR INCIDENTS (Alertas activas por línea y hora)
    # =========================================================================
    inc_files = find_json_files(os.path.join(raw_dir, "incidents"))
    inc_raw = spark.read.option("multiLine", "true").json(inc_files)

    inc_entities = inc_raw.select(
        unix_str_to_ts(F.col("header.timestamp")).alias("inc_ts"),
        F.explode("entity").alias("ent"),
    )

    inc_exploded = inc_entities.select(
        "inc_ts",
        safe_col(inc_entities, "ent.id", "string").alias("alert_id"),
        F.explode("ent.alert.informedEntity").alias("inf"),
    )

    inc_flat = (
        inc_exploded.select(
            "inc_ts",
            "alert_id",
            F.trim(safe_col(inc_exploded, "inf.routeId", "string")).alias("route_id"),
        )
        .filter(F.col("route_id").isNotNull() & F.col("inc_ts").isNotNull())
    )

    inc_agg = (
        inc_flat.withColumn("incident_hour", F.date_trunc("hour", F.col("inc_ts")))
        .groupBy("route_id", "incident_hour")
        .agg(F.countDistinct("alert_id").alias("active_incidents_count"))
    )

    master_df = (
        master_df.withColumn("incident_hour", F.date_trunc("hour", F.col("tu_ts")))
        .join(F.broadcast(inc_agg), on=["route_id", "incident_hour"], how="left")
        .drop("incident_hour")
        .fillna({"active_incidents_count": 0})
        .withColumn("has_active_incident", (F.col("active_incidents_count") > 0).cast("int"))
    )

    # =========================================================================
    # 5. VARIABLES TEMPORALES Y GUARDADO EN PARQUET
    # =========================================================================
    master_df = (
        master_df.withColumn("hour_of_day", F.hour("tu_ts"))
        .withColumn("day_of_week", F.dayofweek("tu_ts"))
    )

    master_df.write.mode("overwrite").partitionBy("event_date").parquet(output_path)

    res = spark.read.parquet(output_path)
    print(f"[OK] Dataset guardado en '{output_path}' con {res.count():,} registros.")
    res.show(5, truncate=False)


if __name__ == "__main__":
    spark = create_spark_session()
    try:
        build_master_dataset(spark)
    finally:
        spark.stop()