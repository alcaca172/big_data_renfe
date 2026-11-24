import json
import math
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from hito3_streaming_inference import (
    STREAM_SCHEMA,
    gtfs_time_to_seconds,
    read_gtfs_clean,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("Renfe_Pipeline_Unit_Tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def haversine_km_expr(lat1_col: str, lon1_col: str, lat2_col: str, lon2_col: str):
    """Expresión PySpark de distancia Haversine en kilómetros usada en el pipeline."""
    r_earth_km = 6371.0
    dlat = F.radians(F.col(lat2_col) - F.col(lat1_col))
    dlon = F.radians(F.col(lon2_col) - F.col(lon1_col))
    a = (
        F.pow(F.sin(dlat / 2.0), 2)
        + F.cos(F.radians(F.col(lat1_col)))
        * F.cos(F.radians(F.col(lat2_col)))
        * F.pow(F.sin(dlon / 2.0), 2)
    )
    return F.lit(2.0 * r_earth_km) * F.asin(F.sqrt(a))


def classify_severity_expr(delay_col: str):
    """Reglas de categorización de gravedad (0: <=3m, 1: 3-10m, 2: >10m)."""
    code = (
        F.when(F.col(delay_col) <= 180.0, F.lit(0.0))
        .when(F.col(delay_col) <= 600.0, F.lit(1.0))
        .otherwise(F.lit(2.0))
    )
    alert = (
        F.when(code == 0.0, F.lit("VERDE (Puntual <=3m)"))
        .when(code == 1.0, F.lit("AMARILLO (Moderado 3-10m)"))
        .otherwise(F.lit("ROJO (Grave >10m)"))
    )
    return code, alert


def test_gtfs_time_to_seconds_standard_and_overnight(spark):
    """Verifica horas estándar, espacios de relleno y horas GTFS posteriores a medianoche (>24:00:00)."""
    df = spark.createDataFrame(
        [
            ("08:15:30", 8 * 3600 + 15 * 60 + 30),
            ("  14:05:00  ", 14 * 3600 + 5 * 60),
            ("24:00:00", 24 * 3600),
            ("25:45:10", 25 * 3600 + 45 * 60 + 10),
        ],
        ["arrival_time", "expected_sec"],
    )

    res = df.withColumn("computed_sec", gtfs_time_to_seconds("arrival_time")).collect()
    for row in res:
        assert row["computed_sec"] == row["expected_sec"]


def test_read_gtfs_clean_trims_padding(spark, tmp_path):
    """Comprueba que la lectura de GTFS limpia los espacios en cabeceras y valores de Renfe."""
    gtfs_file = tmp_path / "stops.txt"
    gtfs_file.write_text(
        " stop_id , stop_name \n"
        " 18001   , Madrid-Atocha Cercanías \n",
        encoding="utf-8",
    )

    df = read_gtfs_clean(spark, str(gtfs_file))
    assert df.columns == ["stop_id", "stop_name"]
    row = df.collect()[0]
    assert row["stop_id"] == "18001"
    assert row["stop_name"] == "Madrid-Atocha Cercanías"


def test_haversine_distance_calculation(spark):
    """Valida el cálculo geodésico Haversine en coordenadas idénticas y estaciones reales."""
    df = spark.createDataFrame(
        [
            # Mismo punto -> 0.0 km
            (40.4065, -3.6896, 40.4065, -3.6896, 0.0, 1e-4),
            # 1 grado de latitud en el meridiano -> ~111.19 km
            (40.0000, -3.7000, 41.0000, -3.7000, 111.19, 0.5),
            # Atocha (40.4065, -3.6896) a Chamartín (40.4721, -3.6825) -> ~7.32 km
            (40.4065, -3.6896, 40.4721, -3.6825, 7.32, 0.25),
        ],
        ["lat1", "lon1", "lat2", "lon2", "expected_km", "tol_km"],
    )

    res = df.withColumn(
        "dist_km", haversine_km_expr("lat1", "lon1", "lat2", "lon2")
    ).collect()

    for row in res:
        assert math.isclose(row["dist_km"], row["expected_km"], abs_tol=row["tol_km"])


def test_severity_categorization_thresholds(spark):
    """Verifica los umbrales de clasificación operativa (0: <=180s, 1: 181-600s, 2: >600s)."""
    df = spark.createDataFrame(
        [
            (-60.0, 0.0, "VERDE (Puntual <=3m)"),
            (180.0, 0.0, "VERDE (Puntual <=3m)"),
            (181.0, 1.0, "AMARILLO (Moderado 3-10m)"),
            (600.0, 1.0, "AMARILLO (Moderado 3-10m)"),
            (601.0, 2.0, "ROJO (Grave >10m)"),
            (3600.0, 2.0, "ROJO (Grave >10m)"),
        ],
        ["delay_sec", "expected_code", "expected_alert"],
    )

    code_col, alert_col = classify_severity_expr("delay_sec")
    res = (
        df.withColumn("severity_label", code_col)
        .withColumn("severity_alert", alert_col)
        .collect()
    )

    for row in res:
        assert row["severity_label"] == row["expected_code"]
        assert row["severity_alert"] == row["expected_alert"]


def test_streaming_schema_and_output_contract(spark):
    """Valida el parseo del JSON GTFS-RT con STREAM_SCHEMA y el esquema final de salida."""
    sample_payload = {
        "header": {"gtfsRealtimeVersion": "2.0", "timestamp": "1795539857"},
        "entity": [
            {
                "id": "TU_1027M75024C8a",
                "tripUpdate": {
                    "trip": {"tripId": "1027M75024C8a", "routeId": "10T0008C8"},
                    "delay": 240.0,
                    "stopTimeUpdate": [
                        {
                            "stopId": "17000",
                            "arrival": {"delay": 240.0, "time": "1795540100"},
                        }
                    ],
                },
            }
        ],
    }

    raw_df = spark.createDataFrame([(json.dumps(sample_payload),)], ["raw_json"])
    parsed = raw_df.select(F.from_json("raw_json", STREAM_SCHEMA).alias("data")).select(
        F.to_timestamp(F.from_unixtime(F.col("data.header.timestamp").cast("long"))).alias("tu_ts"),
        F.explode("data.entity").alias("ent"),
    )

    flat = parsed.select(
        "tu_ts",
        F.regexp_extract(F.col("ent.tripUpdate.trip.tripId"), r"^\d+M\d+(.+)$", 1).alias("linea"),
        F.col("ent.tripUpdate.trip.tripId").alias("trip_id"),
        F.lit("Madrid-Chamartín").alias("stop_name"),
        F.explode("ent.tripUpdate.stopTimeUpdate").alias("stu"),
    ).select(
        "tu_ts",
        "linea",
        "trip_id",
        "stop_name",
        F.col("stu.arrival.delay").cast("double").alias("retraso_actual_s"),
        F.lit(255.0).cast(DoubleType()).alias("pred_retraso_s"),
        F.lit("AMARILLO (Moderado 3-10m)").alias("severity_alert"),
        F.lit(168.0).cast(DoubleType()).alias("congestion_linea_30m_s"),
    )

    expected_cols = [
        "tu_ts",
        "linea",
        "trip_id",
        "stop_name",
        "retraso_actual_s",
        "pred_retraso_s",
        "severity_alert",
        "congestion_linea_30m_s",
    ]
    assert flat.columns == expected_cols

    row = flat.collect()[0]
    assert row["linea"] == "C8a"
    assert row["trip_id"] == "1027M75024C8a"
    assert row["retraso_actual_s"] == 240.0