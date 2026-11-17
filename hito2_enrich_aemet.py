import json
import math
import os
import urllib.request
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


AEMET_API_KEY = os.getenv("AEMET_API_KEY", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGNhY2ExNzJAZ21haWwuY29tIiwianRpIjoiMjgwYTFhMTgtMzA1NC00NDRmLTk5ZTMtOTMxOWIyNDU5NDY0IiwiZXhwIjoxODAzNTc2NDYxLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc5NDkzNjQ2MSwidXNlcklkIjoiMjgwYTFhMTgtMzA1NC00NDRmLTk5ZTMtOTMxOWIyNDU5NDY0Iiwicm9sZSI6IiJ9.2na2hRMiBc7lsRePIv_1LG8SLQl9LiQqukcawcxQFBg")
BASE_URL = "https://opendata.aemet.es/opendata/api"


def aemet_get_json(endpoint: str, api_key: str):
    """Gestiona la doble llamada oficial de AEMET OpenData y la codificación latin-1."""
    sep = "&" if "?" in endpoint else "?"
    url = f"{BASE_URL}{endpoint}{sep}api_key={api_key}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.loads(resp.read().decode("latin-1"))

    if meta.get("estado") != 200 or "datos" not in meta:
        raise RuntimeError(f"Error en AEMET ({endpoint}): {meta}")

    with urllib.request.urlopen(meta["datos"], timeout=60) as data_resp:
        return json.loads(data_resp.read().decode("latin-1"))


def dms_to_decimal(coord_str: str) -> float | None:
    """Convierte coordenadas AEMET (ej. '402443N' o '034041W') a grados decimales."""
    if not coord_str or len(coord_str) < 7:
        return None
    hemi = coord_str[-1].upper()
    digits = coord_str[:-1]
    deg_len = 3 if len(digits) == 7 else 2
    deg = int(digits[:deg_len])
    minutes = int(digits[deg_len : deg_len + 2])
    seconds = int(digits[deg_len + 2 : deg_len + 4])
    val = deg + minutes / 60.0 + seconds / 3600.0
    return -val if hemi in ("S", "W") else val


def haversine_py(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_aemet_float(val: str | None) -> float:
    """Convierte valores numéricos españoles ('12,4', 'Ip' = inapreciable) a float."""
    if val is None:
        return 0.0
    v = str(val).strip().replace(",", ".")
    if v in ("", "Ip", "Acum", "Varias"):
        return 0.05 if v == "Ip" else 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def enrich_with_weather(
    master_parquet: str = "data/processed/master_trains.parquet",
    output_parquet: str = "data/processed/master_trains_weather.parquet",
    cache_dir: str = "data/raw/aemet",
) -> None:
    if AEMET_API_KEY == "PEGA_AQUI_TU_API_KEY_DE_AEMET":
        raise ValueError("Configura tu AEMET_API_KEY antes de ejecutar el script.")

    os.makedirs(cache_dir, exist_ok=True)
    spark = (
        SparkSession.builder.appName("Renfe_Hito2_AEMET_Enrichment")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    try:
        trains_df = spark.read.parquet(master_parquet)

        # 1. Rango de fechas y estaciones Renfe únicas presentes en Parquet
        date_bounds = trains_df.agg(
            F.min("event_date").alias("min_d"),
            F.max("event_date").alias("max_d"),
        ).collect()[0]

        min_date = date_bounds["min_d"]
        max_date = date_bounds["max_d"] + timedelta(days=1)
        print(f"[INFO] Consultando meteorología AEMET entre {min_date} y {max_date}...")

        renfe_stops = (
            trains_df.select("stop_id", "stop_lat", "stop_lon")
            .filter(F.col("stop_lat").isNotNull() & F.col("stop_lon").isNotNull())
            .distinct()
            .collect()
        )

        # 2. Descargar inventario de estaciones AEMET
        stations_cache = os.path.join(cache_dir, "estaciones_aemet.json")
        if os.path.exists(stations_cache):
            with open(stations_cache, "r", encoding="utf-8") as f:
                stations_raw = json.load(f)
        else:
            stations_raw = aemet_get_json(
                "/valores/climatologicos/inventarioestaciones/todasestaciones", AEMET_API_KEY
            )
            with open(stations_cache, "w", encoding="utf-8") as f:
                json.dump(stations_raw, f, ensure_ascii=False, indent=2)

        aemet_stations = []
        for st in stations_raw:
            lat = dms_to_decimal(st.get("latitud", ""))
            lon = dms_to_decimal(st.get("longitud", ""))
            if lat is not None and lon is not None and st.get("indicativo"):
                aemet_stations.append((st["indicativo"], st.get("nombre", ""), lat, lon))

        # 3. Emparejar cada parada de Renfe con su estación AEMET más cercana
        stop_to_aemet = []
        needed_indicativos = set()
        for row in renfe_stops:
            best_id, best_name, best_dist = None, None, float("inf")
            for ind, name, alat, alon in aemet_stations:
                d = haversine_py(row["stop_lat"], row["stop_lon"], alat, alon)
                if d < best_dist:
                    best_id, best_name, best_dist = ind, name, d
            if best_id:
                stop_to_aemet.append((row["stop_id"], best_id, best_name, round(best_dist, 2)))
                needed_indicativos.add(best_id)

        map_df = spark.createDataFrame(
            stop_to_aemet, ["stop_id", "aemet_station_id", "aemet_station_name", "dist_aemet_km"]
        )

        # 4. Descargar observaciones diarias para el rango de fechas de tus datos
        weather_cache = os.path.join(cache_dir, f"clima_{min_date}_{max_date}.json")
        if os.path.exists(weather_cache):
            with open(weather_cache, "r", encoding="utf-8") as f:
                weather_raw = json.load(f)
        else:
            ini_str = f"{min_date}T00:00:00UTC"
            fin_str = f"{max_date}T23:59:59UTC"
            weather_raw = aemet_get_json(
                f"/valores/climatologicos/diarios/datos/fechaini/{ini_str}/fechafin/{fin_str}/todasestaciones",
                AEMET_API_KEY,
            )
            with open(weather_cache, "w", encoding="utf-8") as f:
                json.dump(weather_raw, f, ensure_ascii=False, indent=2)

        weather_rows = []
        for w in weather_raw:
            ind = w.get("indicativo")
            if ind in needed_indicativos and w.get("fecha"):
                weather_rows.append(
                    (
                        ind,
                        w["fecha"],
                        parse_aemet_float(w.get("prec")),
                        parse_aemet_float(w.get("tmed")),
                        parse_aemet_float(w.get("velmedia")),
                        parse_aemet_float(w.get("racha")),
                    )
                )

        weather_df = spark.createDataFrame(
            weather_rows,
            ["aemet_station_id", "weather_date_str", "precip_mm", "temp_avg_c", "wind_avg_ms", "wind_gust_ms"],
        ).withColumn("event_date", F.to_date("weather_date_str")).drop("weather_date_str")

        # 5. Unir meteorología a la tabla maestra de trenes
        enriched_df = (
            trains_df.join(F.broadcast(map_df), on="stop_id", how="left")
            .join(F.broadcast(weather_df), on=["aemet_station_id", "event_date"], how="left")
            .fillna({"precip_mm": 0.0, "temp_avg_c": 15.0, "wind_avg_ms": 0.0, "wind_gust_ms": 0.0})
        )

        enriched_df.write.mode("overwrite").partitionBy("event_date").parquet(output_parquet)

        res = spark.read.parquet(output_parquet)
        print(f"[OK] Dataset enriquecido guardado en '{output_parquet}' con {res.count():,} registros.")
        res.select(
            "stop_name",
            "event_date",
            "arrival_delay_seconds",
            "aemet_station_name",
            "dist_aemet_km",
            "precip_mm",
            "temp_avg_c",
            "wind_avg_ms",
        ).show(5, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    enrich_with_weather()