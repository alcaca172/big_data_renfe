import json
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


DEFAULT_RENFE_TU_URL = os.getenv(
    "RENFE_TRIP_UPDATES_URL",
    "https://gtfsrt.renfe.com/trip_updates.json",
)


def fetch_trip_updates(url: str, timeout_sec: int = 12) -> dict:
    """Descarga y valida el JSON GTFS-RT de trip_updates desde Renfe OpenData."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RenfeBigDataStreamProducer/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        raw_bytes = response.read()
        payload = json.loads(raw_bytes.decode("utf-8"))

    if not isinstance(payload, dict) or "entity" not in payload:
        raise ValueError("Respuesta JSON inesperada: falta el campo raíz 'entity'.")

    return payload


def run_live_producer(
    url: str = DEFAULT_RENFE_TU_URL,
    stream_in_dir: str = "data/stream_in/trip_updates",
    interval_sec: int = 20,
    clean_dir_on_start: bool = True,
    max_cycles: int | None = None,
) -> None:
    """Consulta la API de Renfe cada 20 s y escribe cada captura de forma atómica."""
    if clean_dir_on_start:
        shutil.rmtree(stream_in_dir, ignore_errors=True)
    os.makedirs(stream_in_dir, exist_ok=True)

    print(f"[PRODUCTOR EN VIVO] Conectando a: {url}")
    print(f"[PRODUCTOR EN VIVO] Destino stream: '{stream_in_dir}' | Intervalo: {interval_sec}s")
    print("[PRODUCTOR EN VIVO] Pulsa Ctrl+C para detener la ingesta.\n")

    last_feed_ts = None
    cycle = 0

    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            cycle_start = time.perf_counter()
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            try:
                payload = fetch_trip_updates(url)
                header = payload.get("header", {})
                feed_ts = str(header.get("timestamp", now_str))
                num_entities = len(payload.get("entity", []))

                if feed_ts == last_feed_ts:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Sin cambios en el feed (timestamp={feed_ts}). Esperando siguiente ciclo..."
                    )
                else:
                    filename = f"trip_updates_{feed_ts}_{now_str}.json"
                    tmp_path = os.path.join(stream_in_dir, f"{filename}.tmp")
                    final_path = os.path.join(stream_in_dir, filename)

                    # Escritura atómica (.tmp -> .json) para evitar lecturas parciales en Spark
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False)
                    os.replace(tmp_path, final_path)

                    last_feed_ts = feed_ts
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Captura #{cycle} guardada -> {filename} ({num_entities} trenes activos)"
                    )

            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as err:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[WARN] Fallo transitorio al consultar Renfe: {err}. Reintentando en {interval_sec}s..."
                )

            if max_cycles is not None and cycle >= max_cycles:
                break

            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(1.0, interval_sec - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[PRODUCTOR EN VIVO] Ingesta detenida por el usuario.")


if __name__ == "__main__":
    run_live_producer()