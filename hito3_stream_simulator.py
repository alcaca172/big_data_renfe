import glob
import os
import shutil
import time


def simulate_live_feed(
    source_dir: str = "data/raw/trip_updates",
    stream_in_dir: str = "data/stream_in/trip_updates",
    interval_sec: int = 6,
    max_files: int = 10,
) -> None:
    # Limpiamos la carpeta de entrada antes de iniciar la simulación
    shutil.rmtree(stream_in_dir, ignore_errors=True)
    os.makedirs(stream_in_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(source_dir, "*.json")))[:max_files]
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos JSON en '{source_dir}'.")

    print(f"[SIMULADOR] Inyectando {len(files)} capturas cada {interval_sec}s en '{stream_in_dir}'...")
    for idx, src_path in enumerate(files, start=1):
        filename = os.path.basename(src_path)
        tmp_dst = os.path.join(stream_in_dir, f"{filename}.tmp")
        final_dst = os.path.join(stream_in_dir, filename)

        # Evitar lecturas parciales en Structured Streaming
        shutil.copy2(src_path, tmp_dst)
        os.replace(tmp_dst, final_dst)

        print(f"  [{idx}/{len(files)}] Enviado -> {filename}")
        if idx < len(files):
            time.sleep(interval_sec)

    print("[SIMULADOR] Simulación finalizada.")


if __name__ == "__main__":
    simulate_live_feed()