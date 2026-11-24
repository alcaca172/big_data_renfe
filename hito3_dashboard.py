import glob
import os
import time
import pandas as pd
import streamlit as st


PREDICTIONS_DIR = "data/stream_out/predictions"

st.set_page_config(
    page_title="Renfe Cercanías - Monitor en Tiempo Real",
    page_icon="🚆",
    layout="wide",
)


def load_latest_predictions(parquet_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee de forma segura los ficheros Parquet generados por Spark Structured Streaming."""
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        return pd.DataFrame(), pd.DataFrame()

    frames = []
    for path in files:
        # Ignoramos ficheros vacíos o temporales mientras Spark escribe un micro-batch
        if os.path.getsize(path) == 0:
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(frames, ignore_index=True)
    if full_df.empty or "tu_ts" not in full_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    latest_ts = full_df["tu_ts"].max()
    latest_df = full_df[full_df["tu_ts"] == latest_ts].copy()
    return latest_df, full_df


st.title("🚆 Centro de Control Ferroviario Renfe — Predicción en Tiempo Real")
st.caption("Spark Structured Streaming + SparkML | Actualización continua sobre GTFS-RT")

with st.sidebar:
    st.header("Controles de Monitorización")
    auto_refresh = st.toggle("Refresco automático (cada 5 s)", value=True)
    selected_alerts = st.multiselect(
        "Filtrar por nivel de alerta:",
        options=[
            "ROJO (Grave >10m)",
            "AMARILLO (Moderado 3-10m)",
            "VERDE (Puntual <=3m)",
        ],
        default=[
            "ROJO (Grave >10m)",
            "AMARILLO (Moderado 3-10m)",
            "VERDE (Puntual <=3m)",
        ],
    )

latest_df, full_df = load_latest_predictions(PREDICTIONS_DIR)

if latest_df.empty:
    st.info(
        "Esperando predicciones en `data/stream_out/predictions/`... "
        "Arranca `hito3_streaming_inference.py` y `hito3_live_producer.py` (o el simulador)."
    )
else:
    latest_ts = latest_df["tu_ts"].max()
    st.markdown(f"**Última captura procesada:** `{latest_ts}`")

    # 1. Tarjetas KPI globales
    total_paradas = len(latest_df)
    n_verde = int((latest_df["severity_alert"] == "VERDE (Puntual <=3m)").sum())
    n_amarillo = int((latest_df["severity_alert"] == "AMARILLO (Moderado 3-10m)").sum())
    n_rojo = int((latest_df["severity_alert"] == "ROJO (Grave >10m)").sum())
    avg_pred_min = round(latest_df["pred_retraso_s"].mean() / 60.0, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Paradas monitorizadas", total_paradas)
    c2.metric("🟢 Puntual (<=3m)", n_verde)
    c3.metric("🟡 Moderado (3-10m)", n_amarillo)
    c4.metric("🔴 Grave (>10m)", n_rojo)
    c5.metric("Retraso medio predicho", f"{avg_pred_min} min")

    st.divider()

    # 2. Gráfico de líneas más congestionadas y evolución temporal
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Líneas por Congestión Móvil (30 min)")
        line_cong = (
            latest_df.groupby("linea", as_index=False)["congestion_linea_30m_s"]
            .mean()
            .sort_values("congestion_linea_30m_s", ascending=False)
            .head(10)
            .set_index("linea")
        )
        st.bar_chart(line_cong["congestion_linea_30m_s"])

    with col_right:
        st.subheader("Evolución de Alertas por Micro-Batch")
        hist_alerts = (
            full_df.groupby(["tu_ts", "severity_alert"])
            .size()
            .unstack(fill_value=0)
            .tail(20)
        )
        st.line_chart(hist_alerts)

    # 3. Tabla detallada filtrable
    st.subheader("Detalle de Trenes y Próximas Paradas")
    lineas_disponibles = sorted(latest_df["linea"].dropna().unique().tolist())
    selected_lines = st.multiselect(
        "Filtrar por línea (vacío = todas):",
        options=lineas_disponibles,
        default=[],
    )

    filtered_df = latest_df[latest_df["severity_alert"].isin(selected_alerts)]
    if selected_lines:
        filtered_df = filtered_df[filtered_df["linea"].isin(selected_lines)]

    filtered_df = filtered_df.sort_values("pred_retraso_s", ascending=False)
    st.dataframe(
        filtered_df[
            [
                "linea",
                "trip_id",
                "stop_name",
                "retraso_actual_s",
                "pred_retraso_s",
                "severity_alert",
                "congestion_linea_30m_s",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

if auto_refresh:
    time.sleep(5)
    st.rerun()