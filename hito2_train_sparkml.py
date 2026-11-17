import json
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("Renfe_Hito2_Train_SparkML")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def prepare_dataset(df: DataFrame) -> DataFrame:
    """Limpia nulos con valores por defecto seguros y crea la etiqueta de gravedad."""
    # Rellenamos posibles nulos en telemetría o cruces estáticos para que VectorAssembler no falle
    df_clean = df.fillna(
        {
            "route_short_name": "DESCONOCIDA",
            "train_status": "SIN_TELEMETRIA",
            "stop_sequence": 1,
            "scheduled_arrival_sec": 43200,
            "dist_to_stop_km": -1.0,
            "active_incidents_count": 0,
            "has_active_incident": 0,
            "precip_mm": 0.0,
            "temp_avg_c": 15.0,
            "wind_avg_ms": 0.0,
            "wind_gust_ms": 0.0,
        }
    )

    # Creamos la variable objetivo para Clasificación (Gravedad del retraso):
    #   0 = Puntual / Leve (<= 3 min / 180s)
    #   1 = Retraso Moderado (entre 3 y 10 min / 181s - 600s)
    #   2 = Incidencia / Retraso Grave (> 10 min / > 600s)
    df_labeled = df_clean.withColumn(
        "severity_label",
        F.when(F.col("arrival_delay_seconds") <= 180, 0.0)
        .when(F.col("arrival_delay_seconds") <= 600, 1.0)
        .otherwise(2.0),
    ).withColumn("ts_epoch", F.unix_timestamp("tu_ts"))

    return df_labeled


def temporal_train_test_split(df: DataFrame, train_ratio: float = 0.8):
    """Divide entrenamiento (80% inicial) y test (20% posterior) respetando el orden temporal."""
    cutoff = df.approxQuantile("ts_epoch", [train_ratio], 0.01)[0]
    train_df = df.filter(F.col("ts_epoch") <= cutoff).cache()
    test_df = df.filter(F.col("ts_epoch") > cutoff).cache()

    # Si todos los registros tuvieran exactamente el mismo timestamp, hacemos fallback aleatorio
    if test_df.count() == 0:
        train_df.unpersist()
        test_df.unpersist()
        train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=42)
        train_df.cache()
        test_df.cache()

    return train_df, test_df


def build_feature_stages():
    """Construye las etapas de transformación de variables categóricas y numéricas."""
    categorical_cols = ["route_short_name", "train_status"]
    numeric_cols = [
        "stop_sequence",
        "scheduled_arrival_sec",
        "dist_to_stop_km",
        "active_incidents_count",
        "has_active_incident",
        "hour_of_day",
        "day_of_week",
        "precip_mm",
        "temp_avg_c",
        "wind_avg_ms",
        "wind_gust_ms",
    ]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec", handleInvalid="keep")
        for c in categorical_cols
    ]

    feature_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_inputs, outputCol="features")

    return indexers + encoders + [assembler], feature_inputs


def train_and_evaluate(
    input_parquet: str = "data/processed/master_trains_weather.parquet",
    models_dir: str = "models",
) -> None:
    os.makedirs(models_dir, exist_ok=True)
    spark = create_spark_session()

    try:
        raw_df = spark.read.parquet(input_parquet)
        df = prepare_dataset(raw_df)

        print("\n=== DISTRIBUCIÓN DE CLASES DE GRAVEDAD ===")
        df.groupBy("severity_label").count().orderBy("severity_label").show()

        train_df, test_df = temporal_train_test_split(df, train_ratio=0.8)
        print(f"[INFO] Registros Train: {train_df.count():,} | Registros Test: {test_df.count():,}")

        feature_stages, _ = build_feature_stages()

        # =====================================================================
        # MODELO 1: REGRESIÓN (Predicción de retraso exacto en segundos)
        # =====================================================================
        print("\n=== ENTRENANDO MODELOS DE REGRESIÓN (arrival_delay_seconds) ===")
        reg_models = {
            "RandomForestRegressor": RandomForestRegressor(
                featuresCol="features",
                labelCol="arrival_delay_seconds",
                numTrees=80,
                maxDepth=8,
                seed=42,
            ),
            "GBTRegressor": GBTRegressor(
                featuresCol="features",
                labelCol="arrival_delay_seconds",
                maxIter=50,
                maxDepth=5,
                seed=42,
            ),
        }

        eval_rmse = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="rmse"
        )
        eval_mae = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="mae"
        )
        eval_r2 = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="r2"
        )

        best_reg_name = None
        best_reg_pipeline = None
        best_rmse = float("inf")
        reg_metrics = {}

        for name, estimator in reg_models.items():
            pipeline = Pipeline(stages=feature_stages + [estimator])
            model = pipeline.fit(train_df)
            preds = model.transform(test_df)

            rmse = eval_rmse.evaluate(preds)
            mae = eval_mae.evaluate(preds)
            r2 = eval_r2.evaluate(preds)
            reg_metrics[name] = {"RMSE_sec": round(rmse, 2), "MAE_sec": round(mae, 2), "R2": round(r2, 4)}
            print(f"  -> {name:22s} | RMSE: {rmse:7.2f} s | MAE: {mae:7.2f} s | R2: {r2:6.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_reg_name = name
                best_reg_pipeline = model

        reg_save_path = os.path.join(models_dir, "delay_regressor_pipeline")
        best_reg_pipeline.write().overwrite().save(reg_save_path)
        print(f"[OK] Mejor regresor ({best_reg_name}) guardado en '{reg_save_path}'")

        # =====================================================================
        # MODELO 2: CLASIFICACIÓN (Gravedad del retraso: 0, 1 o 2)
        # =====================================================================
        print("\n=== ENTRENANDO MODELOS DE CLASIFICACIÓN (severity_label) ===")
        clf_models = {
            "RandomForestClassifier": RandomForestClassifier(
                featuresCol="features",
                labelCol="severity_label",
                numTrees=80,
                maxDepth=8,
                seed=42,
            ),
            "LogisticRegression": LogisticRegression(
                featuresCol="features",
                labelCol="severity_label",
                maxIter=100,
                regParam=0.01,
            ),
        }

        eval_f1 = MulticlassClassificationEvaluator(
            labelCol="severity_label", predictionCol="prediction", metricName="f1"
        )
        eval_acc = MulticlassClassificationEvaluator(
            labelCol="severity_label", predictionCol="prediction", metricName="accuracy"
        )

        best_clf_name = None
        best_clf_pipeline = None
        best_clf_preds = None
        best_f1 = -1.0
        clf_metrics = {}

        for name, estimator in clf_models.items():
            pipeline = Pipeline(stages=feature_stages + [estimator])
            model = pipeline.fit(train_df)
            preds = model.transform(test_df)

            f1 = eval_f1.evaluate(preds)
            acc = eval_acc.evaluate(preds)
            clf_metrics[name] = {"F1_weighted": round(f1, 4), "Accuracy": round(acc, 4)}
            print(f"  -> {name:22s} | F1-Score: {f1:6.4f} | Accuracy: {acc:6.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_clf_name = name
                best_clf_pipeline = model
                best_clf_preds = preds

        clf_save_path = os.path.join(models_dir, "severity_classifier_pipeline")
        best_clf_pipeline.write().overwrite().save(clf_save_path)
        print(f"[OK] Mejor clasificador ({best_clf_name}) guardado en '{clf_save_path}'")

        print(f"\n=== MATRIZ DE CONFUSIÓN ({best_clf_name}) ===")
        (
            best_clf_preds.groupBy("severity_label", "prediction")
            .count()
            .orderBy("severity_label", "prediction")
            .show()
        )

        # Guardamos resumen de métricas y etiquetas para documentación e inferencia en Hito 3
        metadata = {
            "severity_classes": {
                "0": "Puntual / Leve (<= 3 min)",
                "1": "Retraso Moderado (3 - 10 min)",
                "2": "Incidencia / Retraso Grave (> 10 min)",
            },
            "best_regressor": best_reg_name,
            "regression_metrics": reg_metrics,
            "best_classifier": best_clf_name,
            "classification_metrics": clf_metrics,
        }
        with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    finally:
        spark.stop()


if __name__ == "__main__":
    train_and_evaluate()