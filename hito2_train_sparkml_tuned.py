import json
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("Renfe_Hito2_Train_SparkML_Tuned")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def engineer_features(df: DataFrame) -> DataFrame:
    """Limpieza, ingeniería de variables ferroviarias y etiquetado de gravedad."""
    df_clean = df.fillna(
        {
            "route_id": "DESCONOCIDA",
            "route_short_name": "DESCONOCIDA",
            "stop_id": "DESCONOCIDA",
            "train_status": "SIN_TELEMETRIA",
            "stop_sequence": 1,
            "scheduled_arrival_sec": 43200,
            "scheduled_departure_sec": 43200,
            "dist_to_stop_km": -1.0,
            "active_incidents_count": 0,
            "has_active_incident": 0,
            "precip_mm": 0.0,
            "temp_avg_c": 15.0,
            "wind_avg_ms": 0.0,
            "wind_gust_ms": 0.0,
        }
    ).withColumn("ts_epoch", F.unix_timestamp("tu_ts"))

    # 1. Variables de contexto operativo y calendario
    df_feat = (
        df_clean.withColumn(
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
                (F.col("scheduled_departure_sec") - F.col("scheduled_arrival_sec")).cast("double"),
            ),
        )
    )

    # 2. Congestión reciente en el corredor (ventana causal estricta: últimos 30 min)
    # rangeBetween(-1800, -1) excluye el instante actual (0) para evitar fuga de datos.
    w_corridor_30m = (
        Window.partitionBy("route_id")
        .orderBy("ts_epoch")
        .rangeBetween(-1800, -1)
    )

    df_feat = (
        df_feat.withColumn(
            "route_rolling_delay_30m",
            F.coalesce(F.avg("arrival_delay_seconds").over(w_corridor_30m), F.lit(0.0)),
        )
        .withColumn(
            "route_rolling_trains_30m",
            F.count("trip_id").over(w_corridor_30m).cast("double"),
        )
    )

    # 3. Etiqueta de gravedad (3 clases)
    df_labeled = df_feat.withColumn(
        "severity_label",
        F.when(F.col("arrival_delay_seconds") <= 180, 0.0)
        .when(F.col("arrival_delay_seconds") <= 600, 1.0)
        .otherwise(2.0),
    )

    return df_labeled


def temporal_train_test_split(df: DataFrame, train_ratio: float = 0.8):
    """Divide entrenamiento (80% inicial) y test (20% posterior) respetando el orden temporal."""
    cutoff = df.approxQuantile("ts_epoch", [train_ratio], 0.01)[0]
    train_df = df.filter(F.col("ts_epoch") <= cutoff).cache()
    test_df = df.filter(F.col("ts_epoch") > cutoff).cache()

    if test_df.count() == 0:
        train_df.unpersist()
        test_df.unpersist()
        train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=42)
        train_df.cache()
        test_df.cache()

    return train_df, test_df


def add_inverse_class_weights(train_df: DataFrame, label_col: str = "severity_label") -> DataFrame:
    """Calcula pesos inversos a la frecuencia de cada clase solo sobre Train."""
    total_rows = train_df.count()
    counts = train_df.groupBy(label_col).count().collect()
    num_classes = len(counts)

    weight_expr = F.lit(1.0)
    for row in counts:
        cls_val = row[label_col]
        cls_count = row["count"]
        w = total_rows / (num_classes * cls_count)
        weight_expr = F.when(F.col(label_col) == cls_val, F.lit(float(w))).otherwise(weight_expr)

    return train_df.withColumn("class_weight", weight_expr)


def build_feature_stages():
    """Preprocesamiento categórico (incluyendo stop_id) y ensamblado de vector."""
    categorical_cols = ["route_short_name", "stop_id", "train_status"]
    numeric_cols = [
        "stop_sequence",
        "scheduled_arrival_sec",
        "scheduled_dwell_sec",
        "dist_to_stop_km",
        "active_incidents_count",
        "has_active_incident",
        "hour_of_day",
        "day_of_week",
        "is_rush_hour",
        "is_weekend",
        "route_rolling_delay_30m",
        "route_rolling_trains_30m",
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

    return indexers + encoders + [assembler]


def print_top_importances(pipeline_model, transformed_df: DataFrame, top_n: int = 8) -> None:
    """Muestra las variables que más peso han tenido en la decisión del árbol."""
    try:
        estimator_model = pipeline_model.stages[-1]
        if not hasattr(estimator_model, "featureImportances"):
            return
        importances = estimator_model.featureImportances.toArray()
        attrs_meta = transformed_df.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})
        idx_to_name = {}
        for attr_group in attrs_meta.values():
            for item in attr_group:
                idx_to_name[item["idx"]] = item["name"]

        ranked = sorted(
            [(idx_to_name.get(i, f"feat_{i}"), float(imp)) for i, imp in enumerate(importances)],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        print(f"  Top {top_n} variables más influyentes:")
        for feat_name, score in ranked:
            print(f"    - {feat_name:35s}: {score:.4f}")
    except Exception:
        pass


def train_tuned_models(
    input_parquet: str = "data/processed/master_trains_weather.parquet",
    models_dir: str = "models",
) -> None:
    os.makedirs(models_dir, exist_ok=True)
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    prev_meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            prev_meta = json.load(f)

    spark = create_spark_session()
    try:
        raw_df = spark.read.parquet(input_parquet)
        df = engineer_features(raw_df)

        train_df, test_df = temporal_train_test_split(df, train_ratio=0.8)
        train_df = add_inverse_class_weights(train_df, "severity_label").cache()
        test_df = test_df.withColumn("class_weight", F.lit(1.0)).cache()

        print(f"[INFO] Train: {train_df.count():,} | Test: {test_df.count():,}")
        feature_stages = build_feature_stages()

        # =====================================================================
        # 1. REGRESIÓN AFINADA (CrossValidator sobre RandomForest + GBT)
        # =====================================================================
        print("\n=== ENTRENANDO REGRESIÓN AFINADA (3-Fold Cross-Validation) ===")
        eval_rmse = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="rmse"
        )
        eval_mae = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="mae"
        )
        eval_r2 = RegressionEvaluator(
            labelCol="arrival_delay_seconds", predictionCol="prediction", metricName="r2"
        )

        # Grid para RandomForestRegressor
        rf_reg = RandomForestRegressor(
            featuresCol="features", labelCol="arrival_delay_seconds", seed=42
        )
        rf_reg_pipe = Pipeline(stages=feature_stages + [rf_reg])
        rf_reg_grid = (
            ParamGridBuilder()
            .addGrid(rf_reg.numTrees, [80, 140])
            .addGrid(rf_reg.maxDepth, [8, 12])
            .addGrid(rf_reg.minInstancesPerNode, [1, 3])
            .build()
        )
        cv_rf_reg = CrossValidator(
            estimator=rf_reg_pipe,
            estimatorParamMaps=rf_reg_grid,
            evaluator=eval_rmse,
            numFolds=3,
            parallelism=2,
            seed=42,
        )
        best_rf_reg_model = cv_rf_reg.fit(train_df).bestModel
        preds_rf_reg = best_rf_reg_model.transform(test_df)

        # Grid para GBTRegressor (con regularización para evitar sobreajuste)
        gbt_reg = GBTRegressor(
            featuresCol="features", labelCol="arrival_delay_seconds", seed=42
        )
        gbt_reg_pipe = Pipeline(stages=feature_stages + [gbt_reg])
        gbt_reg_grid = (
            ParamGridBuilder()
            .addGrid(gbt_reg.maxIter, [30, 50])
            .addGrid(gbt_reg.maxDepth, [3, 5])
            .addGrid(gbt_reg.stepSize, [0.05, 0.1])
            .build()
        )
        cv_gbt_reg = CrossValidator(
            estimator=gbt_reg_pipe,
            estimatorParamMaps=gbt_reg_grid,
            evaluator=eval_rmse,
            numFolds=3,
            parallelism=2,
            seed=42,
        )
        best_gbt_reg_model = cv_gbt_reg.fit(train_df).bestModel
        preds_gbt_reg = best_gbt_reg_model.transform(test_df)

        reg_candidates = {
            "RandomForestRegressor_Tuned": (best_rf_reg_model, preds_rf_reg),
            "GBTRegressor_Tuned": (best_gbt_reg_model, preds_gbt_reg),
        }

        best_reg_name = None
        best_reg_model = None
        best_reg_preds = None
        best_rmse = float("inf")
        tuned_reg_metrics = {}

        for name, (model, preds) in reg_candidates.items():
            rmse = eval_rmse.evaluate(preds)
            mae = eval_mae.evaluate(preds)
            r2 = eval_r2.evaluate(preds)
            tuned_reg_metrics[name] = {
                "RMSE_sec": round(rmse, 2),
                "MAE_sec": round(mae, 2),
                "R2": round(r2, 4),
            }
            print(f"  -> {name:28s} | RMSE: {rmse:7.2f} s | MAE: {mae:7.2f} s | R2: {r2:6.4f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_reg_name = name
                best_reg_model = model
                best_reg_preds = preds

        print_top_importances(best_reg_model, best_reg_preds)

        # =====================================================================
        # 2. CLASIFICACIÓN AFINADA (Pesos de clase + CrossValidator)
        # =====================================================================
        print("\n=== ENTRENANDO CLASIFICACIÓN AFINADA (Pesos de Clase + 3-Fold CV) ===")
        eval_f1 = MulticlassClassificationEvaluator(
            labelCol="severity_label", predictionCol="prediction", metricName="f1"
        )
        eval_acc = MulticlassClassificationEvaluator(
            labelCol="severity_label", predictionCol="prediction", metricName="accuracy"
        )

        rf_clf = RandomForestClassifier(
            featuresCol="features",
            labelCol="severity_label",
            weightCol="class_weight",
            seed=42,
        )
        rf_clf_pipe = Pipeline(stages=feature_stages + [rf_clf])
        rf_clf_grid = (
            ParamGridBuilder()
            .addGrid(rf_clf.numTrees, [80, 140])
            .addGrid(rf_clf.maxDepth, [8, 12])
            .addGrid(rf_clf.minInstancesPerNode, [1, 3])
            .build()
        )
        cv_rf_clf = CrossValidator(
            estimator=rf_clf_pipe,
            estimatorParamMaps=rf_clf_grid,
            evaluator=eval_f1,
            numFolds=3,
            parallelism=2,
            seed=42,
        )
        best_rf_clf_model = cv_rf_clf.fit(train_df).bestModel
        preds_rf_clf = best_rf_clf_model.transform(test_df)

        lr_clf = LogisticRegression(
            featuresCol="features",
            labelCol="severity_label",
            weightCol="class_weight",
            maxIter=120,
        )
        lr_clf_pipe = Pipeline(stages=feature_stages + [lr_clf])
        lr_clf_grid = (
            ParamGridBuilder()
            .addGrid(lr_clf.regParam, [0.005, 0.02, 0.1])
            .addGrid(lr_clf.elasticNetParam, [0.0, 0.5])
            .build()
        )
        cv_lr_clf = CrossValidator(
            estimator=lr_clf_pipe,
            estimatorParamMaps=lr_clf_grid,
            evaluator=eval_f1,
            numFolds=3,
            parallelism=2,
            seed=42,
        )
        best_lr_clf_model = cv_lr_clf.fit(train_df).bestModel
        preds_lr_clf = best_lr_clf_model.transform(test_df)

        clf_candidates = {
            "RandomForestClassifier_Tuned": (best_rf_clf_model, preds_rf_clf),
            "LogisticRegression_Tuned": (best_lr_clf_model, preds_lr_clf),
        }

        best_clf_name = None
        best_clf_model = None
        best_clf_preds = None
        best_f1 = -1.0
        tuned_clf_metrics = {}

        for name, (model, preds) in clf_candidates.items():
            f1 = eval_f1.evaluate(preds)
            acc = eval_acc.evaluate(preds)
            tuned_clf_metrics[name] = {"F1_weighted": round(f1, 4), "Accuracy": round(acc, 4)}
            print(f"  -> {name:28s} | F1-Score: {f1:6.4f} | Accuracy: {acc:6.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_clf_name = name
                best_clf_model = model
                best_clf_preds = preds

        print_top_importances(best_clf_model, best_clf_preds)

        print(f"\n=== MATRIZ DE CONFUSIÓN AFINADA ({best_clf_name}) ===")
        (
            best_clf_preds.groupBy("severity_label", "prediction")
            .count()
            .orderBy("severity_label", "prediction")
            .show()
        )

        # =====================================================================
        # 3. GUARDADO DE MODELOS AFINADOS Y COMPARATIVA CON BASELINE
        # =====================================================================
        reg_tuned_path = os.path.join(models_dir, "delay_regressor_tuned_pipeline")
        clf_tuned_path = os.path.join(models_dir, "severity_classifier_tuned_pipeline")
        best_reg_model.write().overwrite().save(reg_tuned_path)
        best_clf_model.write().overwrite().save(clf_tuned_path)

        # Si superan al baseline previo, actualizamos también los pipelines principales para Hito 3
        prev_best_reg = prev_meta.get("best_regressor")
        prev_rmse = (
            prev_meta.get("regression_metrics", {}).get(prev_best_reg, {}).get("RMSE_sec", float("inf"))
        )
        if best_rmse <= prev_rmse:
            best_reg_model.write().overwrite().save(os.path.join(models_dir, "delay_regressor_pipeline"))
            print(f"[UPGRADE] Regresor principal actualizado (RMSE: {prev_rmse} s -> {best_rmse:.2f} s)")

        prev_best_clf = prev_meta.get("best_classifier")
        prev_f1 = (
            prev_meta.get("classification_metrics", {}).get(prev_best_clf, {}).get("F1_weighted", -1.0)
        )
        if best_f1 >= prev_f1:
            best_clf_model.write().overwrite().save(os.path.join(models_dir, "severity_classifier_pipeline"))
            print(f"[UPGRADE] Clasificador principal actualizado (F1: {prev_f1} -> {best_f1:.4f})")

        prev_meta["tuned_best_regressor"] = best_reg_name
        prev_meta["tuned_regression_metrics"] = tuned_reg_metrics
        prev_meta["tuned_best_classifier"] = best_clf_name
        prev_meta["tuned_classification_metrics"] = tuned_clf_metrics

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(prev_meta, f, ensure_ascii=False, indent=2)
        print(f"[OK] Resultados guardados en '{metadata_path}'.")

    finally:
        spark.stop()


if __name__ == "__main__":
    train_tuned_models()