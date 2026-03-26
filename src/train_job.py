import io
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PACKAGES_PATH, resolve_path
from src.dataset_snapshot import load_or_create_snapshot
from src.inference_engine import (
    PROFILE_NUMERIC_RANGES,
    has_any_keyword,
    normalize_range,
)
from src.mlflow_utils import log_dict_artifact, start_run_if_available

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NAVAVEJ_PACKAGES_PATH = resolve_path("NAVAVEJ_PACKAGES_PATH", DEFAULT_PACKAGES_PATH)
MODEL_OUTPUT_PATH = resolve_path("NAVAVEJ_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_OUTPUT_DIR = MODEL_OUTPUT_PATH.parent


def load_packages_dataframe():
    if NAVAVEJ_PACKAGES_PATH.exists():
        return pd.read_csv(NAVAVEJ_PACKAGES_PATH)

    print(f"ไม่พบ package catalog ที่ {NAVAVEJ_PACKAGES_PATH} จะ fallback เป็น code=name สำหรับเดโม")
    return pd.DataFrame(columns=["code", "name"])


def build_patient_features(patient_profiles_df):
    patient_rows = []

    for patient_id, row in patient_profiles_df.iterrows():
        vitals = row.get("VITALS", {}) or {}
        conditions = str(row.get("CONDITIONS", "")).lower()

        patient_rows.append(
            {
                "PATIENT": patient_id,
                "AGE_NORM": normalize_range(row["AGE"], *PROFILE_NUMERIC_RANGES["AGE"]),
                "PACKAGE_COUNT_NORM": normalize_range(row.get("PACKAGE_COUNT", 0), *PROFILE_NUMERIC_RANGES["PACKAGE_COUNT"]),
                "GENDER_M": 1.0 if row["GENDER"] == "M" else 0.0,
                "GENDER_F": 1.0 if row["GENDER"] == "F" else 0.0,
                "IS_CHILD": 1.0 if row["AGE"] <= 14 else 0.0,
                "IS_ADULT": 1.0 if 15 <= row["AGE"] < 55 else 0.0,
                "IS_SENIOR": 1.0 if row["AGE"] >= 55 else 0.0,
                "HAS_DIABETES": 1.0 if "diabetes" in conditions else 0.0,
                "HAS_HYPERTENSION": 1.0 if "hypertension" in conditions else 0.0,
                "HAS_HYPERLIPIDEMIA": 1.0 if "hyperlipidemia" in conditions else 0.0,
                "HAS_PREGNANCY": 1.0 if ("pregnancy" in conditions or "prenatal" in conditions) else 0.0,
                "HAS_OBESITY": has_any_keyword(conditions, ["obesity", "overweight"]),
                "HAS_STROKE_RISK": has_any_keyword(conditions, ["stroke", "cerebrovascular"]),
                "HAS_HEART_DISEASE": has_any_keyword(conditions, ["coronary", "heart", "cardio", "myocardial"]),
                "HAS_KIDNEY_DISEASE": has_any_keyword(conditions, ["kidney", "renal", "nephro"]),
                "BMI_NORM": normalize_range(vitals.get("BMI", 25.0), *PROFILE_NUMERIC_RANGES["BMI"]),
                "GLUCOSE_NORM": normalize_range(vitals.get("Glucose", 100.0), *PROFILE_NUMERIC_RANGES["Glucose"]),
                "SYSBP_NORM": normalize_range(vitals.get("SysBP", 120.0), *PROFILE_NUMERIC_RANGES["SysBP"]),
                "DIABP_NORM": normalize_range(vitals.get("DiaBP", 80.0), *PROFILE_NUMERIC_RANGES["DiaBP"]),
                "CHOLESTEROL_NORM": normalize_range(vitals.get("Cholesterol", 180.0), *PROFILE_NUMERIC_RANGES["Cholesterol"]),
                "HEART_RATE_NORM": normalize_range(vitals.get("HeartRate", 78.0), *PROFILE_NUMERIC_RANGES["HeartRate"]),
                "RESP_RATE_NORM": normalize_range(vitals.get("RespRate", 16.0), *PROFILE_NUMERIC_RANGES["RespRate"]),
                "HEIGHT_NORM": normalize_range(vitals.get("BodyHeight", 165.0), *PROFILE_NUMERIC_RANGES["BodyHeight"]),
                "WEIGHT_NORM": normalize_range(vitals.get("BodyWeight", 68.0), *PROFILE_NUMERIC_RANGES["BodyWeight"]),
                "HBA1C_NORM": normalize_range(vitals.get("HbA1c", 5.6), *PROFILE_NUMERIC_RANGES["HbA1c"]),
                "EGFR_NORM": normalize_range(vitals.get("eGFR", 90.0), *PROFILE_NUMERIC_RANGES["eGFR"]),
                "TRIGLYCERIDES_NORM": normalize_range(vitals.get("Triglycerides", 150.0), *PROFILE_NUMERIC_RANGES["Triglycerides"]),
                "HDL_NORM": normalize_range(vitals.get("HDL", 50.0), *PROFILE_NUMERIC_RANGES["HDL"]),
                "LDL_NORM": normalize_range(vitals.get("LDL", 110.0), *PROFILE_NUMERIC_RANGES["LDL"]),
            }
        )

    return pd.DataFrame(patient_rows).set_index("PATIENT")


def run_etl_and_training():
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    force_rebuild_snapshot = os.getenv("NAVAVEJ_REBUILD_SNAPSHOT", "").lower() in {"1", "true", "yes"}
    snapshot, snapshot_path, created = load_or_create_snapshot(force_rebuild=force_rebuild_snapshot)
    packages_df = load_packages_dataframe()

    action_text = "สร้างใหม่" if created else "โหลดของเดิม"
    print(f"[1/3] Dataset Snapshot: {action_text} ที่ {snapshot_path}")

    txn_df = snapshot["transactions"].copy()
    patient_profiles_df = snapshot["patient_profiles"].copy()

    if txn_df.empty:
        raise RuntimeError("ไม่สามารถสร้าง transaction สำหรับเทรนโมเดลได้")

    if packages_df.empty:
        package_codes = sorted(txn_df["PACKAGE"].unique())
        packages_df = pd.DataFrame({"code": package_codes, "name": package_codes})

    print("[2/3] AI Training: สร้าง collaborative filtering matrix จาก snapshot กลาง...")
    matrix = txn_df.pivot_table(index="PATIENT", columns="PACKAGE", values="COUNT", fill_value=0)
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
    patient_features_df = build_patient_features(patient_profiles_df)

    print(f"[3/3] Save Model Artifact: บันทึกโมเดลไว้ที่ {MODEL_OUTPUT_PATH}...")
    artifacts = {
        "cf_matrix": item_sim_df,
        "transactions": txn_df,
        "packages": packages_df,
        "patient_features": patient_features_df,
        "patient_package_matrix": matrix,
        "patient_profiles": patient_profiles_df,
        "cold_start_patients": snapshot["cold_start_patients"],
        "existing_train_patients": snapshot["existing_train_patients"],
        "existing_test_patients": snapshot["existing_test_patients"],
        "dataset_snapshot_path": str(snapshot_path),
        "snapshot_metadata": snapshot["metadata"],
    }
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    mlflow_experiment = os.getenv("NAVAVEJ_MLFLOW_EXPERIMENT", "navavej-recommender-benchmark")
    mlflow, run_context = start_run_if_available(
        run_name="train_hybrid_recommender",
        experiment_name=mlflow_experiment,
    )
    with run_context:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "model_name": "hybrid_recommender",
                    "artifact_type": "training",
                    "snapshot_rebuilt": created,
                    "snapshot_seed": snapshot["metadata"]["seed"],
                    "cold_start_target": snapshot["metadata"]["cold_start_target"],
                    "age_cap": PROFILE_NUMERIC_RANGES["AGE"][1],
                    "total_patients": snapshot["metadata"]["total_patients"],
                    "existing_patients": snapshot["metadata"]["existing_patients"],
                    "feature_count": int(len(patient_features_df.columns)),
                    "package_catalog_size": int(len(packages_df)),
                }
            )
            mlflow.set_tags(
                {
                    "model_family": "hybrid",
                    "stage": "training",
                    "dataset_type": "snapshot",
                }
            )
            mlflow.log_metrics(
                {
                    "transactions_count": float(len(txn_df)),
                    "patient_profile_count": float(len(patient_profiles_df)),
                    "cold_start_count": float(len(snapshot["cold_start_patients"])),
                    "existing_train_count": float(len(snapshot["existing_train_patients"])),
                    "existing_test_count": float(len(snapshot["existing_test_patients"])),
                    "cf_matrix_package_count": float(len(item_sim_df.columns)),
                }
            )
            mlflow.log_artifact(str(Path(snapshot_path)))
            mlflow.log_artifact(str(MODEL_OUTPUT_PATH))
            log_dict_artifact(
                mlflow,
                {
                    "snapshot_path": str(snapshot_path),
                    "model_output_path": str(MODEL_OUTPUT_PATH),
                    "snapshot_metadata": snapshot["metadata"],
                    "feature_columns": list(patient_features_df.columns),
                },
                "training_summary.json",
            )
        else:
            print("MLflow not installed; skipped MLflow logging for training run.")

    print("ระบบอัปเดตโมเดลเสร็จแล้ว และทุกโมเดลสามารถใช้ snapshot เดียวกันเพื่อเทียบผลได้")


if __name__ == "__main__":
    run_etl_and_training()
