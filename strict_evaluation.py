import io
import os
import random
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset_snapshot import load_or_create_snapshot
from src.inference_engine import (
    PROFILE_NUMERIC_RANGES,
    apply_clinical_rules,
    has_any_keyword,
    normalize_range,
)
from src.mlflow_utils import log_dict_artifact, start_run_if_available

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def build_patient_features(txn_df):
    patient_rows = []
    for patient_id, history in txn_df.groupby("PATIENT"):
        row = history.iloc[0]
        conditions = str(row.get("CONDITIONS", "")).lower()
        vitals = row.get("VITALS", {}) or {}

        patient_rows.append(
            {
                "PATIENT": patient_id,
                "AGE_NORM": normalize_range(row["AGE"], *PROFILE_NUMERIC_RANGES["AGE"]),
                "PACKAGE_COUNT_NORM": normalize_range(len(history), *PROFILE_NUMERIC_RANGES["PACKAGE_COUNT"]),
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


def normalize_scores(scores):
    positive_scores = scores[scores > 0]
    if positive_scores.empty:
        return pd.Series(0.0, index=scores.index)

    max_score = positive_scores.max()
    return scores.clip(lower=0) / max_score if max_score else pd.Series(0.0, index=scores.index)


def build_cf_scores(cf_matrix, purchased_pkgs):
    scores = pd.Series(0.0, index=cf_matrix.columns)
    for pkg in purchased_pkgs:
        if pkg in cf_matrix.index:
            scores += cf_matrix[pkg]

    return scores.drop(purchased_pkgs, errors="ignore")


def build_profile_scores(train_df, patient_features, patient_id, purchased_pkgs, all_packages, top_k=5):
    if patient_id not in patient_features.index or patient_features.empty:
        return pd.Series(0.0, index=all_packages)

    patient_matrix = train_df.pivot_table(index="PATIENT", columns="PACKAGE", values="COUNT", fill_value=0)
    aligned_features = patient_features.loc[patient_features.index.intersection(patient_matrix.index)]
    if aligned_features.empty or patient_id not in aligned_features.index or len(aligned_features) <= 1:
        return pd.Series(0.0, index=all_packages)

    target_vector = aligned_features.loc[[patient_id]]
    similarity_values = cosine_similarity(target_vector, aligned_features)[0]
    similarity_series = pd.Series(similarity_values, index=aligned_features.index).drop(patient_id, errors="ignore")
    similarity_series = similarity_series[similarity_series > 0].sort_values(ascending=False).head(top_k)

    if similarity_series.empty:
        return pd.Series(0.0, index=all_packages)

    similar_patient_packages = patient_matrix.loc[similarity_series.index]
    weighted_scores = similar_patient_packages.mul(similarity_series, axis=0).sum(axis=0)
    weighted_scores = weighted_scores / similarity_series.sum()

    return weighted_scores.reindex(all_packages, fill_value=0.0).drop(purchased_pkgs, errors="ignore")


def predict_top_packages_strict(cf_matrix, train_df, patient_features, patient_id, purchased_pkgs, age, gender):
    cf_scores = build_cf_scores(cf_matrix, purchased_pkgs)
    profile_scores = build_profile_scores(train_df, patient_features, patient_id, purchased_pkgs, cf_matrix.columns, top_k=5)
    scores = (0.7 * normalize_scores(cf_scores)) + (0.3 * normalize_scores(profile_scores))
    # Use the shared authoritative clinical rules from inference_engine
    scores = apply_clinical_rules(scores, age, gender)
    valid_scores = scores[scores > -50.0]
    return valid_scores.sort_values(ascending=False).head(3).index.tolist()


def run_evaluation():
    snapshot, snapshot_path, created = load_or_create_snapshot(force_rebuild=False)
    txn_df = snapshot["transactions"].copy()
    train_patients = snapshot["existing_train_patients"]
    test_patients = snapshot["existing_test_patients"]

    train_df = txn_df[txn_df["PATIENT"].isin(train_patients)]
    test_df = txn_df[txn_df["PATIENT"].isin(test_patients)]

    print(f"🏥 Total Patients in Dataset: {len(txn_df['PATIENT'].unique())}")
    print(f"📚 Training Patients (80% for CF Matrix): {len(train_patients)}")
    print(f"📝 Testing Patients (20% Held-out Unseen): {len(test_patients)}")
    print(f"🧊 Snapshot Source: {snapshot_path} ({'สร้างใหม่' if created else 'ใช้ชุดเดิม'})")

    matrix = train_df.pivot_table(index="PATIENT", columns="PACKAGE", values="COUNT", fill_value=0)
    cf_matrix = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)
    patient_features = build_patient_features(train_df)
    all_packages = txn_df["PACKAGE"].unique()

    for pkg in all_packages:
        if pkg not in cf_matrix.columns:
            cf_matrix[pkg] = 0.0
            cf_matrix.loc[pkg] = 0.0

    hits = 0
    violations = 0
    valid_test_cases = 0
    hidden_packages_total = 0

    print("กำลังไล่จำลองทดสอบเดาใจคนไข้กลุ่ม 20% ที่เหลือ (ซ่อน 2 แพ็กเกจ เหลือให้เห็น 1)...")
    for patient in test_patients:
        history = test_df[test_df["PATIENT"] == patient]
        pkgs = sorted(history["PACKAGE"].tolist())

        if len(pkgs) >= 3:
            valid_test_cases += 1
            age = history.iloc[0]["AGE"]
            gender = history.iloc[0]["GENDER"]

            patient_rng = random.Random(f"snapshot-eval:{patient}")
            sampled_pkgs = patient_rng.sample(pkgs, 3)
            visible_pkg = sampled_pkgs[0]
            hidden_pkgs = sampled_pkgs[1:]
            known_pkgs = [visible_pkg]
            hidden_packages_total += len(hidden_pkgs)

            synthetic_patient_row = history.iloc[0].copy()
            synthetic_patient_row["PATIENT"] = patient
            synthetic_patient_row["PACKAGE"] = known_pkgs[0]
            synthetic_df = pd.concat([train_df, pd.DataFrame([synthetic_patient_row])], ignore_index=True)
            synthetic_features = build_patient_features(synthetic_df)

            top_3 = predict_top_packages_strict(cf_matrix, synthetic_df, synthetic_features, patient, known_pkgs, age, gender)

            for hidden_pkg in hidden_pkgs:
                if hidden_pkg in top_3:
                    hits += 1

            # Violation check uses same rule sets from inference_engine (via apply_clinical_rules)
            from src.inference_engine import (
                CLINICAL_RULE_MEN_PACKAGES,
                CLINICAL_RULE_MIN_AGE_PACKAGES,
                CLINICAL_RULE_PEDIATRIC_PACKAGES,
                CLINICAL_RULE_WOMEN_PACKAGES,
            )
            for predicted_pkg in top_3:
                if gender == "M" and predicted_pkg in CLINICAL_RULE_WOMEN_PACKAGES:
                    violations += 1
                if gender == "F" and predicted_pkg in CLINICAL_RULE_MEN_PACKAGES:
                    violations += 1
                if age > 14 and predicted_pkg in CLINICAL_RULE_PEDIATRIC_PACKAGES:
                    violations += 1
                if age < 20 and predicted_pkg in CLINICAL_RULE_MIN_AGE_PACKAGES:
                    violations += 1

    print("\n==================================================")
    print("📊 [STRICT EVALUATION] Fixed Snapshot Split (No Leakage) 📊")
    print("==================================================")
    if valid_test_cases == 0:
        print("ไม่พบเคสทดสอบ")
        hit_rate = 0.0
    else:
        hit_rate = (hits / hidden_packages_total) * 100 if hidden_packages_total else 0.0
        print(f"🎯 อัตราเดาใจถูกต่อแพ็กเกจที่ถูกซ่อน (True Predictive Hit Rate) : {hit_rate:.2f}% (แม่น {hits}/{hidden_packages_total} แพ็กเกจ)")
        print(f"🧪 จำนวนเคสทดสอบ: {valid_test_cases} คน (เห็น 1 แพ็กเกจ, ซ่อน 2 แพ็กเกจ)")
        print(f"⚕️ ตรวจสอบการผิดกฎการแพทย์ (Medical Violations) : {violations} ครั้ง")
    print("==================================================")

    mlflow_experiment = os.getenv("NAVAVEJ_MLFLOW_EXPERIMENT", "navavej-recommender-benchmark")
    mlflow, run_context = start_run_if_available(
        run_name="evaluate_hybrid_recommender",
        experiment_name=mlflow_experiment,
    )
    with run_context:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "model_name": "hybrid_recommender",
                    "artifact_type": "evaluation",
                    "evaluation_mode": "fixed_snapshot_split_random_holdout",
                    "snapshot_path": str(snapshot_path),
                    "snapshot_seed": snapshot["metadata"]["seed"],
                    "hide_count": 2,
                    "visible_count": 1,
                    "top_k_eval": 3,
                    "cf_weight": 0.7,
                    "profile_weight": 0.3,
                    "profile_top_k": 5,
                }
            )
            mlflow.set_tags(
                {
                    "model_family": "hybrid",
                    "stage": "evaluation",
                    "dataset_type": "snapshot",
                }
            )
            mlflow.log_metrics(
                {
                    "hit_rate": hit_rate,
                    "hits": float(hits),
                    "hidden_packages_total": float(hidden_packages_total),
                    "valid_test_cases": float(valid_test_cases),
                    "medical_violations": float(violations),
                    "training_patients": float(len(train_patients)),
                    "testing_patients": float(len(test_patients)),
                }
            )
            log_dict_artifact(
                mlflow,
                {
                    "snapshot_path": str(snapshot_path),
                    "results": {
                        "hit_rate": hit_rate,
                        "hits": hits,
                        "hidden_packages_total": hidden_packages_total,
                        "valid_test_cases": valid_test_cases,
                        "medical_violations": violations,
                    },
                },
                "evaluation_summary.json",
            )
        else:
            print("MLflow not installed; skipped MLflow logging for evaluation run.")


if __name__ == "__main__":
    run_evaluation()
