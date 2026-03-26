import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_ENCOUNTERS_PATH, DEFAULT_SNAPSHOT_PATH, resolve_path
from src.simulation_rules import (
    assign_chronic_packages,
    assign_general_screening_packages,
    assign_male_packages,
    assign_pediatric_packages,
    assign_senior_packages,
    assign_wellness_packages,
    assign_women_packages,
    midpoint,
    safe_float,
)


SNAPSHOT_SEED = 42
DEFAULT_COLD_START_TARGET = 150


def normalize_age(raw_age):
    return max(0, min(int(raw_age), 100))


def _condition_aware_ranges(conditions: str, age: int, gender: str) -> dict:
    """Return per-vital (min, max) fallback ranges tuned to the patient's profile.

    Priority order when multiple conditions overlap:
    - Diseased range wins over normal range for the affected vital.
    - Ranges are still bounded within physiologically plausible limits.
    - This only affects the *fallback* path (no real measurement found).
    """
    # Start with healthy-population defaults
    ranges = {
        "BMI":          (18.5, 24.9),
        "Glucose":      (70.0, 99.0),
        "SysBP":        (100.0, 129.0),
        "DiaBP":        (60.0, 84.0),
        "Cholesterol":  (150.0, 199.0),
        "HeartRate":    (55.0,  90.0),
        "RespRate":     (12.0,  18.0),
        "BodyHeight":   (150.0, 185.0),
        "BodyWeight":   (50.0,  85.0),
        "HbA1c":        (4.0,   5.6),
        "eGFR":         (75.0, 120.0),
        "Triglycerides":(70.0, 149.0),
        "HDL":          (50.0,  80.0),
        "LDL":          (50.0, 129.0),
    }

    has = lambda kw: kw in conditions  # noqa: E731  (quick inline helper)

    # --- Diabetes / pre-diabetes -----------------------------------------
    # NOTE: Check prediabetes first — "prediabetes" contains "diabetes" as a
    # substring, so ordering matters to avoid false positive matching.
    if has("prediabetes"):
        ranges["Glucose"]  = (100.0, 125.0)
        ranges["HbA1c"]   = (5.7,   6.4)
    elif has("diabetes"):
        ranges["Glucose"]  = (130.0, 280.0)
        ranges["HbA1c"]   = (6.5,   12.0)

    # --- Hypertension -------------------------------------------------------
    if has("hypertension"):
        ranges["SysBP"]  = (140.0, 180.0)
        ranges["DiaBP"]  = (90.0,  110.0)

    # --- Hyperlipidemia / lipid disorders -----------------------------------
    if has("hyperlipidemia") or has("hypercholesterolemia"):
        ranges["Cholesterol"]   = (220.0, 310.0)
        ranges["LDL"]           = (140.0, 220.0)
        ranges["Triglycerides"] = (175.0, 400.0)
        ranges["HDL"]           = (25.0,   45.0)

    # --- Obesity / overweight -----------------------------------------------
    if has("obesity"):
        ranges["BMI"]        = (30.0, 40.0)
        ranges["BodyWeight"] = (90.0, 160.0)
    elif has("overweight"):
        ranges["BMI"]        = (25.0, 29.9)
        ranges["BodyWeight"] = (75.0,  95.0)

    # --- Kidney / renal disease ---------------------------------------------
    if has("kidney") or has("renal") or has("nephro"):
        ranges["eGFR"] = (10.0, 59.0)

    # --- Coronary / heart disease -------------------------------------------
    if has("coronary") or has("heart failure") or has("myocardial"):
        ranges["HeartRate"]  = (55.0, 110.0)
        ranges["Cholesterol"] = (max(ranges["Cholesterol"][0], 200.0), 280.0)
        ranges["LDL"]         = (max(ranges["LDL"][0], 120.0), 200.0)

    # --- Stroke / cerebrovascular -------------------------------------------
    if has("stroke") or has("cerebrovascular"):
        ranges["SysBP"] = (max(ranges["SysBP"][0], 140.0), 185.0)
        ranges["DiaBP"] = (max(ranges["DiaBP"][0],  85.0), 110.0)

    # --- Age-related adjustments --------------------------------------------
    if age >= 60:
        # Mild eGFR decline with age even without kidney disease
        cur_lo, cur_hi = ranges["eGFR"]
        ranges["eGFR"] = (max(cur_lo - 15.0, 10.0), max(cur_hi - 15.0, 30.0))
        # Blood pressure tends to rise with age
        bp_lo, bp_hi = ranges["SysBP"]
        ranges["SysBP"] = (min(bp_lo + 10.0, 160.0), min(bp_hi + 10.0, 185.0))

    # --- Pregnancy (BMI / weight shift) ------------------------------------
    if has("pregnancy") or has("prenatal"):
        ranges["BodyWeight"] = (60.0, 95.0)
        ranges["Glucose"]    = (max(ranges["Glucose"][0], 80.0), min(ranges["Glucose"][1] + 20.0, 130.0))

    return ranges


def _build_patient_snapshot(encounters_df, cold_start_target):
    np.random.seed(SNAPSHOT_SEED)
    random.seed(SNAPSHOT_SEED)

    patients_df = encounters_df.drop_duplicates(subset=["PATIENT"]).copy()
    cold_start_rng = random.Random(SNAPSHOT_SEED)
    cold_start_count = min(cold_start_target, len(patients_df))
    cold_start_patients = set(cold_start_rng.sample(patients_df["PATIENT"].tolist(), cold_start_count))

    transactions = []
    patient_profiles = []

    for _, row in patients_df.iterrows():
        patient_id = row["PATIENT"]

        raw_conds = str(row["CONDITIONS"]).split(" | ")
        exclude_words = [
            "employment",
            "housing",
            "certificate",
            "social",
            "refugee",
            "stress",
            "income",
            "education",
            "finding",
            "not in labor",
            "activity",
            "unemployment",
        ]
        real_conds = [c for c in raw_conds if not any(w in c.lower() for w in exclude_words) and c != "nan"]
        conditions_clean = " | ".join(real_conds) if real_conds else "ไม่มีโรคประจำตัว"
        conditions = conditions_clean.lower()

        gender = row["PATIENT_GENDER"]
        try:
            age = normalize_age(datetime.now().year - int(str(row["PATIENT_BIRTHDATE"])[:4]))
        except Exception:
            age = 35

        bought_items = set()
        patient_records = encounters_df[encounters_df["PATIENT"] == patient_id]

        cond_ranges = _condition_aware_ranges(conditions, age, gender)

        def safe_get(col, vital_key, min_val, max_val):
            """Return the latest real measurement, or a condition-aware random fallback."""
            if col in patient_records.columns:
                valid_values = patient_records[patient_records[col].notna()][col]
                if not valid_values.empty:
                    val = valid_values.iloc[-1]
                    if isinstance(val, str) and " " in val:
                        try:
                            val = float(val.split(" ")[0])
                        except Exception:
                            pass
                    if isinstance(val, (int, float, np.number)):
                        return str(round(float(val), 1))
                    return str(val)
            # No real measurement — use a deterministic midpoint from the
            # condition-aware range so snapshots remain reproducible.
            lo, hi = cond_ranges.get(vital_key, (min_val, max_val))
            return str(midpoint(lo, hi))

        vitals = {
            "BMI":          safe_get("body_mass_index",                              "BMI",          18.5, 29.9),
            "Glucose":      safe_get("glucose",                                      "Glucose",      70.0, 99.0),
            "SysBP":        safe_get("systolic_blood_pressure",                      "SysBP",       100.0, 129.0),
            "DiaBP":        safe_get("diastolic_blood_pressure",                     "DiaBP",        60.0, 84.0),
            "Cholesterol":  safe_get("total_cholesterol",                            "Cholesterol", 150.0, 199.0),
            "HeartRate":    safe_get("heart_rate",                                   "HeartRate",    55.0, 90.0),
            "RespRate":     safe_get("respiratory_rate",                             "RespRate",     12.0, 18.0),
            "BodyHeight":   safe_get("body_height",                                  "BodyHeight",  150.0, 185.0),
            "BodyWeight":   safe_get("body_weight",                                  "BodyWeight",   50.0, 85.0),
            "HbA1c":        safe_get("hemoglobin_a1c_hemoglobin_total_in_blood",     "HbA1c",        4.0,  5.6),
            "eGFR":         safe_get("estimated_glomerular_filtration_rate",         "eGFR",        75.0, 120.0),
            "Triglycerides":safe_get("triglycerides",                                "Triglycerides",70.0, 149.0),
            "HDL":          safe_get("high_density_lipoprotein_cholesterol",         "HDL",          50.0, 80.0),
            "LDL":          safe_get("low_density_lipoprotein_cholesterol",          "LDL",          50.0, 129.0),
        }

        bmi = safe_float(vitals["BMI"], 25.0)
        glucose = safe_float(vitals["Glucose"], 100.0)
        sys_bp = safe_float(vitals["SysBP"], 120.0)
        dia_bp = safe_float(vitals["DiaBP"], 80.0)
        cholesterol = safe_float(vitals["Cholesterol"], 180.0)
        hba1c = safe_float(vitals["HbA1c"], 5.6)
        egfr = safe_float(vitals["eGFR"], 90.0)
        triglycerides = safe_float(vitals["Triglycerides"], 150.0)
        ldl = safe_float(vitals["LDL"], 110.0)

        if gender == "F":
            bought_items.update(assign_women_packages(age, conditions, bmi, glucose))

        if gender == "M":
            bought_items.update(assign_male_packages(age, conditions, sys_bp, cholesterol))

        bought_items.update(assign_pediatric_packages(age, bmi, glucose))
        bought_items.update(assign_senior_packages(age, conditions, sys_bp))
        bought_items.update(
            assign_chronic_packages(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr)
        )
        bought_items.update(assign_wellness_packages(age, bmi, glucose, cholesterol, triglycerides))
        bought_items.update(assign_general_screening_packages(age, bmi, sys_bp, cholesterol))

        customer_type = "new" if patient_id in cold_start_patients else "existing"
        effective_package_count = 0 if customer_type == "new" else len(bought_items)
        patient_profiles.append(
            {
                "PATIENT": patient_id,
                "AGE": age,
                "GENDER": gender,
                "CONDITIONS": conditions_clean,
                "VITALS": vitals,
                "CUSTOMER_TYPE": customer_type,
                "PACKAGE_COUNT": effective_package_count,
            }
        )

        if customer_type == "new":
            continue

        for pkg in bought_items:
            transactions.append(
                {
                    "PATIENT": patient_id,
                    "PACKAGE": pkg,
                    "AGE": age,
                    "GENDER": gender,
                    "COUNT": 1,
                    "CONDITIONS": conditions_clean,
                    "VITALS": vitals,
                }
            )

    txn_df = pd.DataFrame(transactions)
    if txn_df.empty:
        raise RuntimeError("ไม่สามารถสร้าง transaction dataset จาก snapshot ได้")

    patient_profiles_df = pd.DataFrame(patient_profiles).set_index("PATIENT")
    existing_patients = sorted(txn_df["PATIENT"].unique().tolist())
    train_patients, test_patients = train_test_split(
        existing_patients,
        test_size=0.2,
        random_state=SNAPSHOT_SEED,
    )

    return {
        "transactions": txn_df,
        "patient_profiles": patient_profiles_df,
        "cold_start_patients": sorted(cold_start_patients),
        "existing_train_patients": sorted(train_patients),
        "existing_test_patients": sorted(test_patients),
        "metadata": {
            "seed": SNAPSHOT_SEED,
            "cold_start_target": cold_start_target,
            "total_patients": int(len(patient_profiles_df)),
            "existing_patients": int(len(existing_patients)),
        },
    }


def load_or_create_snapshot(force_rebuild=False, cold_start_target=DEFAULT_COLD_START_TARGET):
    snapshot_path = resolve_path("NAVAVEJ_SNAPSHOT_PATH", DEFAULT_SNAPSHOT_PATH)
    if snapshot_path.exists() and not force_rebuild:
        with open(snapshot_path, "rb") as f:
            return pickle.load(f), snapshot_path, False

    encounters_path = resolve_path("NAVAVEJ_ENCOUNTERS_PATH", DEFAULT_ENCOUNTERS_PATH)
    encounters_df = pd.read_csv(encounters_path, low_memory=False)
    snapshot = _build_patient_snapshot(encounters_df, cold_start_target=cold_start_target)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)
    return snapshot, snapshot_path, True


def main():
    force_rebuild = os.getenv("NAVAVEJ_REBUILD_SNAPSHOT", "").lower() in {"1", "true", "yes"}
    snapshot, snapshot_path, created = load_or_create_snapshot(force_rebuild=force_rebuild)
    action = "สร้างใหม่" if created else "โหลดของเดิม"
    print(f"{action} dataset snapshot ที่ {snapshot_path}")
    print(
        f"Patients ทั้งหมด: {snapshot['metadata']['total_patients']} | "
        f"Existing: {snapshot['metadata']['existing_patients']} | "
        f"Cold-start: {len(snapshot['cold_start_patients'])}"
    )


if __name__ == "__main__":
    main()
