from datetime import datetime
import io
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DEFAULT_ENCOUNTERS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PACKAGES_PATH,
    resolve_path,
)
from src.simulation_rules import (
    clamp_probability,
    compute_chronic_probability,
    compute_general_screening_probability,
    compute_kid_package_probability,
    compute_male_health_probability,
    compute_maternity_probability,
    compute_wellness_probability,
    compute_women_health_probability,
    safe_float,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MERGED_ENCOUNTERS_PATH = resolve_path("NAVAVEJ_ENCOUNTERS_PATH", DEFAULT_ENCOUNTERS_PATH)
NAVAVEJ_PACKAGES_PATH = resolve_path("NAVAVEJ_PACKAGES_PATH", DEFAULT_PACKAGES_PATH)
MODEL_OUTPUT_PATH = resolve_path("NAVAVEJ_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_OUTPUT_DIR = MODEL_OUTPUT_PATH.parent

PROFILE_NUMERIC_RANGES = {
    "AGE": (0.0, 100.0),
    "PACKAGE_COUNT": (1.0, 12.0),
    "BMI": (15.0, 40.0),
    "Glucose": (60.0, 200.0),
    "SysBP": (80.0, 200.0),
    "DiaBP": (50.0, 130.0),
    "Cholesterol": (100.0, 320.0),
    "HeartRate": (40.0, 140.0),
    "RespRate": (8.0, 30.0),
    "BodyHeight": (45.0, 210.0),
    "BodyWeight": (2.0, 180.0),
    "HbA1c": (4.0, 12.0),
    "eGFR": (15.0, 130.0),
    "Triglycerides": (50.0, 400.0),
    "HDL": (20.0, 100.0),
    "LDL": (40.0, 220.0),
}


def load_packages_dataframe():
    if NAVAVEJ_PACKAGES_PATH.exists():
        return pd.read_csv(NAVAVEJ_PACKAGES_PATH)

    print(f"⚠️ ไม่พบไฟล์ package catalog ที่ {NAVAVEJ_PACKAGES_PATH} จะ fallback เป็น code=ชื่อชั่วคราวสำหรับเดโม")
    return pd.DataFrame(columns=["code", "name"])


def normalize_range(value, min_val, max_val):
    clipped = min(max(float(value), min_val), max_val)
    return (clipped - min_val) / (max_val - min_val)


def has_any_keyword(text, keywords):
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0

def build_patient_features(txn_df):
    patient_rows = []

    for patient_id, history in txn_df.groupby("PATIENT"):
        row = history.iloc[0]
        vitals = row.get("VITALS", {}) or {}
        conditions = str(row.get("CONDITIONS", "")).lower()

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

def run_etl_and_training():
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    print("[1/3] 📥 Data Extraction: ดึงข้อมูลประวัติคนไข้จากระบบ HIS...")
    encounters_df = pd.read_csv(MERGED_ENCOUNTERS_PATH, low_memory=False)
    packages_df = load_packages_dataframe()
    
    np.random.seed(42)
    random.seed(42)
    patients_df = encounters_df.drop_duplicates(subset=['PATIENT']).copy()
    transactions = []
    
    for _, row in patients_df.iterrows():
        patient_id = row['PATIENT']
        
        # กรองโรคที่เป็น Social Determinants (SDOH) ออก เพื่อความสมจริงในแอป
        raw_conds = str(row['CONDITIONS']).split(' | ')
        exclude_words = ['employment', 'housing', 'certificate', 'social', 'refugee', 'stress', 'income', 'education', 'finding', 'not in labor', 'activity', 'unemployment']
        real_conds = [c for c in raw_conds if not any(w in c.lower() for w in exclude_words) and c != 'nan']
        conditions_clean = " | ".join(real_conds) if real_conds else "ไม่มีโรคประจำตัว"
        conditions = conditions_clean.lower()
        
        gender = row['PATIENT_GENDER']
        try: age = datetime.now().year - int(str(row['PATIENT_BIRTHDATE'])[:4])
        except: age = 35
            
        bought_items = set()
        
        # ดึงค่า Vitals/Labs จาก "ทุกประวัติ" ของคนไข้คนนี้ (เผื่อแวะมาหลายรอบแล้วบางรอบไม่ได้เจาะเลือด)
        patient_records = encounters_df[encounters_df['PATIENT'] == patient_id]
        
        def safe_get(col, min_val, max_val):
            if col in patient_records.columns:
                valid_values = patient_records[patient_records[col].notna()][col]
                if not valid_values.empty:
                    val = valid_values.iloc[-1] # เอาค่าล่าสุด
                    if isinstance(val, str) and ' ' in val: # ดึงตัวเลขจาก "120.0 mm[Hg]"
                        try: val = float(val.split(' ')[0])
                        except: pass
                    if isinstance(val, (int, float, np.number)):
                        return str(round(float(val), 1))
                    return str(val)
            # เติมตัวเลขจำลองสมจริง (Mock) ถ้าประวัติคนไข้อันไหนไม่มีผลแล็บ เพื่อให้ Demo สวยงาม
            return str(round(np.random.uniform(min_val, max_val), 1))
            
        vitals = {
            'BMI': safe_get('body_mass_index', 18.5, 29.9),
            'Glucose': safe_get('glucose', 80.0, 125.0),
            'SysBP': safe_get('systolic_blood_pressure', 110.0, 140.0),
            'DiaBP': safe_get('diastolic_blood_pressure', 70.0, 90.0),
            'Cholesterol': safe_get('total_cholesterol', 150.0, 240.0),
            'HeartRate': safe_get('heart_rate', 55.0, 105.0),
            'RespRate': safe_get('respiratory_rate', 12.0, 22.0),
            'BodyHeight': safe_get('body_height', 120.0, 185.0),
            'BodyWeight': safe_get('body_weight', 35.0, 110.0),
            'HbA1c': safe_get('hemoglobin_a1c_hemoglobin_total_in_blood', 4.8, 9.5),
            'eGFR': safe_get('estimated_glomerular_filtration_rate', 40.0, 110.0),
            'Triglycerides': safe_get('triglycerides', 70.0, 260.0),
            'HDL': safe_get('high_density_lipoprotein_cholesterol', 30.0, 80.0),
            'LDL': safe_get('low_density_lipoprotein_cholesterol', 60.0, 180.0),
        }

        bmi = safe_float(vitals['BMI'], 25.0)
        glucose = safe_float(vitals['Glucose'], 100.0)
        sys_bp = safe_float(vitals['SysBP'], 120.0)
        dia_bp = safe_float(vitals['DiaBP'], 80.0)
        cholesterol = safe_float(vitals['Cholesterol'], 180.0)
        hba1c = safe_float(vitals['HbA1c'], 5.6)
        egfr = safe_float(vitals['eGFR'], 90.0)
        triglycerides = safe_float(vitals['Triglycerides'], 150.0)
        ldl = safe_float(vitals['LDL'], 110.0)
        
        # 1. Maternity & Female Health
        if gender == 'F':
            maternity_prob = compute_maternity_probability(age, conditions)
            if 18 <= age <= 45 and ('pregnancy' in conditions or 'prenatal' in conditions):
                if np.random.rand() < maternity_prob: bought_items.add('NVV-PK-0007')
                if np.random.rand() < max(maternity_prob - 0.10, 0.15): bought_items.add('NVV-PK-0061')

            women_health_prob = compute_women_health_probability(age, conditions, bmi, glucose)
            if 15 <= age <= 45 and np.random.rand() < max(women_health_prob - 0.08, 0.12): bought_items.add('NVV-PK-0059')
            if age >= 30 and np.random.rand() < women_health_prob: bought_items.add('NVV-PK-0035')
            if age >= 30 and np.random.rand() < max(women_health_prob - 0.18, 0.08): bought_items.add('NVV-PK-0036')
            if age >= 35 and np.random.rand() < max(women_health_prob + 0.08, 0.25): bought_items.add('NVV-PK-0015')
            if age >= 25 and np.random.rand() < 0.05: bought_items.add('NVV-PK-0028')

        # 2. Male Health
        if gender == 'M':
            male_health_prob = compute_male_health_probability(age, conditions, sys_bp, cholesterol)
            if age >= 40 and np.random.rand() < male_health_prob: bought_items.add('NVV-PK-0014')
            if 40 <= age <= 55 and np.random.rand() < max(male_health_prob - 0.10, 0.10): bought_items.add('NVV-PK-0040')

        # 3. Kids & Teens
        kid_prob = compute_kid_package_probability(age, bmi, glucose)
        if age <= 14:
            if np.random.rand() < kid_prob: bought_items.add('NVV-PK-0092')
            if np.random.rand() < max(kid_prob - 0.12, 0.12): bought_items.add('NVV-PK-0084')
            if np.random.rand() < max(kid_prob - 0.28, 0.04): bought_items.add('NVV-PK-0082')

        # 4. Seniors & Neurology
        if age >= 55:
            senior_prob = clamp_probability(0.24 + (0.10 if age >= 65 else 0.0) + (0.05 if 'stroke' in conditions else 0.0))
            if np.random.rand() < senior_prob: bought_items.add('NVV-PK-0017')
            if np.random.rand() < max(senior_prob - 0.04, 0.12): bought_items.add('NVV-PK-0018')
            if np.random.rand() < max(senior_prob - 0.10, 0.08): bought_items.add('NVV-PK-0003')

        # 5. Chronic & Internal Med
        chronic_prob = compute_chronic_probability(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr)
        if age >= 50 or 'diabetes' in conditions or 'hypertension' in conditions or 'hyperlipidemia' in conditions:
            if np.random.rand() < chronic_prob: bought_items.add('NVV-PK-0002')
            if np.random.rand() < max(chronic_prob - 0.08, 0.12): bought_items.add('NVV-PK-0078')
            if np.random.rand() < max(chronic_prob - 0.04, 0.14): bought_items.add('NVV-PK-0081')
            if np.random.rand() < max(chronic_prob - 0.14, 0.08): bought_items.add('NVV-PK-0064')

        # 6. Wellness & Anti-Aging (IV Drips)
        wellness_prob = compute_wellness_probability(age, bmi, glucose, cholesterol, triglycerides)
        if 25 <= age <= 60:
            if np.random.rand() < wellness_prob: bought_items.add('NVV-PK-0046')
            if np.random.rand() < max(wellness_prob - 0.01, 0.08): bought_items.add('NVV-PK-0047')
            if np.random.rand() < max(wellness_prob - 0.05, 0.05): bought_items.add('NVV-PK-0083')
            if np.random.rand() < max(wellness_prob - 0.04, 0.05): bought_items.add('NVV-PK-0062')
            if np.random.rand() < max(wellness_prob - 0.08, 0.03): bought_items.add('NVV-PK-0048')

        # 7. General & Dental (ทุกเพศทุกวัย)
        general_prob = compute_general_screening_probability(age, bmi, sys_bp, cholesterol)
        if age >= 20:
            if np.random.rand() < max(general_prob + 0.10, 0.20): bought_items.add('NVV-PK-0031')
            if np.random.rand() < max(general_prob - 0.02, 0.10): bought_items.add('NVV-PK-0044')
            if np.random.rand() < general_prob: bought_items.add('NVV-PK-0004')
            if np.random.rand() < max(general_prob - 0.08, 0.06): bought_items.add('NVV-PK-0001')
            if np.random.rand() < max(general_prob - 0.12, 0.03): bought_items.add('NVV-PK-0098')
            if np.random.rand() < max(general_prob - 0.06, 0.05): bought_items.add('NVV-PK-0072')
            if np.random.rand() < max(general_prob - 0.07, 0.05): bought_items.add('NVV-PK-0045')

        for pkg in bought_items:
            transactions.append({
                'PATIENT': patient_id, 'PACKAGE': pkg, 
                'AGE': age, 'GENDER': gender, 'COUNT': 1, 
                'CONDITIONS': conditions_clean, 'VITALS': vitals
            })

    txn_df = pd.DataFrame(transactions)
    if txn_df.empty:
        raise RuntimeError("ไม่สามารถสร้าง transaction สำหรับเทรนโมเดลได้")

    if packages_df.empty:
        package_codes = sorted(txn_df["PACKAGE"].unique())
        packages_df = pd.DataFrame({"code": package_codes, "name": package_codes})
    
    print("[2/3] 🧠 AI Training: นำประวัติก้อนใหญ่เข้ากระบวนการ Collaborative Filtering...")
    matrix = txn_df.pivot_table(index='PATIENT', columns='PACKAGE', values='COUNT', fill_value=0)
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
    patient_features_df = build_patient_features(txn_df)
    
    print(f"[3/3] 💾 Save Model Artifact: บันทึกความรู้ทั้งหมดไว้ที่ {MODEL_OUTPUT_PATH}...")
    artifacts = {
        'cf_matrix': item_sim_df,
        'transactions': txn_df,       
        'packages': packages_df,
        'patient_features': patient_features_df,
        'patient_package_matrix': matrix,
    }
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print("✅ ระบบอัปเดตโมเดลประจำวันเสร็จสิ้นแล้ว! (Production Training Complete)")

if __name__ == "__main__":
    run_etl_and_training()
