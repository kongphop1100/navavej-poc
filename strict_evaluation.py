from datetime import datetime
import io
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_ENCOUNTERS_PATH, resolve_path

# บังคับคอนโซล Windows ให้พิมพ์ Emoji ออกมาได้โดยไม่ error cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MERGED_ENCOUNTERS_PATH = resolve_path("NAVAVEJ_ENCOUNTERS_PATH", DEFAULT_ENCOUNTERS_PATH)
encounters_df = pd.read_csv(MERGED_ENCOUNTERS_PATH, low_memory=False)

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


def normalize_range(value, min_val, max_val):
    clipped = min(max(float(value), min_val), max_val)
    return (clipped - min_val) / (max_val - min_val)


def has_any_keyword(text, keywords):
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0


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

np.random.seed(42)
random.seed(42)
patients_df = encounters_df.drop_duplicates(subset=['PATIENT']).copy()
transactions = []

for _, row in patients_df.iterrows():
    patient_id = row['PATIENT']
    conditions = str(row['CONDITIONS']).lower()
    gender = row['PATIENT_GENDER']
    try: age = datetime.now().year - int(str(row['PATIENT_BIRTHDATE'])[:4])
    except: age = 35
        
    bought_items = set()
    
    if gender == 'F' and 18 <= age <= 45 and ('pregnancy' in conditions or 'prenatal' in conditions):
        if np.random.rand() > 0.2: bought_items.add('NVV-PK-0007')
        if np.random.rand() > 0.4: bought_items.add('NVV-PK-0092')
            
    if gender == 'M' and age >= 40:
        if np.random.rand() > 0.5: bought_items.add('NVV-PK-0014')
    if gender == 'F' and age >= 35:
        if np.random.rand() > 0.4: bought_items.add('NVV-PK-0015')
            
    if age >= 50 or 'diabetes' in conditions or 'hypertension' in conditions:
        if np.random.rand() > 0.3: bought_items.add('NVV-PK-0002')
        if np.random.rand() > 0.4: bought_items.add('NVV-PK-0046')
        if np.random.rand() > 0.7: bought_items.add('NVV-PK-0003')

    if np.random.rand() > 0.6: bought_items.add('NVV-PK-0044')
    if np.random.rand() > 0.8: bought_items.add('NVV-PK-0001')

    vitals = {
        'BMI': round(np.random.uniform(18.5, 29.9), 1),
        'Glucose': round(np.random.uniform(80.0, 125.0), 1),
        'SysBP': round(np.random.uniform(110.0, 140.0), 1),
        'DiaBP': round(np.random.uniform(70.0, 90.0), 1),
        'Cholesterol': round(np.random.uniform(150.0, 240.0), 1),
        'HeartRate': round(np.random.uniform(55.0, 105.0), 1),
        'RespRate': round(np.random.uniform(12.0, 22.0), 1),
        'BodyHeight': round(np.random.uniform(120.0, 185.0), 1),
        'BodyWeight': round(np.random.uniform(35.0, 110.0), 1),
        'HbA1c': round(np.random.uniform(4.8, 9.5), 1),
        'eGFR': round(np.random.uniform(40.0, 110.0), 1),
        'Triglycerides': round(np.random.uniform(70.0, 260.0), 1),
        'HDL': round(np.random.uniform(30.0, 80.0), 1),
        'LDL': round(np.random.uniform(60.0, 180.0), 1),
    }
    
    for pkg in bought_items:
        transactions.append({
            'PATIENT': patient_id,
            'PACKAGE': pkg,
            'AGE': age,
            'GENDER': gender,
            'COUNT': 1,
            'CONDITIONS': conditions,
            'VITALS': vitals,
        })

txn_df = pd.DataFrame(transactions)

unique_patients = txn_df['PATIENT'].unique()
train_patients, test_patients = train_test_split(list(unique_patients), test_size=0.2, random_state=42)

train_df = txn_df[txn_df['PATIENT'].isin(train_patients)]
test_df = txn_df[txn_df['PATIENT'].isin(test_patients)]

print(f"🏥 Total Patients in Dataset: {len(unique_patients)}")
print(f"📚 Training Patients (80% for CF Matrix): {len(train_patients)}")
print(f"📝 Testing Patients (20% Held-out Unseen): {len(test_patients)}")

# สร้างหัวสมอง AI แบบไม่โกง (เรียนรู้จากกลุ่ม 80% แรกหน้าเดิม)
matrix = train_df.pivot_table(index='PATIENT', columns='PACKAGE', values='COUNT', fill_value=0)
cf_matrix = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)
patient_features = build_patient_features(train_df)
all_packages = txn_df['PACKAGE'].unique()

# อุดรอยรั่วเผื่อแพ็กเกจบางอันไม่เคยมีใครซื้อใน 80% แรกเลย (จะได้ไม่ Error ตอนเรียกใช้)
for pkg in all_packages:
    if pkg not in cf_matrix.columns:
        cf_matrix[pkg] = 0.0
        cf_matrix.loc[pkg] = 0.0

def build_cf_scores(cf_matrix, purchased_pkgs):
    scores = pd.Series(0.0, index=cf_matrix.columns)
    for pkg in purchased_pkgs:
        if pkg in cf_matrix.index:
            scores += cf_matrix[pkg]

    return scores.drop(purchased_pkgs, errors='ignore')


def build_profile_scores(train_df, patient_features, patient_id, purchased_pkgs, all_packages, top_k=5):
    if patient_id not in patient_features.index or patient_features.empty:
        return pd.Series(0.0, index=all_packages)

    patient_matrix = train_df.pivot_table(index='PATIENT', columns='PACKAGE', values='COUNT', fill_value=0)
    aligned_features = patient_features.loc[patient_features.index.intersection(patient_matrix.index)]
    if aligned_features.empty or patient_id not in aligned_features.index or len(aligned_features) <= 1:
        return pd.Series(0.0, index=all_packages)

    target_vector = aligned_features.loc[[patient_id]]
    similarity_values = cosine_similarity(target_vector, aligned_features)[0]
    similarity_series = pd.Series(similarity_values, index=aligned_features.index).drop(patient_id, errors='ignore')
    similarity_series = similarity_series[similarity_series > 0].sort_values(ascending=False).head(top_k)

    if similarity_series.empty:
        return pd.Series(0.0, index=all_packages)

    similar_patient_packages = patient_matrix.loc[similarity_series.index]
    weighted_scores = similar_patient_packages.mul(similarity_series, axis=0).sum(axis=0)
    weighted_scores = weighted_scores / similarity_series.sum()

    return weighted_scores.reindex(all_packages, fill_value=0.0).drop(purchased_pkgs, errors='ignore')


def apply_clinical_rules(scores, age, gender):
    for pkg_id in scores.index:
        if gender == 'M' and pkg_id in ['NVV-PK-0007', 'NVV-PK-0012', 'NVV-PK-0015', 'NVV-PK-0035']: scores[pkg_id] = -999.0
        if gender == 'F' and pkg_id in ['NVV-PK-0014', 'NVV-PK-0040']: scores[pkg_id] = -999.0
        if age > 14 and pkg_id in ['NVV-PK-0092']: scores[pkg_id] = -999.0
        if age < 35 and pkg_id in ['NVV-PK-0017', 'NVV-PK-0014', 'NVV-PK-0015']: scores[pkg_id] = -999.0

    return scores


def predict_top_packages_strict(cf_matrix, train_df, patient_features, patient_id, purchased_pkgs, age, gender):
    cf_scores = build_cf_scores(cf_matrix, purchased_pkgs)
    profile_scores = build_profile_scores(train_df, patient_features, patient_id, purchased_pkgs, cf_matrix.columns, top_k=5)
    scores = (0.7 * normalize_scores(cf_scores)) + (0.3 * normalize_scores(profile_scores))
    scores = apply_clinical_rules(scores, age, gender)
    valid_scores = scores[scores > -50.0]
    return valid_scores.sort_values(ascending=False).head(3).index.tolist()

hits = 0
violations = 0
valid_test_cases = 0
hidden_packages_total = 0

print("กำลังไล่จำลองทดสอบเดาใจคนไข้กลุ่ม 20% ที่เหลือ (ซ่อน 2 แพ็กเกจ เหลือให้เห็น 1)...")
for patient in test_patients:
    history = test_df[test_df['PATIENT'] == patient]
    pkgs = history['PACKAGE'].tolist()
    
    if len(pkgs) >= 3:
        valid_test_cases += 1
        age = history.iloc[0]['AGE']
        gender = history.iloc[0]['GENDER']
        
        visible_pkg = random.choice(pkgs)
        remaining_pkgs = [p for p in pkgs if p != visible_pkg]
        hidden_pkgs = random.sample(remaining_pkgs, 2)
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
            
        for p in top_3:
            if gender == 'M' and p in ['NVV-PK-0007', 'NVV-PK-0015', 'NVV-PK-0035']: violations += 1
            if gender == 'F' and p in ['NVV-PK-0014', 'NVV-PK-0040']: violations += 1

print("\n==================================================")
print("📊 [STRICT EVALUATION] 80/20 Train-Test Split (No Leakage) 📊")
print("==================================================")
if valid_test_cases == 0:
    print("ไม่พบเคสทดสอบ")
else:
    hit_rate = (hits / hidden_packages_total) * 100 if hidden_packages_total else 0.0
    print(f"🎯 อัตราเดาใจถูกต่อแพ็กเกจที่ถูกซ่อน (True Predictive Hit Rate) : {hit_rate:.2f}% (แม่น {hits}/{hidden_packages_total} แพ็กเกจ)")
    print(f"🧪 จำนวนเคสทดสอบ: {valid_test_cases} คน (เห็น 1 แพ็กเกจ, ซ่อน 2 แพ็กเกจ)")
    print(f"⚕️ ตรวจสอบการผิดกฎการแพทย์ (Medical Violations) : {violations} ครั้ง")
print("==================================================")
