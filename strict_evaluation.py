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
    
    for pkg in bought_items:
        transactions.append({'PATIENT': patient_id, 'PACKAGE': pkg, 'AGE': age, 'GENDER': gender, 'COUNT': 1, 'CONDITIONS': conditions})

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
all_packages = txn_df['PACKAGE'].unique()

# อุดรอยรั่วเผื่อแพ็กเกจบางอันไม่เคยมีใครซื้อใน 80% แรกเลย (จะได้ไม่ Error ตอนเรียกใช้)
for pkg in all_packages:
    if pkg not in cf_matrix.columns:
        cf_matrix[pkg] = 0.0
        cf_matrix.loc[pkg] = 0.0

def predict_top_packages_strict(cf_matrix, purchased_pkgs, age, gender):
    scores = pd.Series(0.0, index=cf_matrix.columns)
    for pkg in purchased_pkgs:
        if pkg in cf_matrix.index:
            scores += cf_matrix[pkg]
            
    scores = scores.drop(purchased_pkgs, errors='ignore')
    
    for pkg_id in scores.index:
        if gender == 'M' and pkg_id in ['NVV-PK-0007', 'NVV-PK-0012', 'NVV-PK-0015', 'NVV-PK-0035']: scores[pkg_id] = -999.0
        if gender == 'F' and pkg_id in ['NVV-PK-0014', 'NVV-PK-0040']: scores[pkg_id] = -999.0
        if age > 14 and pkg_id in ['NVV-PK-0092']: scores[pkg_id] = -999.0
        if age < 35 and pkg_id in ['NVV-PK-0017', 'NVV-PK-0014', 'NVV-PK-0015']: scores[pkg_id] = -999.0
            
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
        
        hidden_pkgs = random.sample(pkgs, 2)
        known_pkgs = [p for p in pkgs if p not in hidden_pkgs]
        hidden_packages_total += len(hidden_pkgs)
        
        top_3 = predict_top_packages_strict(cf_matrix, known_pkgs, age, gender)
        
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
