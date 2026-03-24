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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MERGED_ENCOUNTERS_PATH = resolve_path("NAVAVEJ_ENCOUNTERS_PATH", DEFAULT_ENCOUNTERS_PATH)
NAVAVEJ_PACKAGES_PATH = resolve_path("NAVAVEJ_PACKAGES_PATH", DEFAULT_PACKAGES_PATH)
MODEL_OUTPUT_PATH = resolve_path("NAVAVEJ_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_OUTPUT_DIR = MODEL_OUTPUT_PATH.parent


def load_packages_dataframe():
    if NAVAVEJ_PACKAGES_PATH.exists():
        return pd.read_csv(NAVAVEJ_PACKAGES_PATH)

    print(f"⚠️ ไม่พบไฟล์ package catalog ที่ {NAVAVEJ_PACKAGES_PATH} จะ fallback เป็น code=ชื่อชั่วคราวสำหรับเดโม")
    return pd.DataFrame(columns=["code", "name"])

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
        
        # 1. Maternity & Female Health
        if gender == 'F':
            if 18 <= age <= 45 and ('pregnancy' in conditions or 'prenatal' in conditions):
                if np.random.rand() > 0.1: bought_items.add('NVV-PK-0007') # คลอดธรรมชาติ
                if np.random.rand() > 0.3: bought_items.add('NVV-PK-0061') # ผ่าตัดคลอด
            if 15 <= age <= 45:
                if np.random.rand() > 0.7: bought_items.add('NVV-PK-0059') # HPV Vax 9 สายพันธุ์
            if age >= 30:
                if np.random.rand() > 0.6: bought_items.add('NVV-PK-0035') # มะเร็งปากมดลูก
                if np.random.rand() > 0.8: bought_items.add('NVV-PK-0036') # Longevist F
            if age >= 35:
                if np.random.rand() > 0.4: bought_items.add('NVV-PK-0015') # Pro Diamond F
            if age >= 25:
                if np.random.rand() > 0.9: bought_items.add('NVV-PK-0028') # ตกแต่งเลเบีย

        # 2. Male Health
        if gender == 'M':
            if age >= 40:
                if np.random.rand() > 0.5: bought_items.add('NVV-PK-0014') # Pro Diamond M
            if 40 <= age <= 55:
                if np.random.rand() > 0.7: bought_items.add('NVV-PK-0040') # Hormone M
                
        # 3. Kids & Teens
        if age <= 14:
            if np.random.rand() > 0.3: bought_items.add('NVV-PK-0092') # เด็ก Play and Learn 1
            if np.random.rand() > 0.5: bought_items.add('NVV-PK-0084') # ทันตกรรมเด็ก
            if np.random.rand() > 0.8: bought_items.add('NVV-PK-0082') # COVID เด็ก
            
        # 4. Seniors & Neurology
        if age >= 55:
            if np.random.rand() > 0.6: bought_items.add('NVV-PK-0017') # Dementia
            if np.random.rand() > 0.5: bought_items.add('NVV-PK-0018') # Stroke Screening
            if np.random.rand() > 0.7: bought_items.add('NVV-PK-0003') # MRI+MRA Brain
            
        # 5. Chronic & Internal Med
        if age >= 50 or 'diabetes' in conditions or 'hypertension' in conditions or 'hyperlipidemia' in conditions:
            if np.random.rand() > 0.3: bought_items.add('NVV-PK-0002') # ตรวจสุขภาพ 
            if np.random.rand() > 0.5: bought_items.add('NVV-PK-0078') # Echo หัวใจ
            if np.random.rand() > 0.6: bought_items.add('NVV-PK-0081') # Echocardiogram 2569
            if np.random.rand() > 0.7: bought_items.add('NVV-PK-0064') # มะเร็งลำไส้ CEA
            
        # 6. Wellness & Anti-Aging (IV Drips)
        if 25 <= age <= 60:
            if np.random.rand() > 0.8: bought_items.add('NVV-PK-0046') # Liver Detox
            if np.random.rand() > 0.8: bought_items.add('NVV-PK-0047') # Immune Booster
            if np.random.rand() > 0.85: bought_items.add('NVV-PK-0083') # Brain Booster
            if np.random.rand() > 0.85: bought_items.add('NVV-PK-0062') # High C
            if np.random.rand() > 0.9: bought_items.add('NVV-PK-0048') # Mitochondria
            
        # 7. General & Dental (ทุกเพศทุกวัย)
        if age >= 20:
            if np.random.rand() > 0.7: bought_items.add('NVV-PK-0031') # ขูดหินปูน
            if np.random.rand() > 0.85: bought_items.add('NVV-PK-0044') # ฟอกสีฟัน
            if np.random.rand() > 0.8: bought_items.add('NVV-PK-0004') # Chest X-ray
            if np.random.rand() > 0.9: bought_items.add('NVV-PK-0001') # Sleep Test
            if np.random.rand() > 0.9: bought_items.add('NVV-PK-0098') # HD Mall Sleep Test
            if np.random.rand() > 0.85: bought_items.add('NVV-PK-0072') # ไวรัสตับอักเสบบี
            if np.random.rand() > 0.85: bought_items.add('NVV-PK-0045') # อีสุกอีใส
        
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
            'Cholesterol': safe_get('total_cholesterol', 150.0, 240.0)
        }
        
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
    
    print(f"[3/3] 💾 Save Model Artifact: บันทึกความรู้ทั้งหมดไว้ที่ {MODEL_OUTPUT_PATH}...")
    artifacts = {
        'cf_matrix': item_sim_df,
        'transactions': txn_df,       
        'packages': packages_df       
    }
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print("✅ ระบบอัปเดตโมเดลประจำวันเสร็จสิ้นแล้ว! (Production Training Complete)")

if __name__ == "__main__":
    run_etl_and_training()
