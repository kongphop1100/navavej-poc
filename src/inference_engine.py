import os
import pickle

import pandas as pd

from src.config import DEFAULT_MODEL_PATH, resolve_path

class UpsellRecommenderEngine:
    def __init__(self, model_path=None):
        """โหลดโมเดล Pickle มาเก็บบน RAM ตอนเซิร์ฟเวอร์เริ่มทำงาน"""
        model_path = resolve_path("NAVAVEJ_MODEL_PATH", DEFAULT_MODEL_PATH) if model_path is None else model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError("ไม่พบไฟล์โมเดล! เซิร์ฟเวอร์ไม่สามารถให้บริการได้ ทักไปถาม Data Team ให้รันสร้างโมเดลก่อนครับ")
            
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
            
        self.cf_matrix = artifacts['cf_matrix']
        self.txn_df = artifacts['transactions']
        self.packages_df = artifacts['packages']
        
    def get_patient_profile(self, patient_id):
        # ในระบบสมบูรณ์แบบ ตรงนี้จะวิ่งไปเรียกฐานข้อมูล PostgreSQL/SQL Server
        history = self.txn_df[self.txn_df['PATIENT'] == patient_id]
        if history.empty:
            return None, []
            
        profile = {
            'AGE': history.iloc[0]['AGE'], 
            'GENDER': history.iloc[0]['GENDER'], 
            'CONDITIONS': history.iloc[0]['CONDITIONS'],
            'VITALS': history.iloc[0].get('VITALS', {})
        }
        purchased_pkgs = history['PACKAGE'].tolist()
        return profile, purchased_pkgs
        
    def recommend(self, patient_id):
        profile, purchased_pkgs = self.get_patient_profile(patient_id)
        if not profile:
            return pd.Series(dtype=float)
            
        age, gender = profile['AGE'], profile['GENDER']
        
        scores = pd.Series(0.0, index=self.cf_matrix.columns)
        for pkg in purchased_pkgs:
            if pkg in self.cf_matrix.index:
                scores += self.cf_matrix[pkg]
                
        scores = scores.drop(purchased_pkgs, errors='ignore')
        
        # กฎ Medical เคร่งครัด ปรับเป็นลบเพื่อกันติดโผตอนดึง .head(3)
        for pkg_id in scores.index:
            # ของผู้หญิง ห้ามเสนอให้ผู้ชาย
            if gender == 'M' and pkg_id in ['NVV-PK-0007', 'NVV-PK-0061', 'NVV-PK-0059', 'NVV-PK-0035', 'NVV-PK-0036', 'NVV-PK-0015', 'NVV-PK-0028']: scores[pkg_id] = -999.0
            # ของผู้ชาย ห้ามเสนอผู้หญิง
            if gender == 'F' and pkg_id in ['NVV-PK-0014', 'NVV-PK-0040']: scores[pkg_id] = -999.0
            # ของเด็ก ห้ามเสนอให้ผู้ใหญ่ (เกิน 14 ปี)
            if age > 14 and pkg_id in ['NVV-PK-0092', 'NVV-PK-0084', 'NVV-PK-0082']: scores[pkg_id] = -999.0
            # แพ็กเกจผู้ใหญ่ ห้ามเสนอให้เด็ก (ต่ำกว่า 20 ปี)
            if age < 20 and pkg_id in ['NVV-PK-0014', 'NVV-PK-0015', 'NVV-PK-0017', 'NVV-PK-0018', 'NVV-PK-0040', 'NVV-PK-0036', 'NVV-PK-0028', 'NVV-PK-0001', 'NVV-PK-0098']: scores[pkg_id] = -999.0
                
        valid_scores = scores[scores > -50.0]
        return valid_scores.sort_values(ascending=False).head(3)
