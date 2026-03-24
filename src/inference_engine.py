import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
        self.patient_features = artifacts.get('patient_features')
        self.patient_package_matrix = artifacts.get('patient_package_matrix')

        if self.patient_package_matrix is None:
            self.patient_package_matrix = self.txn_df.pivot_table(
                index='PATIENT', columns='PACKAGE', values='COUNT', fill_value=0
            )
        if self.patient_features is None:
            self.patient_features = pd.DataFrame(index=self.patient_package_matrix.index)
        
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

    
    def _build_cf_scores(self, purchased_pkgs):
        scores = pd.Series(0.0, index=self.cf_matrix.columns)
        for pkg in purchased_pkgs:
            if pkg in self.cf_matrix.index:
                scores += self.cf_matrix[pkg]

        return scores.drop(purchased_pkgs, errors='ignore')

    def _build_profile_scores(self, patient_id, purchased_pkgs, top_k=5):
        if patient_id not in self.patient_features.index or self.patient_features.empty:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        aligned_features = self.patient_features.loc[
            self.patient_features.index.intersection(self.patient_package_matrix.index)
        ]
        if patient_id not in aligned_features.index or len(aligned_features) <= 1:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        target_vector = aligned_features.loc[[patient_id]]
        similarity_values = cosine_similarity(target_vector, aligned_features)[0]
        similarity_series = pd.Series(similarity_values, index=aligned_features.index).drop(patient_id, errors='ignore')
        similarity_series = similarity_series[similarity_series > 0].sort_values(ascending=False).head(top_k)

        if similarity_series.empty:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        similar_patient_packages = self.patient_package_matrix.loc[similarity_series.index]
        weighted_scores = similar_patient_packages.mul(similarity_series, axis=0).sum(axis=0)
        weighted_scores = weighted_scores / similarity_series.sum()

        return weighted_scores.reindex(self.cf_matrix.columns, fill_value=0.0).drop(purchased_pkgs, errors='ignore')

    def _normalize_scores(self, scores):
        positive_scores = scores[scores > 0]
        if positive_scores.empty:
            return pd.Series(0.0, index=scores.index)

        max_score = positive_scores.max()
        return scores.clip(lower=0) / max_score if max_score else pd.Series(0.0, index=scores.index)

    def _apply_clinical_rules(self, scores, age, gender):
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

        return scores

    def recommend(self, patient_id):
        profile, purchased_pkgs = self.get_patient_profile(patient_id)
        if not profile:
            return pd.Series(dtype=float)

        age, gender = profile['AGE'], profile['GENDER']
        cf_scores = self._build_cf_scores(purchased_pkgs)
        profile_scores = self._build_profile_scores(patient_id, purchased_pkgs)
        final_scores = (0.6 * self._normalize_scores(cf_scores)) + (0.4 * self._normalize_scores(profile_scores))
        final_scores = self._apply_clinical_rules(final_scores, age, gender)

        valid_scores = final_scores[final_scores > -50.0]
        return valid_scores.sort_values(ascending=False).head(3)
