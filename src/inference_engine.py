import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_MODEL_PATH, resolve_path

# ---------------------------------------------------------------------------
# Shared constants — imported by train_job.py and strict_evaluation.py
# ---------------------------------------------------------------------------

PROFILE_NUMERIC_RANGES = {
    "AGE": (0.0, 100.0),
    "PACKAGE_COUNT": (0.0, 12.0),
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

# Clinical rule constants — single source of truth
CLINICAL_RULE_WOMEN_PACKAGES = frozenset({
    "NVV-PK-0007", "NVV-PK-0061", "NVV-PK-0059",
    "NVV-PK-0035", "NVV-PK-0036", "NVV-PK-0015", "NVV-PK-0028",
})
CLINICAL_RULE_MEN_PACKAGES = frozenset({"NVV-PK-0014", "NVV-PK-0040"})
CLINICAL_RULE_PEDIATRIC_PACKAGES = frozenset({"NVV-PK-0092", "NVV-PK-0084", "NVV-PK-0082"})
CLINICAL_RULE_MIN_AGE_PACKAGES = frozenset({
    "NVV-PK-0014", "NVV-PK-0015", "NVV-PK-0017", "NVV-PK-0018",
    "NVV-PK-0040", "NVV-PK-0036", "NVV-PK-0028", "NVV-PK-0001", "NVV-PK-0098",
})


def normalize_range(value, min_val, max_val):
    """Clamp-and-normalise a numeric value to [0, 1]."""
    clipped = min(max(float(value), min_val), max_val)
    return (clipped - min_val) / (max_val - min_val)


def has_any_keyword(text, keywords):
    """Return 1.0 if any keyword is found in text, else 0.0."""
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0


def apply_clinical_rules(scores: pd.Series, age: int, gender: str) -> pd.Series:
    """Apply gender- and age-based medical exclusion rules.

    Packages that violate a rule receive a score of -999.0 so they are
    filtered out downstream.  This is the single authoritative implementation
    used by both the inference engine and the offline evaluation script.
    """
    for pkg_id in scores.index:
        if gender == "M" and pkg_id in CLINICAL_RULE_WOMEN_PACKAGES:
            scores[pkg_id] = -999.0
        if gender == "F" and pkg_id in CLINICAL_RULE_MEN_PACKAGES:
            scores[pkg_id] = -999.0
        if age > 14 and pkg_id in CLINICAL_RULE_PEDIATRIC_PACKAGES:
            scores[pkg_id] = -999.0
        if age < 20 and pkg_id in CLINICAL_RULE_MIN_AGE_PACKAGES:
            scores[pkg_id] = -999.0
    return scores


# ---------------------------------------------------------------------------

PACKAGE_GROUPS = {
    "chronic_screening": {"NVV-PK-0002", "NVV-PK-0078", "NVV-PK-0081", "NVV-PK-0064", "NVV-PK-0003"},
    "women_health": {"NVV-PK-0007", "NVV-PK-0061", "NVV-PK-0059", "NVV-PK-0035", "NVV-PK-0036", "NVV-PK-0015", "NVV-PK-0028"},
    "male_health": {"NVV-PK-0014", "NVV-PK-0040"},
    "pediatric": {"NVV-PK-0092", "NVV-PK-0084", "NVV-PK-0082"},
    "wellness": {"NVV-PK-0046", "NVV-PK-0047", "NVV-PK-0083", "NVV-PK-0062", "NVV-PK-0048"},
    "general_screening": {"NVV-PK-0031", "NVV-PK-0044", "NVV-PK-0004", "NVV-PK-0001", "NVV-PK-0098", "NVV-PK-0072", "NVV-PK-0045"},
}

DEFAULT_CF_WEIGHT = 0.7
DEFAULT_PROFILE_WEIGHT = 0.3
DEFAULT_PROFILE_TOP_K = 10
DEFAULT_FEATURE_WEIGHTS = {
    "PACKAGE_COUNT_NORM": 0.25,
}


class UpsellRecommenderEngine:
    def __init__(
        self,
        model_path=None,
        cf_weight=DEFAULT_CF_WEIGHT,
        profile_weight=DEFAULT_PROFILE_WEIGHT,
        profile_top_k=DEFAULT_PROFILE_TOP_K,
        feature_weights=None,
    ):
        """โหลดโมเดล Pickle มาเก็บบน RAM ตอนเซิร์ฟเวอร์เริ่มทำงาน"""
        model_path = resolve_path("NAVAVEJ_MODEL_PATH", DEFAULT_MODEL_PATH) if model_path is None else model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError("ไม่พบไฟล์โมเดล! เซิร์ฟเวอร์ไม่สามารถให้บริการได้ ทักไปถาม Data Team ให้รันสร้างโมเดลก่อนครับ")

        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        self.cf_matrix = artifacts["cf_matrix"]
        self.txn_df = artifacts["transactions"]
        self.packages_df = artifacts["packages"]
        self.patient_features = artifacts.get("patient_features")
        self.patient_package_matrix = artifacts.get("patient_package_matrix")
        self.patient_profiles = artifacts.get("patient_profiles")
        self.cold_start_patients = set(artifacts.get("cold_start_patients", []))

        if self.patient_package_matrix is None:
            self.patient_package_matrix = self.txn_df.pivot_table(
                index="PATIENT", columns="PACKAGE", values="COUNT", fill_value=0
            )
        if self.patient_features is None:
            self.patient_features = pd.DataFrame(index=self.patient_package_matrix.index)
        if self.patient_profiles is None:
            grouped = self.txn_df.groupby("PATIENT").first()[["AGE", "GENDER", "CONDITIONS", "VITALS"]].copy()
            grouped["CUSTOMER_TYPE"] = "existing"
            grouped["PACKAGE_COUNT"] = self.txn_df.groupby("PATIENT").size()
            self.patient_profiles = grouped

        self.cf_weight = float(cf_weight)
        self.profile_weight = float(profile_weight)
        self.profile_top_k = int(profile_top_k)
        self.feature_weights = dict(DEFAULT_FEATURE_WEIGHTS)
        if feature_weights:
            self.feature_weights.update(feature_weights)

    def get_patient_profile(self, patient_id):
        if patient_id not in self.patient_profiles.index:
            return None, []

        profile_row = self.patient_profiles.loc[patient_id]
        history = self.txn_df[self.txn_df["PATIENT"] == patient_id]
        profile = {
            "AGE": profile_row["AGE"],
            "GENDER": profile_row["GENDER"],
            "CONDITIONS": profile_row["CONDITIONS"],
            "VITALS": profile_row.get("VITALS", {}),
            "CUSTOMER_TYPE": profile_row.get("CUSTOMER_TYPE", "existing"),
        }
        purchased_pkgs = history["PACKAGE"].tolist() if not history.empty else []
        return profile, purchased_pkgs

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _package_group(self, package_id):
        for group_name, package_ids in PACKAGE_GROUPS.items():
            if package_id in package_ids:
                return group_name
        return "general"

    def _weighted_patient_features(self):
        if self.patient_features.empty:
            return self.patient_features

        weighted_features = self.patient_features.copy()
        for column, weight in self.feature_weights.items():
            if column in weighted_features.columns:
                weighted_features[column] = weighted_features[column] * float(weight)
        return weighted_features

    def _build_cf_scores(self, purchased_pkgs):
        scores = pd.Series(0.0, index=self.cf_matrix.columns)
        for pkg in purchased_pkgs:
            if pkg in self.cf_matrix.index:
                scores += self.cf_matrix[pkg]
        return scores.drop(purchased_pkgs, errors="ignore")

    def _build_profile_scores(self, patient_id, purchased_pkgs, top_k=None):
        if patient_id not in self.patient_features.index or self.patient_features.empty:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        top_k = self.profile_top_k if top_k is None else int(top_k)
        weighted_features = self._weighted_patient_features()
        aligned_features = weighted_features.loc[weighted_features.index.intersection(self.patient_package_matrix.index)]
        if aligned_features.empty or len(aligned_features) <= 1:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        target_vector = weighted_features.loc[[patient_id]]
        similarity_values = cosine_similarity(target_vector, aligned_features)[0]
        similarity_series = pd.Series(similarity_values, index=aligned_features.index).drop(patient_id, errors="ignore")
        similarity_series = similarity_series[similarity_series > 0].sort_values(ascending=False).head(top_k)

        if similarity_series.empty:
            return pd.Series(0.0, index=self.cf_matrix.columns)

        similar_patient_packages = self.patient_package_matrix.loc[similarity_series.index]
        weighted_scores = similar_patient_packages.mul(similarity_series, axis=0).sum(axis=0)
        weighted_scores = weighted_scores / similarity_series.sum()

        return weighted_scores.reindex(self.cf_matrix.columns, fill_value=0.0).drop(purchased_pkgs, errors="ignore")

    def _normalize_scores(self, scores):
        positive_scores = scores[scores > 0]
        if positive_scores.empty:
            return pd.Series(0.0, index=scores.index)

        max_score = positive_scores.max()
        return scores.clip(lower=0) / max_score if max_score else pd.Series(0.0, index=scores.index)

    def _apply_clinical_rules(self, scores, age, gender):
        """Delegate to the module-level authoritative implementation."""
        return apply_clinical_rules(scores, age, gender)

    def recommend(self, patient_id):
        profile, purchased_pkgs = self.get_patient_profile(patient_id)
        if not profile:
            return pd.Series(dtype=float)

        age, gender = profile["AGE"], profile["GENDER"]
        cf_scores = self._build_cf_scores(purchased_pkgs) if purchased_pkgs else pd.Series(0.0, index=self.cf_matrix.columns)
        profile_scores = self._build_profile_scores(patient_id, purchased_pkgs)
        if purchased_pkgs:
            final_scores = (self.cf_weight * self._normalize_scores(cf_scores)) + (self.profile_weight * self._normalize_scores(profile_scores))
        else:
            final_scores = self._normalize_scores(profile_scores)
        final_scores = self._apply_clinical_rules(final_scores, age, gender)

        valid_scores = final_scores[final_scores > -50.0]
        return valid_scores.sort_values(ascending=False).head(3)

    def _build_reason_for_package(self, package_id, profile, purchased_pkgs, cf_score, profile_score):
        age = int(profile.get("AGE", 0))
        conditions = str(profile.get("CONDITIONS", "")).lower()
        vitals = profile.get("VITALS", {}) or {}

        glucose = self._safe_float(vitals.get("Glucose"), 0.0)
        sys_bp = self._safe_float(vitals.get("SysBP"), 0.0)
        dia_bp = self._safe_float(vitals.get("DiaBP"), 0.0)
        hba1c = self._safe_float(vitals.get("HbA1c"), 0.0)

        reason_parts = []
        if cf_score >= profile_score and purchased_pkgs:
            seed_packages = ", ".join(str(pkg) for pkg in purchased_pkgs[:2])
            reason_parts.append(f"สัมพันธ์กับแพ็กเกจเดิมของผู้ป่วย ({seed_packages})")
        elif profile_score > 0:
            reason_parts.append("โปรไฟล์ใกล้เคียงกับกลุ่มผู้ป่วยที่มักเลือกแพ็กเกจนี้")

        package_group = self._package_group(package_id)
        if package_group == "chronic_screening":
            clinical_reasons = [f"ผู้ป่วยอายุ {age} ปี"] if age >= 50 else []
            if "hypertension" in conditions or sys_bp >= 140 or dia_bp >= 90:
                clinical_reasons.append("มีความดันสูง")
            if "diabetes" in conditions or glucose >= 126 or hba1c >= 6.5:
                clinical_reasons.append("มีน้ำตาลสูง")
            if clinical_reasons:
                reason_parts.append(f"{' '.join(clinical_reasons)} จึงเข้ากลุ่ม chronic screening")
        elif package_group == "women_health":
            if profile.get("GENDER") == "F":
                if "pregnancy" in conditions or "prenatal" in conditions:
                    reason_parts.append(f"เป็นผู้ป่วยหญิงอายุ {age} ปี และมีบริบทการตั้งครรภ์/ฝากครรภ์")
                elif age >= 30:
                    reason_parts.append(f"เป็นผู้ป่วยหญิงอายุ {age} ปี อยู่ในช่วงที่เหมาะกับโปรแกรมสุขภาพผู้หญิง")
        elif package_group == "male_health":
            if profile.get("GENDER") == "M":
                reason_parts.append(f"เป็นผู้ป่วยชายอายุ {age} ปี จึงเหมาะกับโปรแกรมสุขภาพผู้ชาย")
        elif package_group == "pediatric":
            if age <= 14:
                reason_parts.append(f"ผู้ป่วยอายุ {age} ปี อยู่ในกลุ่มกุมารเวช")
        elif package_group == "wellness":
            reason_parts.append("โปรไฟล์โดยรวมใกล้เคียงกับกลุ่มที่มักเลือกโปรแกรม wellness")
        elif package_group == "general_screening":
            reason_parts.append("เป็นแพ็กเกจตรวจหรือดูแลทั่วไปที่สัมพันธ์กับพฤติกรรมการใช้บริการเดิม")

        if not reason_parts:
            reason_parts.append("ได้คะแนนรวมจากความสัมพันธ์ของแพ็กเกจและความคล้ายของโปรไฟล์ผู้ป่วย")

        return " | ".join(reason_parts[:2])

    def recommend_with_explanations(self, patient_id):
        profile, purchased_pkgs = self.get_patient_profile(patient_id)
        if not profile:
            return []

        age, gender = profile["AGE"], profile["GENDER"]
        cf_scores = self._build_cf_scores(purchased_pkgs) if purchased_pkgs else pd.Series(0.0, index=self.cf_matrix.columns)
        profile_scores = self._build_profile_scores(patient_id, purchased_pkgs)
        normalized_cf_scores = self._normalize_scores(cf_scores)
        normalized_profile_scores = self._normalize_scores(profile_scores)
        recommendation_mode = "hybrid" if purchased_pkgs else "profile_only"
        if purchased_pkgs:
            final_scores = (self.cf_weight * normalized_cf_scores) + (self.profile_weight * normalized_profile_scores)
        else:
            final_scores = normalized_profile_scores
        final_scores = self._apply_clinical_rules(final_scores, age, gender)
        valid_scores = final_scores[final_scores > -50.0].sort_values(ascending=False).head(3)

        recommendations = []
        for package_id, final_score in valid_scores.items():
            recommendations.append(
                {
                    "package_id": str(package_id),
                    "final_score": float(final_score),
                    "cf_score": float(normalized_cf_scores.get(package_id, 0.0)),
                    "profile_score": float(normalized_profile_scores.get(package_id, 0.0)),
                    "recommendation_mode": recommendation_mode,
                    "reason": self._build_reason_for_package(
                        package_id,
                        profile,
                        purchased_pkgs,
                        normalized_cf_scores.get(package_id, 0.0),
                        normalized_profile_scores.get(package_id, 0.0),
                    ),
                }
            )

        return recommendations
