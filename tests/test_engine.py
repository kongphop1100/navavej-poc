"""Tests for UpsellRecommenderEngine using a minimal in-memory mock artifact.

These tests do NOT touch the real .pkl file and do NOT require the full
merged_encounters.csv dataset.  The fixture builds the smallest possible
artifact that exercises the recommendation path.
"""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference_engine import (
    CLINICAL_RULE_MEN_PACKAGES,
    CLINICAL_RULE_WOMEN_PACKAGES,
    UpsellRecommenderEngine,
)

# ---------------------------------------------------------------------------
# Minimal test packages — one women's, one men's, two neutral
# ---------------------------------------------------------------------------
NEUTRAL_A = "NVV-PK-0031"
NEUTRAL_B = "NVV-PK-0044"
WOMENS_PKG = next(iter(CLINICAL_RULE_WOMEN_PACKAGES))   # e.g. NVV-PK-0007
MENS_PKG = next(iter(CLINICAL_RULE_MEN_PACKAGES))        # e.g. NVV-PK-0014
ALL_PKGS = [NEUTRAL_A, NEUTRAL_B, WOMENS_PKG, MENS_PKG]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Build a minimal model artifact and return a loaded engine."""
    patients = ["P001", "P002", "P003", "P004", "P005"]

    # Transactions: P001–P004 bought both neutral packages + one gender pkg
    txn_rows = []
    for pid in patients[:4]:
        for pkg in [NEUTRAL_A, NEUTRAL_B, WOMENS_PKG]:
            txn_rows.append({"PATIENT": pid, "PACKAGE": pkg, "AGE": 35,
                             "GENDER": "F", "COUNT": 1,
                             "CONDITIONS": "hypertension",
                             "VITALS": {}})
    txn_df = pd.DataFrame(txn_rows)

    matrix = txn_df.pivot_table(index="PATIENT", columns="PACKAGE", values="COUNT", fill_value=0)
    # Add the men's package column so the CF matrix covers it
    matrix[MENS_PKG] = 0

    from sklearn.metrics.pairwise import cosine_similarity
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

    profiles = pd.DataFrame([
        {"PATIENT": "P001", "AGE": 35, "GENDER": "F",
         "CONDITIONS": "hypertension", "VITALS": {}, "CUSTOMER_TYPE": "existing", "PACKAGE_COUNT": 3},
        {"PATIENT": "P002", "AGE": 45, "GENDER": "M",
         "CONDITIONS": "", "VITALS": {}, "CUSTOMER_TYPE": "existing", "PACKAGE_COUNT": 3},
        {"PATIENT": "P003", "AGE": 10, "GENDER": "M",
         "CONDITIONS": "", "VITALS": {}, "CUSTOMER_TYPE": "existing", "PACKAGE_COUNT": 3},
        {"PATIENT": "NEW1", "AGE": 30, "GENDER": "F",
         "CONDITIONS": "", "VITALS": {}, "CUSTOMER_TYPE": "new", "PACKAGE_COUNT": 0},
    ]).set_index("PATIENT")

    # Add P002-P004 with men's pkg transactions so the model has male history
    male_rows = [
        {"PATIENT": "P002", "PACKAGE": NEUTRAL_A, "AGE": 45, "GENDER": "M",
         "COUNT": 1, "CONDITIONS": "", "VITALS": {}},
        {"PATIENT": "P002", "PACKAGE": MENS_PKG, "AGE": 45, "GENDER": "M",
         "COUNT": 1, "CONDITIONS": "", "VITALS": {}},
    ]
    txn_df = pd.concat([txn_df, pd.DataFrame(male_rows)], ignore_index=True)

    packages_df = pd.DataFrame({
        "code": ALL_PKGS,
        "name": [f"Package {p}" for p in ALL_PKGS],
    })

    artifact = {
        "cf_matrix": item_sim_df,
        "transactions": txn_df,
        "packages": packages_df,
        "patient_features": pd.DataFrame(index=profiles.index),  # empty → profile fallback
        "patient_package_matrix": matrix,
        "patient_profiles": profiles,
        "cold_start_patients": ["NEW1"],
        "existing_train_patients": ["P001", "P002", "P003"],
        "existing_test_patients": ["P004"],
        "dataset_snapshot_path": "mock",
        "snapshot_metadata": {"seed": 0, "cold_start_target": 1,
                               "total_patients": 4, "existing_patients": 3},
    }

    tmp = tmp_path_factory.mktemp("model")
    pkl_path = tmp / "model_artifacts.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    return UpsellRecommenderEngine(model_path=str(pkl_path))


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------
class TestEngineLoads:
    def test_engine_loads_without_error(self, engine):
        assert engine is not None

    def test_patient_profiles_populated(self, engine):
        assert len(engine.patient_profiles) > 0


# ---------------------------------------------------------------------------
# Gender clinical rules via recommend_with_explanations
# ---------------------------------------------------------------------------
class TestGenderRecommendations:
    def test_male_patient_not_recommended_womens_packages(self, engine):
        recs = engine.recommend_with_explanations("P002")
        recommended_ids = {r["package_id"] for r in recs}
        for pkg in CLINICAL_RULE_WOMEN_PACKAGES:
            assert pkg not in recommended_ids, f"Women's package {pkg} recommended to male patient"

    def test_female_patient_not_recommended_mens_packages(self, engine):
        recs = engine.recommend_with_explanations("P001")
        recommended_ids = {r["package_id"] for r in recs}
        for pkg in CLINICAL_RULE_MEN_PACKAGES:
            assert pkg not in recommended_ids, f"Men's package {pkg} recommended to female patient"


# ---------------------------------------------------------------------------
# Child patient
# ---------------------------------------------------------------------------
class TestChildRecommendations:
    def test_child_not_recommended_adult_age_packages(self, engine):
        from src.inference_engine import CLINICAL_RULE_MIN_AGE_PACKAGES
        recs = engine.recommend_with_explanations("P003")
        recommended_ids = {r["package_id"] for r in recs}
        for pkg in CLINICAL_RULE_MIN_AGE_PACKAGES:
            assert pkg not in recommended_ids, f"Age-restricted package {pkg} recommended to child"


# ---------------------------------------------------------------------------
# Cold-start / new patient
# ---------------------------------------------------------------------------
class TestColdStartPatient:
    def test_new_patient_does_not_crash(self, engine):
        # NEW1 is a cold-start patient with no purchase history
        recs = engine.recommend_with_explanations("NEW1")
        # Should return a list (possibly empty) without raising
        assert isinstance(recs, list)

    def test_unknown_patient_returns_empty(self, engine):
        recs = engine.recommend_with_explanations("DOES_NOT_EXIST")
        assert recs == []


# ---------------------------------------------------------------------------
# Score structure
# ---------------------------------------------------------------------------
class TestRecommendationStructure:
    def test_recommendations_have_required_fields(self, engine):
        recs = engine.recommend_with_explanations("P001")
        required_keys = {"package_id", "final_score", "cf_score", "profile_score",
                         "recommendation_mode", "reason"}
        for rec in recs:
            assert required_keys.issubset(rec.keys()), f"Missing keys in {rec}"

    def test_final_score_between_zero_and_one(self, engine):
        recs = engine.recommend_with_explanations("P001")
        for rec in recs:
            assert 0.0 <= rec["final_score"] <= 1.0, f"Score out of range: {rec}"

    def test_maximum_three_recommendations(self, engine):
        recs = engine.recommend_with_explanations("P001")
        assert len(recs) <= 3
