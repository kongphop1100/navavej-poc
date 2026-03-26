"""Tests for _condition_aware_ranges() in dataset_snapshot.py."""
import pytest

from src.dataset_snapshot import _condition_aware_ranges


def ranges_for(conditions: str, age: int = 35, gender: str = "M") -> dict:
    return _condition_aware_ranges(conditions, age, gender)


class TestHealthyDefaults:
    def test_healthy_glucose_in_normal_range(self):
        r = ranges_for("")
        lo, hi = r["Glucose"]
        assert lo >= 70.0 and hi <= 100.0

    def test_healthy_hba1c_below_5_7(self):
        r = ranges_for("")
        _, hi = r["HbA1c"]
        assert hi <= 5.7

    def test_healthy_bmi_normal(self):
        r = ranges_for("")
        lo, hi = r["BMI"]
        assert lo >= 18.0 and hi <= 25.0

    def test_healthy_egfr_normal(self):
        r = ranges_for("")
        lo, _ = r["eGFR"]
        assert lo >= 60.0


class TestDiabetes:
    def test_glucose_elevated(self):
        r = ranges_for("diabetes mellitus")
        lo, _ = r["Glucose"]
        assert lo >= 126.0, "Diabetic glucose lower bound must be ≥ 126"

    def test_hba1c_elevated(self):
        r = ranges_for("diabetes mellitus")
        lo, _ = r["HbA1c"]
        assert lo >= 6.5, "Diabetic HbA1c lower bound must be ≥ 6.5"

    def test_prediabetes_glucose_between_100_125(self):
        r = ranges_for("prediabetes")
        lo, hi = r["Glucose"]
        assert lo >= 100.0 and hi <= 126.0


class TestHypertension:
    def test_sysbp_elevated(self):
        r = ranges_for("hypertension")
        lo, _ = r["SysBP"]
        assert lo >= 140.0, "Hypertensive SysBP lower bound must be ≥ 140"

    def test_diabp_elevated(self):
        r = ranges_for("hypertension")
        lo, _ = r["DiaBP"]
        assert lo >= 90.0, "Hypertensive DiaBP lower bound must be ≥ 90"


class TestHyperlipidemia:
    def test_cholesterol_elevated(self):
        r = ranges_for("hyperlipidemia")
        lo, _ = r["Cholesterol"]
        assert lo >= 220.0

    def test_ldl_elevated(self):
        r = ranges_for("hyperlipidemia")
        lo, _ = r["LDL"]
        assert lo >= 130.0

    def test_hdl_lowered(self):
        r = ranges_for("hyperlipidemia")
        _, hi = r["HDL"]
        assert hi <= 50.0, "HDL should be lower in hyperlipidemia"

    def test_triglycerides_elevated(self):
        r = ranges_for("hyperlipidemia")
        lo, _ = r["Triglycerides"]
        assert lo >= 150.0


class TestObesity:
    def test_bmi_above_30(self):
        r = ranges_for("obesity")
        lo, _ = r["BMI"]
        assert lo >= 30.0

    def test_weight_elevated(self):
        r = ranges_for("obesity")
        lo, _ = r["BodyWeight"]
        assert lo >= 90.0

    def test_overweight_less_severe_than_obesity(self):
        r_obese = ranges_for("obesity")
        r_over = ranges_for("overweight")
        assert r_obese["BMI"][0] > r_over["BMI"][0]


class TestKidneyDisease:
    def test_egfr_reduced(self):
        r = ranges_for("kidney disease")
        _, hi = r["eGFR"]
        assert hi < 60.0, "eGFR upper bound should be < 60 for kidney disease"

    def test_renal_keyword_also_works(self):
        r = ranges_for("renal failure")
        _, hi = r["eGFR"]
        assert hi < 60.0


class TestAgeAdjustments:
    def test_elderly_egfr_lower_than_young(self):
        r_young = ranges_for("", age=35)
        r_old = ranges_for("", age=65)
        assert r_old["eGFR"][1] < r_young["eGFR"][1]

    def test_elderly_sysbp_higher_than_young(self):
        r_young = ranges_for("", age=35)
        r_old = ranges_for("", age=65)
        assert r_old["SysBP"][0] >= r_young["SysBP"][0]


class TestMultipleConditions:
    def test_diabetes_and_hypertension_both_applied(self):
        r = ranges_for("diabetes | hypertension")
        assert r["Glucose"][0] >= 126.0
        assert r["SysBP"][0] >= 140.0

    def test_all_chronic_conditions(self):
        r = ranges_for("diabetes | hypertension | hyperlipidemia | obesity")
        assert r["Glucose"][0] >= 126.0
        assert r["SysBP"][0] >= 140.0
        assert r["Cholesterol"][0] >= 220.0
        assert r["BMI"][0] >= 30.0
