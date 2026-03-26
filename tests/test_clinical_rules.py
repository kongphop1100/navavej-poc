"""Tests for apply_clinical_rules — the highest-risk logic in the system.

These tests act as a regression guard: if any clinical rule is accidentally
removed or modified, at least one test here will fail immediately.
"""
import pandas as pd
import pytest

from src.inference_engine import (
    CLINICAL_RULE_MEN_PACKAGES,
    CLINICAL_RULE_MIN_AGE_PACKAGES,
    CLINICAL_RULE_PEDIATRIC_PACKAGES,
    CLINICAL_RULE_WOMEN_PACKAGES,
    apply_clinical_rules,
)

SENTINEL = -999.0
VALID_SCORE = 0.8


def make_scores(packages: list[str], score: float = VALID_SCORE) -> pd.Series:
    """Helper: build a uniform-score Series for the given package list."""
    return pd.Series({pkg: score for pkg in packages})


# ---------------------------------------------------------------------------
# Women's health packages must never be recommended to male patients
# ---------------------------------------------------------------------------
class TestGenderRules:
    def test_women_packages_excluded_for_male(self):
        scores = make_scores(list(CLINICAL_RULE_WOMEN_PACKAGES))
        result = apply_clinical_rules(scores.copy(), age=35, gender="M")
        for pkg in CLINICAL_RULE_WOMEN_PACKAGES:
            assert result[pkg] == SENTINEL, f"{pkg} should be excluded for male"

    def test_women_packages_kept_for_female(self):
        scores = make_scores(list(CLINICAL_RULE_WOMEN_PACKAGES))
        result = apply_clinical_rules(scores.copy(), age=35, gender="F")
        for pkg in CLINICAL_RULE_WOMEN_PACKAGES:
            assert result[pkg] == VALID_SCORE, f"{pkg} should be kept for female"

    def test_men_packages_excluded_for_female(self):
        scores = make_scores(list(CLINICAL_RULE_MEN_PACKAGES))
        result = apply_clinical_rules(scores.copy(), age=45, gender="F")
        for pkg in CLINICAL_RULE_MEN_PACKAGES:
            assert result[pkg] == SENTINEL, f"{pkg} should be excluded for female"

    def test_men_packages_kept_for_male(self):
        scores = make_scores(list(CLINICAL_RULE_MEN_PACKAGES))
        result = apply_clinical_rules(scores.copy(), age=45, gender="M")
        for pkg in CLINICAL_RULE_MEN_PACKAGES:
            assert result[pkg] == VALID_SCORE, f"{pkg} should be kept for male"


# ---------------------------------------------------------------------------
# Pediatric packages must never be recommended to adults (age > 14)
# ---------------------------------------------------------------------------
class TestPediatricRules:
    def test_pediatric_packages_excluded_for_adults(self):
        scores = make_scores(list(CLINICAL_RULE_PEDIATRIC_PACKAGES))
        for age in [15, 20, 45, 60]:
            result = apply_clinical_rules(scores.copy(), age=age, gender="M")
            for pkg in CLINICAL_RULE_PEDIATRIC_PACKAGES:
                assert result[pkg] == SENTINEL, f"{pkg} should be excluded for age {age}"

    def test_pediatric_packages_kept_for_children(self):
        scores = make_scores(list(CLINICAL_RULE_PEDIATRIC_PACKAGES))
        for age in [1, 8, 14]:
            result = apply_clinical_rules(scores.copy(), age=age, gender="M")
            for pkg in CLINICAL_RULE_PEDIATRIC_PACKAGES:
                assert result[pkg] == VALID_SCORE, f"{pkg} should be kept for age {age}"


# ---------------------------------------------------------------------------
# Age-restricted packages must not be recommended to patients under 20
# ---------------------------------------------------------------------------
class TestMinAgRules:
    def test_min_age_packages_excluded_for_under_20(self):
        scores = make_scores(list(CLINICAL_RULE_MIN_AGE_PACKAGES))
        for age in [10, 15, 18, 19]:
            result = apply_clinical_rules(scores.copy(), age=age, gender="M")
            for pkg in CLINICAL_RULE_MIN_AGE_PACKAGES:
                assert result[pkg] == SENTINEL, f"{pkg} should be excluded for age {age}"

    def test_min_age_packages_kept_for_adults(self):
        # Some min-age packages also appear in CLINICAL_RULE_WOMEN_PACKAGES,
        # so they must be tested with gender="F" (the only valid gender for them).
        # Pure min-age packages (not women-only) can be tested with either gender.
        pure_min_age = CLINICAL_RULE_MIN_AGE_PACKAGES - CLINICAL_RULE_WOMEN_PACKAGES
        scores = make_scores(list(pure_min_age))
        result = apply_clinical_rules(scores.copy(), age=20, gender="M")
        for pkg in pure_min_age:
            assert result[pkg] == VALID_SCORE, f"{pkg} should be allowed for age 20 male"


# ---------------------------------------------------------------------------
# Unrelated packages must not be touched by any rule
# ---------------------------------------------------------------------------
class TestNeutralPackages:
    NEUTRAL = ["NVV-PK-0031", "NVV-PK-0044", "NVV-PK-0004"]

    def test_neutral_packages_not_affected_for_male_adult(self):
        scores = make_scores(self.NEUTRAL)
        result = apply_clinical_rules(scores.copy(), age=40, gender="M")
        for pkg in self.NEUTRAL:
            assert result[pkg] == VALID_SCORE

    def test_neutral_packages_not_affected_for_female_adult(self):
        scores = make_scores(self.NEUTRAL)
        result = apply_clinical_rules(scores.copy(), age=40, gender="F")
        for pkg in self.NEUTRAL:
            assert result[pkg] == VALID_SCORE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_scores_returns_empty(self):
        result = apply_clinical_rules(pd.Series(dtype=float), age=30, gender="M")
        assert result.empty

    def test_already_negative_scores_not_affected(self):
        """Rules should not reset already-excluded package scores."""
        pkg = "NVV-PK-0031"
        scores = pd.Series({pkg: -999.0})
        result = apply_clinical_rules(scores.copy(), age=30, gender="M")
        assert result[pkg] == -999.0

    def test_zero_score_not_affected_by_rules(self):
        """A package with score 0 that is not rule-blocked should stay 0."""
        pkg = "NVV-PK-0031"
        scores = pd.Series({pkg: 0.0})
        result = apply_clinical_rules(scores.copy(), age=35, gender="F")
        assert result[pkg] == pytest.approx(0.0)
