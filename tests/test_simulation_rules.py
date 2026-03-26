"""Tests for simulation probability functions in src/simulation_rules.py."""
import pytest

from src.simulation_rules import (
    clamp_probability,
    compute_chronic_probability,
    compute_general_screening_probability,
    compute_kid_package_probability,
    compute_male_health_probability,
    compute_maternity_probability,
    compute_wellness_probability,
    compute_women_health_probability,
)

# All probability functions must return values in [0.02, 0.95] (clamp bounds)
PROB_MIN = 0.02
PROB_MAX = 0.95


class TestClampProbability:
    def test_value_within_range_unchanged(self):
        assert clamp_probability(0.5) == pytest.approx(0.5)

    def test_below_min_clamped(self):
        assert clamp_probability(0.0) == pytest.approx(PROB_MIN)

    def test_above_max_clamped(self):
        assert clamp_probability(2.0) == pytest.approx(PROB_MAX)

    def test_custom_bounds(self):
        assert clamp_probability(0.5, min_value=0.6, max_value=0.9) == pytest.approx(0.6)


class TestMaternityProbability:
    def test_pregnant_woman_in_range(self):
        p = compute_maternity_probability(28, "pregnancy detected")
        assert PROB_MIN <= p <= PROB_MAX

    def test_non_pregnant_returns_min(self):
        p = compute_maternity_probability(28, "hypertension")
        assert p == pytest.approx(PROB_MIN)

    def test_older_pregnant_higher_than_younger(self):
        p_young = compute_maternity_probability(25, "pregnancy")
        p_old = compute_maternity_probability(32, "pregnancy")
        assert p_old >= p_young


class TestWomenHealthProbability:
    def test_young_woman_returns_zero_base(self):
        p = compute_women_health_probability(25, "", 22.0, 90.0)
        assert p == pytest.approx(PROB_MIN)

    def test_older_woman_higher_probability(self):
        p30 = compute_women_health_probability(30, "", 22.0, 90.0)
        p40 = compute_women_health_probability(40, "", 22.0, 90.0)
        assert p40 >= p30

    def test_bmi_raises_probability(self):
        p_low = compute_women_health_probability(35, "", 22.0, 90.0)
        p_high = compute_women_health_probability(35, "", 28.0, 90.0)
        assert p_high >= p_low

    def test_always_in_bounds(self):
        p = compute_women_health_probability(55, "hyperlipidemia", 35.0, 150.0)
        assert PROB_MIN <= p <= PROB_MAX


class TestMaleHealthProbability:
    def test_young_male_returns_min(self):
        p = compute_male_health_probability(30, "", 120.0, 180.0)
        assert p == pytest.approx(PROB_MIN)

    def test_older_male_gets_positive_probability(self):
        p = compute_male_health_probability(50, "", 120.0, 180.0)
        assert p > PROB_MIN

    def test_hypertension_raises_probability(self):
        p_base = compute_male_health_probability(45, "", 120.0, 180.0)
        p_htn = compute_male_health_probability(45, "hypertension", 150.0, 230.0)
        assert p_htn >= p_base

    def test_always_in_bounds(self):
        p = compute_male_health_probability(70, "hypertension", 160.0, 260.0)
        assert PROB_MIN <= p <= PROB_MAX


class TestKidPackageProbability:
    def test_adult_returns_min(self):
        p = compute_kid_package_probability(20, 22.0, 90.0)
        assert p == pytest.approx(PROB_MIN)

    def test_child_gets_positive_probability(self):
        p = compute_kid_package_probability(8, 16.0, 90.0)
        assert p > PROB_MIN

    def test_younger_child_higher_probability(self):
        p_young = compute_kid_package_probability(6, 16.0, 90.0)
        p_older = compute_kid_package_probability(12, 16.0, 90.0)
        assert p_young >= p_older

    def test_always_in_bounds(self):
        p = compute_kid_package_probability(4, 15.0, 130.0)
        assert PROB_MIN <= p <= PROB_MAX


class TestChronicProbability:
    def test_healthy_young_returns_min(self):
        p = compute_chronic_probability(30, "", 22.0, 90.0, 115.0, 75.0, 5.2, 100.0, 95.0)
        assert p == pytest.approx(PROB_MIN)

    def test_diabetic_gets_elevated_probability(self):
        p = compute_chronic_probability(55, "diabetes", 25.0, 130.0, 120.0, 80.0, 7.0, 100.0, 80.0)
        assert p > 0.3

    def test_multiple_conditions_cumulative(self):
        p_one = compute_chronic_probability(60, "diabetes", 25.0, 90.0, 120.0, 80.0, 5.5, 100.0, 80.0)
        p_multi = compute_chronic_probability(60, "diabetes | hypertension | hyperlipidemia", 28.0, 140.0, 145.0, 92.0, 7.5, 150.0, 55.0)
        assert p_multi >= p_one

    def test_always_in_bounds(self):
        p = compute_chronic_probability(75, "diabetes | hypertension | hyperlipidemia", 35.0, 200.0, 180.0, 110.0, 12.0, 220.0, 15.0)
        assert PROB_MIN <= p <= PROB_MAX


class TestWellnessProbability:
    def test_out_of_age_range_returns_min(self):
        p_too_young = compute_wellness_probability(20, 22.0, 90.0, 180.0, 130.0)
        p_too_old = compute_wellness_probability(65, 22.0, 90.0, 180.0, 130.0)
        assert p_too_young == pytest.approx(PROB_MIN)
        assert p_too_old == pytest.approx(PROB_MIN)

    def test_in_range_adult_gets_probability(self):
        p = compute_wellness_probability(40, 22.0, 90.0, 180.0, 130.0)
        assert p > PROB_MIN

    def test_always_in_bounds(self):
        p = compute_wellness_probability(45, 35.0, 130.0, 260.0, 400.0)
        assert PROB_MIN <= p <= PROB_MAX


class TestGeneralScreeningProbability:
    def test_under_20_returns_min(self):
        p = compute_general_screening_probability(18, 22.0, 110.0, 180.0)
        assert p == pytest.approx(PROB_MIN)

    def test_adult_gets_probability(self):
        p = compute_general_screening_probability(30, 22.0, 110.0, 180.0)
        assert p > PROB_MIN

    def test_always_in_bounds(self):
        p = compute_general_screening_probability(50, 35.0, 160.0, 260.0)
        assert PROB_MIN <= p <= PROB_MAX
