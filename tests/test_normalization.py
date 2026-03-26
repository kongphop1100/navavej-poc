"""Tests for shared helper functions: normalize_range and has_any_keyword."""
import pytest

from src.inference_engine import has_any_keyword, normalize_range


class TestNormalizeRange:
    def test_midpoint(self):
        assert normalize_range(50, 0, 100) == pytest.approx(0.5)

    def test_at_minimum_returns_zero(self):
        assert normalize_range(0, 0, 100) == pytest.approx(0.0)

    def test_at_maximum_returns_one(self):
        assert normalize_range(100, 0, 100) == pytest.approx(1.0)

    def test_below_minimum_clamped_to_zero(self):
        assert normalize_range(-10, 0, 100) == pytest.approx(0.0)

    def test_above_maximum_clamped_to_one(self):
        assert normalize_range(999, 0, 100) == pytest.approx(1.0)

    def test_float_precision(self):
        result = normalize_range(15.0, 15.0, 40.0)  # BMI at minimum
        assert result == pytest.approx(0.0)

    def test_string_value_coerced(self):
        # vitals stored as strings from dataset_snapshot safe_get()
        result = normalize_range("25.0", 15.0, 40.0)
        assert 0.0 <= result <= 1.0


class TestHasAnyKeyword:
    def test_keyword_present_returns_one(self):
        assert has_any_keyword("patient has diabetes", ["diabetes"]) == 1.0

    def test_keyword_absent_returns_zero(self):
        assert has_any_keyword("patient has hypertension", ["diabetes"]) == 0.0

    def test_multiple_keywords_any_match(self):
        assert has_any_keyword("obesity noted", ["obesity", "overweight"]) == 1.0

    def test_empty_text(self):
        assert has_any_keyword("", ["diabetes"]) == 0.0

    def test_empty_keywords(self):
        assert has_any_keyword("diabetes", []) == 0.0
