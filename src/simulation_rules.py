def clamp_probability(value, min_value=0.02, max_value=0.95):
    return max(min(float(value), max_value), min_value)


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def compute_maternity_probability(age, conditions):
    probability = 0.0
    if 18 <= age <= 45 and ("pregnancy" in conditions or "prenatal" in conditions):
        probability = 0.72
        if age >= 30:
            probability += 0.05
    return clamp_probability(probability)


def compute_women_health_probability(age, conditions, bmi, glucose):
    probability = 0.0
    if age >= 30:
        probability = 0.35
        if age >= 35:
            probability += 0.12
        if bmi >= 27:
            probability += 0.05
        if glucose >= 110:
            probability += 0.04
        if "hyperlipidemia" in conditions:
            probability += 0.04
    return clamp_probability(probability)


def compute_male_health_probability(age, conditions, sys_bp, cholesterol):
    probability = 0.0
    if age >= 40:
        probability = 0.38
        if age >= 50:
            probability += 0.10
        if sys_bp >= 135:
            probability += 0.07
        if cholesterol >= 220:
            probability += 0.05
        if "hypertension" in conditions:
            probability += 0.05
    return clamp_probability(probability)


def compute_kid_package_probability(age, bmi, glucose):
    probability = 0.0
    if age <= 14:
        probability = 0.45
        if age <= 8:
            probability += 0.12
        if bmi < 18.5:
            probability += 0.04
        if glucose >= 110:
            probability += 0.03
    return clamp_probability(probability)


def compute_chronic_probability(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr):
    probability = 0.0
    if age >= 50 or "diabetes" in conditions or "hypertension" in conditions or "hyperlipidemia" in conditions:
        probability = 0.28
        if age >= 60:
            probability += 0.10
        if "diabetes" in conditions:
            probability += 0.12
        if "hypertension" in conditions:
            probability += 0.10
        if "hyperlipidemia" in conditions:
            probability += 0.08
        if bmi >= 27:
            probability += 0.05
        if glucose >= 126:
            probability += 0.08
        if sys_bp >= 140 or dia_bp >= 90:
            probability += 0.06
        if hba1c >= 6.5:
            probability += 0.08
        if ldl >= 140:
            probability += 0.05
        if egfr <= 60:
            probability += 0.05
    return clamp_probability(probability)


def compute_wellness_probability(age, bmi, glucose, cholesterol, triglycerides):
    probability = 0.0
    if 25 <= age <= 60:
        probability = 0.14
        if bmi >= 27:
            probability += 0.04
        if glucose >= 110:
            probability += 0.03
        if cholesterol >= 220:
            probability += 0.03
        if triglycerides >= 180:
            probability += 0.03
    return clamp_probability(probability)


def compute_general_screening_probability(age, bmi, sys_bp, cholesterol):
    probability = 0.0
    if age >= 20:
        probability = 0.18
        if age >= 35:
            probability += 0.05
        if bmi >= 27:
            probability += 0.03
        if sys_bp >= 135:
            probability += 0.03
        if cholesterol >= 220:
            probability += 0.03
    return clamp_probability(probability)
