def clamp_probability(value, min_value=0.02, max_value=0.95):
    return max(min(float(value), max_value), min_value)


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def midpoint(low, high):
    return round((float(low) + float(high)) / 2.0, 1)


def has_any_keyword(text, keywords):
    return any(keyword in text for keyword in keywords)


def conditions_flag(conditions, *keywords):
    return has_any_keyword(conditions, keywords)


def compute_chronic_risk_level(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr):
    risk = 0
    if age >= 50:
        risk += 1
    if age >= 60:
        risk += 1
    if "diabetes" in conditions:
        risk += 2
    if "hypertension" in conditions:
        risk += 2
    if "hyperlipidemia" in conditions:
        risk += 1
    if bmi >= 27:
        risk += 1
    if glucose >= 126:
        risk += 2
    if sys_bp >= 140 or dia_bp >= 90:
        risk += 2
    if hba1c >= 6.5:
        risk += 2
    if ldl >= 140:
        risk += 1
    if egfr <= 60:
        risk += 1
    return risk


def compute_general_screening_risk(age, bmi, sys_bp, cholesterol):
    risk = 0
    if age >= 20:
        risk += 1
    if age >= 35:
        risk += 1
    if age >= 50:
        risk += 1
    if bmi >= 27:
        risk += 1
    if sys_bp >= 135:
        risk += 1
    if cholesterol >= 220:
        risk += 1
    return risk


def compute_wellness_risk(age, bmi, glucose, cholesterol, triglycerides):
    risk = 0
    if 25 <= age <= 60:
        risk += 1
    if bmi >= 27:
        risk += 1
    if glucose >= 110:
        risk += 1
    if cholesterol >= 220:
        risk += 1
    if triglycerides >= 180:
        risk += 1
    return risk


def assign_women_packages(age, conditions, bmi, glucose):
    packages = set()
    if not (15 <= age <= 80):
        return packages

    if 18 <= age <= 45 and ("pregnancy" in conditions or "prenatal" in conditions):
        packages.update({"NVV-PK-0007", "NVV-PK-0061"})

    if 15 <= age <= 45:
        packages.add("NVV-PK-0059")
    if age >= 30:
        packages.add("NVV-PK-0035")
    if age >= 30 and (bmi >= 27 or glucose >= 110 or "hyperlipidemia" in conditions):
        packages.add("NVV-PK-0036")
    if age >= 35:
        packages.add("NVV-PK-0015")
    if age >= 25 and (bmi >= 27 or "obesity" in conditions or "overweight" in conditions):
        packages.add("NVV-PK-0028")

    return packages


def assign_male_packages(age, conditions, sys_bp, cholesterol):
    packages = set()
    if age >= 40:
        packages.add("NVV-PK-0014")
    if 40 <= age <= 55 and (
        sys_bp >= 135 or cholesterol >= 220 or "hypertension" in conditions
    ):
        packages.add("NVV-PK-0040")
    return packages


def assign_pediatric_packages(age, bmi, glucose):
    packages = set()
    if age > 14:
        return packages

    packages.add("NVV-PK-0092")
    if age <= 8 or bmi < 18.5 or glucose >= 110:
        packages.add("NVV-PK-0084")
    if glucose >= 126 or bmi < 16.5:
        packages.add("NVV-PK-0082")
    return packages


def assign_senior_packages(age, conditions, sys_bp):
    packages = set()
    if age >= 55:
        packages.add("NVV-PK-0017")
    if age >= 60 or "stroke" in conditions or "cerebrovascular" in conditions:
        packages.add("NVV-PK-0018")
    if age >= 65 or sys_bp >= 140 or "stroke" in conditions:
        packages.add("NVV-PK-0003")
    return packages


def assign_chronic_packages(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr):
    packages = set()
    risk = compute_chronic_risk_level(age, conditions, bmi, glucose, sys_bp, dia_bp, hba1c, ldl, egfr)
    if risk < 2:
        return packages

    packages.add("NVV-PK-0002")
    if risk >= 3:
        packages.add("NVV-PK-0078")
    if risk >= 4:
        packages.add("NVV-PK-0081")
    if risk >= 5:
        packages.add("NVV-PK-0064")
    return packages


def assign_wellness_packages(age, bmi, glucose, cholesterol, triglycerides):
    packages = set()
    risk = compute_wellness_risk(age, bmi, glucose, cholesterol, triglycerides)
    if risk < 1:
        return packages

    packages.add("NVV-PK-0046")
    if risk >= 2:
        packages.add("NVV-PK-0047")
    if risk >= 3:
        packages.add("NVV-PK-0083")
        packages.add("NVV-PK-0062")
    if risk >= 4:
        packages.add("NVV-PK-0048")
    return packages


def assign_general_screening_packages(age, bmi, sys_bp, cholesterol):
    packages = set()
    risk = compute_general_screening_risk(age, bmi, sys_bp, cholesterol)
    if risk < 1:
        return packages

    packages.update({"NVV-PK-0031", "NVV-PK-0044"})
    if risk >= 2:
        packages.add("NVV-PK-0004")
    if risk >= 3:
        packages.add("NVV-PK-0001")
        packages.add("NVV-PK-0072")
    if risk >= 4:
        packages.add("NVV-PK-0098")
        packages.add("NVV-PK-0045")
    return packages
