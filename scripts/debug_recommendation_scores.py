import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference_engine import UpsellRecommenderEngine


def format_top_scores(scores: pd.Series, title: str, top_k: int):
    print(f"\n=== {title} ===")
    if scores.empty:
        print("ไม่มีคะแนน")
        return

    for rank, (package_id, score) in enumerate(scores.sort_values(ascending=False).head(top_k).items(), start=1):
        print(f"{rank}. {package_id} -> {score:.4f}")


def build_hybrid_scores(cf_scores: pd.Series, profile_scores: pd.Series, cf_weight: float, profile_weight: float):
    return (cf_weight * cf_scores) + (profile_weight * profile_scores)


def main():
    parser = argparse.ArgumentParser(description="Debug CF/Profile/Hybrid scores for one patient")
    parser.add_argument("--patient-id", dest="patient_id", help="Patient ID to inspect")
    parser.add_argument("--top-k", dest="top_k", type=int, default=10, help="How many rows to print")
    parser.add_argument("--cf-weight", dest="cf_weight", type=float, default=0.7, help="CF weight for hybrid comparison")
    parser.add_argument("--profile-weight", dest="profile_weight", type=float, default=0.3, help="Profile weight for hybrid comparison")
    parser.add_argument("--profile-top-k", dest="profile_top_k", type=int, default=10, help="Number of similar patients to use for profile similarity")
    parser.add_argument("--package-count-weight", dest="package_count_weight", type=float, default=0.25, help="Weight for PACKAGE_COUNT_NORM in profile similarity")
    args = parser.parse_args()

    engine = UpsellRecommenderEngine(
        cf_weight=args.cf_weight,
        profile_weight=args.profile_weight,
        profile_top_k=args.profile_top_k,
        feature_weights={"PACKAGE_COUNT_NORM": args.package_count_weight},
    )
    patient_id = args.patient_id or engine.patient_profiles.index[0]

    profile, purchased_pkgs = engine.get_patient_profile(patient_id)
    if not profile:
        raise SystemExit(f"ไม่พบ patient_id: {patient_id}")

    age = profile["AGE"]
    gender = profile["GENDER"]

    cf_scores = engine._build_cf_scores(purchased_pkgs) if purchased_pkgs else pd.Series(0.0, index=engine.cf_matrix.columns)
    profile_scores = engine._build_profile_scores(patient_id, purchased_pkgs)
    normalized_cf_scores = engine._normalize_scores(cf_scores)
    normalized_profile_scores = engine._normalize_scores(profile_scores)

    cf_valid = engine._apply_clinical_rules(normalized_cf_scores.copy(), age, gender)
    profile_valid = engine._apply_clinical_rules(normalized_profile_scores.copy(), age, gender)
    hybrid_valid = engine._apply_clinical_rules(
        build_hybrid_scores(normalized_cf_scores, normalized_profile_scores, args.cf_weight, args.profile_weight),
        age,
        gender,
    )
    hybrid_7030_valid = engine._apply_clinical_rules(
        build_hybrid_scores(normalized_cf_scores, normalized_profile_scores, 0.7, 0.3),
        age,
        gender,
    )
    hybrid_5050_valid = engine._apply_clinical_rules(
        build_hybrid_scores(normalized_cf_scores, normalized_profile_scores, 0.5, 0.5),
        age,
        gender,
    )

    cf_valid = cf_valid[cf_valid > -50.0]
    profile_valid = profile_valid[profile_valid > -50.0]
    hybrid_valid = hybrid_valid[hybrid_valid > -50.0]
    hybrid_7030_valid = hybrid_7030_valid[hybrid_7030_valid > -50.0]
    hybrid_5050_valid = hybrid_5050_valid[hybrid_5050_valid > -50.0]

    print(f"Patient ID: {patient_id}")
    print(f"Customer Type: {profile.get('CUSTOMER_TYPE', 'existing')}")
    print(f"Age: {age} | Gender: {gender}")
    print(f"Purchased Packages: {', '.join(purchased_pkgs) if purchased_pkgs else 'ไม่มี'}")
    print(f"Hybrid Weights: CF={args.cf_weight:.2f}, Profile={args.profile_weight:.2f}")
    print(f"Profile Similarity Top-K: {args.profile_top_k}")
    print(f"PACKAGE_COUNT_NORM Weight: {args.package_count_weight:.2f}")

    format_top_scores(cf_valid, "CF-only", args.top_k)
    format_top_scores(profile_valid, "Profile-only", args.top_k)
    format_top_scores(hybrid_valid, f"Hybrid {args.cf_weight:.1f} / {args.profile_weight:.1f}", args.top_k)
    format_top_scores(hybrid_7030_valid, "Hybrid 0.7 / 0.3", args.top_k)
    format_top_scores(hybrid_5050_valid, "Hybrid 0.5 / 0.5", args.top_k)


if __name__ == "__main__":
    main()
