"""
Navavej Walk-in Package Recommendation — POC
Cosine Similarity Pipeline (NB10) + Gemini Rerank

Run locally:  python -m streamlit run app_poc.py
Deploy:       Streamlit Community Cloud (set GEMINI_API_KEY in Secrets)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# Load .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path("data")

# ============================================================
# Load Config (from NB10)
# ============================================================
@st.cache_resource
def load_config():
    with open(DATA_DIR / "cosine_recommendation_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    # Convert string keys to int for package_coverage
    pkg_cov = {int(k): v for k, v in cfg["package_coverage"].items()}
    sp = {int(k): v for k, v in cfg["specialized_parents"].items()}
    vp = {int(k): v for k, v in cfg["vaccine_parents"].items()}
    return {
        "need_dims": cfg["need_dims"],
        "checkup_tiers": cfg["checkup_tiers"],
        "specialized_parents": sp,
        "vaccine_parents": vp,
        "package_coverage": pkg_cov,
        "need_reasons_th": cfg["need_reasons_th"],
        "feature_reasons_th": cfg["feature_reasons_th"],
    }


@st.cache_resource
def load_packages():
    return pd.read_csv(DATA_DIR / "navavej_packages_scraped.csv")


@st.cache_resource
def load_pkg_tests():
    """Load canonical test sets per sub_id from NB09 artifacts."""
    import pickle
    with open(DATA_DIR / "content_matching_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
    return arts["pkg_tests"]  # {sub_id: set of canonical test names}


# Main tier sub_id → test list mapping
TIER_TESTS = {
    759: "Standard",
    760: "Gold",
    761: "Platinum Male",
    762: "Platinum Female",
    763: "Diamond Male",
    764: "Diamond Female",
}


@st.cache_resource
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        return None


# ============================================================
# Core Pipeline (ported from NB10)
# ============================================================
def compute_patient_needs(form, need_dims):
    """Map walk-in form → 10-dim health needs vector."""
    age = form.get("age", 30)
    bmi = form.get("bmi", 22)
    is_female = 1 if form.get("gender") == "F" else 0
    is_male = 1 - is_female

    # Disease flags
    htn = form.get("has_hypertension", 0)
    dm = form.get("has_diabetes", 0)
    chol = form.get("has_high_cholesterol", 0)
    heart = form.get("has_heart_disease", 0)
    cancer = form.get("has_cancer_history", 0)
    obesity = 1 if bmi >= 30 else 0

    # Family history
    fam_heart = form.get("family_heart", 0)
    fam_dm = form.get("family_diabetes", 0)
    fam_cancer = form.get("family_cancer", 0)

    # Behavior
    smoking_current = 1 if form.get("smoking_status") == "current" else 0
    smoking_former = 1 if form.get("smoking_status") == "former" else 0
    smoking_any = max(smoking_current, smoking_former)
    exercise_low = 1 if form.get("exercise_days_per_week", 2) <= 1 else 0
    alcohol_regular = 1 if form.get("alcohol") == "regular" else 0
    alcohol_social = 1 if form.get("alcohol") == "social" else 0

    # Age ramps (smooth 0→1 over 10-year bands)
    age_30 = min(max(age - 25, 0) / 10, 1.0)
    age_40 = min(max(age - 35, 0) / 10, 1.0)
    age_50 = min(max(age - 45, 0) / 10, 1.0)
    age_60 = min(max(age - 55, 0) / 10, 1.0)
    bmi_high = min(max(bmi - 23, 0) / 7, 1.0)

    needs = np.zeros(len(need_dims))

    # [0] cardiac
    needs[0] = np.clip(
        heart * 0.30 + htn * 0.15 + chol * 0.10
        + fam_heart * 0.15 + smoking_current * 0.15
        + smoking_former * 0.05 + age_40 * 0.05 + bmi_high * 0.05, 0, 1)

    # [1] diabetes
    needs[1] = np.clip(
        dm * 0.30 + obesity * 0.15 + bmi_high * 0.15
        + fam_dm * 0.15 + exercise_low * 0.05
        + age_40 * 0.10 + chol * 0.05, 0, 1)

    # [2] cancer
    needs[2] = np.clip(
        cancer * 0.30 + fam_cancer * 0.20
        + smoking_current * 0.15 + smoking_former * 0.08
        + age_50 * 0.15 + alcohol_regular * 0.10 + alcohol_social * 0.03
        + obesity * 0.05, 0, 1)

    # [3] women_health
    if is_female:
        if age < 30:
            base, ab = 0.10, age_30 * 0.10
        elif age < 40:
            base, ab = 0.25, age_40 * 0.15
        elif age < 50:
            base, ab = 0.45, age_50 * 0.15
        else:
            base, ab = 0.60, age_60 * 0.10
        needs[3] = np.clip(base + ab + fam_cancer * 0.15 + cancer * 0.15, 0, 1)

    # [4] neuro
    needs[4] = np.clip(
        htn * 0.15 + age_60 * 0.15 + dm * 0.10
        + smoking_current * 0.10 + smoking_former * 0.05
        + heart * 0.10 + fam_heart * 0.05, 0, 1)

    # [5] hormone
    needs[5] = np.clip(
        is_female * age_50 * 0.35 + is_male * age_50 * 0.15
        + age_40 * 0.15 + obesity * 0.10
        + exercise_low * 0.05 + dm * 0.05, 0, 1)

    # [6] bone_joint
    needs[6] = np.clip(
        age_60 * 0.30 + is_female * age_50 * 0.25
        + exercise_low * 0.15 + smoking_current * 0.10
        + smoking_former * 0.05 + obesity * 0.05 + bmi_high * 0.10, 0, 1)

    # [7] respiratory
    needs[7] = np.clip(
        smoking_current * 0.40 + smoking_former * 0.20
        + age_50 * 0.10 + exercise_low * 0.05 + obesity * 0.05
        + fam_cancer * 0.05 + age_60 * 0.15, 0, 1)

    # [8] gi
    needs[8] = np.clip(
        alcohol_regular * 0.30 + alcohol_social * 0.10
        + age_50 * 0.25 + obesity * 0.10
        + smoking_current * 0.08 + smoking_former * 0.03
        + fam_cancer * 0.10 + cancer * 0.15, 0, 1)

    # [9] general
    needs[9] = np.clip(
        0.30 + age_40 * 0.15 + smoking_any * 0.05
        + bmi_high * 0.05 + exercise_low * 0.05, 0, 1)

    return needs


def compute_risk_score(form):
    """Risk score for tier selection."""
    risk = sum(1 for k in [
        "has_hypertension", "has_diabetes", "has_high_cholesterol",
        "has_heart_disease", "has_cancer_history",
    ] if form.get(k, 0))
    risk += 1 if form.get("bmi", 0) >= 30 else 0
    risk += sum(0.5 for k in [
        "family_heart", "family_diabetes", "family_cancer",
    ] if form.get(k, 0))
    if form.get("age", 0) >= 50:
        risk += 1
    return risk


def select_main_package(form, checkup_tiers):
    """Rule-based main tier selection."""
    age = form.get("age", 30)
    gender = form.get("gender", "M")
    risk = compute_risk_score(form)
    g = "f" if gender == "F" else "m"

    if age >= 40 or risk >= 3:
        tier = f"diamond_{g}"
    elif age >= 35 or risk >= 2:
        tier = f"platinum_{g}"
    elif age >= 20:
        tier = "gold"
    else:
        tier = "standard"

    pkg = checkup_tiers[tier]
    reasons = []
    if risk >= 3:
        reasons.append(f"risk score {risk:.0f}")
    if age >= 40:
        reasons.append(f"อายุ {age} ปี")
    if not reasons:
        reasons.append(f"อายุ {age} ปี")

    return {
        "tier": tier,
        "sub_id": pkg["sub_id"],
        "name": pkg["name"],
        "price": pkg["price"],
        "reasons": reasons,
        "risk_score": risk,
    }


def filter_addon_candidates(form, df_pkg, specialized_parents, vaccine_parents):
    """Hard filter: gender, age, special rules → eligible addons."""
    gender = form.get("gender", "M")
    age = form.get("age", 30)

    addon_pids = set(specialized_parents) | set(vaccine_parents)
    eligible = df_pkg[df_pkg.parent_id.isin(addon_pids)].copy()

    # Gender filter
    mask = eligible["target_gender"].apply(
        lambda g: g == "all"
        or (g == "male" and gender == "M")
        or (g == "female" and gender == "F")
    )
    eligible = eligible[mask]

    # Age filter
    eligible = eligible[
        eligible.apply(
            lambda r: (pd.isna(r.get("age_min")) or age >= float(r.get("age_min") or 0))
            and (pd.isna(r.get("age_max")) or age <= float(r.get("age_max") or 999)),
            axis=1,
        )
    ]

    # Special rules
    if gender == "M":
        eligible = eligible[~eligible.parent_id.isin([18850, 18947, 18964, 19017])]
    if age > 45:
        eligible = eligible[eligible.parent_id != 18894]
    if age < 40 and not any(form.get(k, 0) for k in ["has_cancer_history", "family_cancer"]):
        eligible = eligible[eligible.parent_id != 19115]

    # Select median-priced variant per parent
    best_subs = []
    for _, grp in eligible.groupby("parent_id"):
        sorted_grp = grp.sort_values("sale_price")
        best_subs.append(sorted_grp.iloc[len(sorted_grp) // 2])

    if not best_subs:
        return pd.DataFrame()
    return pd.DataFrame(best_subs).reset_index(drop=True)


def rank_addons(form, eligible, cfg, top_k=5):
    """Cosine similarity between patient needs and package coverage."""
    if eligible.empty:
        return []

    need_dims = cfg["need_dims"]
    pkg_cov = cfg["package_coverage"]
    needs = compute_patient_needs(form, need_dims)

    # Use specific dims (exclude 'general')
    specific = [d for d in need_dims if d != "general"]
    idx = [need_dims.index(d) for d in specific]
    p_vec = needs[idx].reshape(1, -1)

    # If patient is very healthy, include general
    if p_vec.sum() < 0.1:
        specific = need_dims
        idx = list(range(len(need_dims)))
        p_vec = needs[idx].reshape(1, -1)

    results = []
    for pid in eligible["parent_id"].unique():
        if pid not in pkg_cov:
            continue
        pkg_vec = np.array([pkg_cov[pid].get(d, 0) for d in specific]).reshape(1, -1)
        if pkg_vec.sum() < 0.01:
            continue
        sim = float(cosine_similarity(p_vec, pkg_vec)[0, 0])
        sub_row = eligible[eligible.parent_id == pid].iloc[0]
        cat = cfg["specialized_parents"].get(pid, cfg["vaccine_parents"].get(pid, ""))
        results.append({
            "parent_id": pid,
            "sub_id": int(sub_row["sub_id"]),
            "category": cat,
            "name": str(sub_row.get("card_name", sub_row.get("parent_name", "")))[:80],
            "price": int(sub_row.get("sale_price", 0)),
            "similarity": round(sim, 4),
        })

    results.sort(key=lambda x: -x["similarity"])
    return results[:top_k]


def explain_addon(form, rec, cfg):
    """Generate Thai explanation for an addon recommendation."""
    pid = rec["parent_id"]
    pkg_cov = cfg["package_coverage"].get(pid, {})
    need_dims = cfg["need_dims"]
    needs = compute_patient_needs(form, need_dims)
    reasons_th = cfg["need_reasons_th"]
    feature_th = cfg["feature_reasons_th"]

    # Top contributing dimensions
    contribs = []
    for i, dim in enumerate(need_dims):
        if needs[i] > 0.1 and pkg_cov.get(dim, 0) > 0.3:
            contribs.append((dim, needs[i] * pkg_cov.get(dim, 0)))
    contribs.sort(key=lambda x: -x[1])
    dim_reasons = [reasons_th.get(c[0], c[0]) for c in contribs[:3]]

    # Patient risk factors
    features = [feature_th[k] for k in feature_th if form.get(k, 0)]

    parts = []
    if dim_reasons:
        parts.append(", ".join(dim_reasons))
    if features:
        parts.append("ปัจจัยเสี่ยง: " + ", ".join(features[:3]))
    return " | ".join(parts) if parts else "ตรวจสุขภาพทั่วไปประจำปี"


def recommend(form, df_pkg, cfg, pkg_tests, top_k=5):
    """Full pipeline: tier → filter → rank → explain."""
    main = select_main_package(form, cfg["checkup_tiers"])
    eligible = filter_addon_candidates(
        form, df_pkg, cfg["specialized_parents"], cfg["vaccine_parents"]
    )
    addons = rank_addons(form, eligible, cfg, top_k=top_k)
    for a in addons:
        a["reason_th"] = explain_addon(form, a, cfg)

    # Get main package test list
    main_tests = sorted(pkg_tests.get(main["sub_id"], set()))

    needs = compute_patient_needs(form, cfg["need_dims"])
    return {
        "main": main,
        "main_tests": main_tests,
        "addons": addons,
        "needs": dict(zip(cfg["need_dims"], needs.round(3).tolist())),
        "eligible_count": len(eligible),
    }


# ============================================================
# Gemini Rerank (optional)
# ============================================================
class AddonRec(BaseModel):
    package_name: str = Field(description="Addon package name in Thai")
    confidence: float = Field(description="Reranked confidence 0-1")
    reason_for_nurse: str = Field(description="Short Thai explanation for nurse")


class GeminiResult(BaseModel):
    summary: str = Field(description="1-2 sentence Thai summary for nurse")
    addons: list[AddonRec] = Field(description="Reranked addons, best first")


def gemini_rerank(form, result, gemini_client):
    gender_th = "หญิง" if form["gender"] == "F" else "ชาย"
    lines = [f"เพศ: {gender_th}", f"อายุ: {form['age']} ปี", f"BMI: {form['bmi']}"]
    conds = []
    for k, th in [
        ("has_hypertension", "ความดันสูง"), ("has_diabetes", "เบาหวาน"),
        ("has_high_cholesterol", "ไขมันสูง"), ("has_heart_disease", "โรคหัวใจ"),
        ("has_cancer_history", "ประวัติมะเร็ง"),
    ]:
        if form.get(k):
            conds.append(th)
    if conds:
        lines.append(f"โรคประจำตัว: {', '.join(conds)}")
    fam = []
    for k, th in [
        ("family_heart", "โรคหัวใจ"), ("family_diabetes", "เบาหวาน"),
        ("family_cancer", "มะเร็ง"),
    ]:
        if form.get(k):
            fam.append(th)
    if fam:
        lines.append(f"ประวัติครอบครัว: {', '.join(fam)}")
    if form.get("smoking_status") == "current":
        lines.append("สูบบุหรี่")
    if form.get("alcohol") == "regular":
        lines.append("ดื่มแอลกอฮอล์เป็นประจำ")

    profile = "\n".join(lines)
    m = result["main"]
    main_tests = result.get("main_tests", [])
    main_tests_text = ", ".join(main_tests) if main_tests else "ไม่มีข้อมูล"

    addons_text = ""
    for i, a in enumerate(result["addons"], 1):
        addons_text += f"{i}. {a['name']} (cosine: {a['similarity']:.2f}, ฿{a['price']:,}) — {a['reason_th']}\n"

    prompt = (
        "คุณเป็นระบบช่วยพยาบาลแนะนำแพ็คเกจตรวจสุขภาพเสริม (add-on) ให้คนไข้ walk-in ของโรงพยาบาลนวเวช\n\n"
        f"## ข้อมูลคนไข้\n{profile}\n\n"
        f"## Main Package\nโปรแกรมดูแลสุขภาพ {m['name']} (฿{m['price']:,})\n"
        f"รายการตรวจที่รวมอยู่แล้ว ({len(main_tests)} รายการ): {main_tests_text}\n\n"
        f"## Add-on Candidates (จาก cosine similarity)\n{addons_text}\n"
        "## คำสั่ง\n"
        "1. Rerank add-on ตามความเหมาะสมทางคลินิก — แนะนำเฉพาะรายการที่เพิ่มเติมจาก main package จริงๆ ไม่ซ้ำซ้อน\n"
        "2. ให้ confidence (0-1) ใหม่\n"
        "3. เขียนเหตุผลสั้นๆ ภาษาไทย สำหรับพยาบาลใช้พูดกับคนไข้\n"
        "4. สรุป 1-2 ประโยค\n"
        "5. อย่าลืมพิจารณาข้อมูลคนไข้และความเหมาะสมของแต่ละแพ็คเกจอย่างรอบคอบ หากไม่สมเหตุสมผลอย่าแนะนำแพ็คเกจนั้นมาเลย"
        "6. แนะนำได้ไม่เกิน 3 แพ็คเกจ และถ้าไม่แนะนำเลยก็ให้บอกว่าแพ็คเกจหลักเหมาะสมแล้ว  ไม่ต้องเพิ่มอะไร"
    )

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": GeminiResult,
            "temperature": 0.3,
        },
    )
    return GeminiResult.model_validate_json(response.text)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Navavej Package Recommendation", page_icon="🏥", layout="wide")
st.title("🏥 ระบบแนะนำแพ็คเกจตรวจสุขภาพ")
st.caption("Cosine Similarity + Gemini Rerank — สำหรับคนไข้ Walk-in")

cfg = load_config()
df_pkg = load_packages()
pkg_tests = load_pkg_tests()
gemini_client = get_gemini_client()

# --- Sidebar ---
with st.sidebar:
    st.header("📋 ข้อมูลคนไข้")

    gender = st.radio("เพศ", ["F", "M"],
                       format_func=lambda x: "หญิง" if x == "F" else "ชาย", horizontal=True)
    age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=40)

    st.markdown("**น้ำหนัก / ส่วนสูง**")
    wc, hc = st.columns(2)
    with wc:
        weight = st.number_input("น้ำหนัก (กก.)", 30.0, 200.0, 65.0, 0.5)
    with hc:
        height = st.number_input("ส่วนสูง (ซม.)", 100.0, 220.0, 165.0, 0.5)
    bmi = weight / ((height / 100) ** 2)
    bmi_label = "ปกติ" if bmi < 23 else "น้ำหนักเกิน" if bmi < 25 else "อ้วน" if bmi < 30 else "อ้วนมาก"
    st.caption(f"BMI = **{bmi:.1f}** ({bmi_label})")

    st.markdown("---")
    st.markdown("**โรคประจำตัว** (เคยได้รับการวินิจฉัย)")
    has_htn = st.checkbox("ความดันโลหิตสูง")
    has_dm = st.checkbox("เบาหวาน")
    has_chol = st.checkbox("ไขมันในเลือดสูง")
    has_heart = st.checkbox("โรคหัวใจ")
    has_cancer = st.checkbox("เคยเป็นมะเร็ง")

    st.markdown("**คนในครอบครัวเคยเป็น**")
    fam_heart = st.checkbox("โรคหัวใจ", key="fh")
    fam_dm = st.checkbox("เบาหวาน", key="fd")
    fam_cancer = st.checkbox("มะเร็ง", key="fc")

    st.markdown("---")
    st.markdown("**พฤติกรรม**")
    smoking = st.radio("สูบบุหรี่", ["never", "former", "current"],
                       format_func=lambda x: {"never": "ไม่สูบ", "former": "เคยสูบ (เลิกแล้ว)", "current": "สูบอยู่"}[x])
    exercise = st.slider("ออกกำลังกาย (วัน/สัปดาห์)", 0, 7, 2)
    alcohol = st.radio("ดื่มแอลกอฮอล์", ["none", "social", "regular"],
                       format_func=lambda x: {"none": "ไม่ดื่ม", "social": "เป็นครั้งคราว", "regular": "เป็นประจำ"}[x])

    use_gemini = st.toggle("ใช้ Gemini อธิบาย", value=True, disabled=gemini_client is None)
    run_btn = st.button("🔍 แนะนำแพ็คเกจ", type="primary", use_container_width=True)

# --- Main ---
if run_btn:
    form = {
        "gender": gender, "age": age, "bmi": round(bmi, 1),
        "smoking_status": smoking, "exercise_days_per_week": exercise, "alcohol": alcohol,
        "has_hypertension": int(has_htn), "has_diabetes": int(has_dm),
        "has_high_cholesterol": int(has_chol), "has_heart_disease": int(has_heart),
        "has_cancer_history": int(has_cancer),
        "family_heart": int(fam_heart), "family_diabetes": int(fam_dm),
        "family_cancer": int(fam_cancer),
    }

    with st.spinner("กำลังวิเคราะห์..."):
        result = recommend(form, df_pkg, cfg, pkg_tests, top_k=7)

    main = result["main"]

    # --- Main Package ---
    st.subheader("📦 แพ็คเกจหลัก")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tier", main["name"])
    c2.metric("ราคา", f"฿{main['price']:,}")
    c3.metric("Risk Score", f"{main['risk_score']:.0f}")

    # --- Health Needs Radar ---
    needs = result["needs"]
    with st.expander("📊 Health Needs Profile"):
        dims = cfg["need_dims"]
        reasons_th = cfg["need_reasons_th"]
        for dim in dims:
            val = needs.get(dim, 0)
            if val > 0.05:
                label = reasons_th.get(dim, dim)
                st.progress(min(val, 1.0), text=f"{label} ({val:.0%})")

    # --- Gemini Rerank ---
    st.markdown("---")
    gemini_result = None
    if use_gemini and gemini_client and result["addons"]:
        with st.spinner("Gemini กำลังวิเคราะห์..."):
            try:
                gemini_result = gemini_rerank(form, result, gemini_client)
            except Exception as e:
                st.error(f"Gemini error: {e}")

    if gemini_result:
        st.subheader("✨ แนะนำ Add-on (Gemini)")
        st.info(f"💬 {gemini_result.summary}")
        for i, a in enumerate(gemini_result.addons, 1):
            with st.container():
                c1, c2 = st.columns([1, 3])
                c1.markdown(f"**{i}. {a.package_name}**")
                c1.progress(a.confidence, text=f"{a.confidence:.0%}")
                c2.markdown(f"🗣️ _{a.reason_for_nurse}_")
    else:
        st.subheader("📋 แนะนำ Add-on")
        for i, a in enumerate(result["addons"][:5], 1):
            with st.container():
                c1, c2 = st.columns([2, 3])
                c1.markdown(f"**{i}. {a['name']}**")
                c1.caption(f"฿{a['price']:,} | cosine: {a['similarity']:.2f}")
                c2.markdown(f"_{a['reason_th']}_")

    # --- Details ---
    with st.expander("🔬 Cosine Ranking (ก่อน Gemini)"):
        for i, a in enumerate(result["addons"], 1):
            st.write(f"{i}. **{a['name']}** — sim={a['similarity']:.3f} ฿{a['price']:,}")
            st.caption(f"   {a['reason_th']}")

    with st.expander("👤 ข้อมูลคนไข้"):
        st.json(form)

elif not run_btn:
    st.info("👈 กรอกข้อมูลคนไข้ทางซ้าย แล้วกด **แนะนำแพ็คเกจ**")
