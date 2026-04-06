"""
Navavej Walk-in Package Recommendation — POC (Notebook 12)
3-Layer Pipeline: Rule-Based → K-Means → XGBoost + Gemini Reranker

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
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Load .env locally, fall back to st.secrets on Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# Constants
# ============================================================
DATA_DIR = Path("data")

ADDON_NAMES = {
    18860: "โปรแกรมตรวจสุขภาพหัวใจ",
    18894: "ตรวจความพร้อมก่อนสมรส/วางแผนมีบุตร",
    18895: "ตรวจสมรรถภาพปอด",
    18900: "คัดกรองสารก่อภูมิแพ้ (เจาะเลือด)",
    18903: "คัดกรองโรคไทรอยด์",
    18905: "คัดกรองโรคหลอดเลือดสมอง",
    18922: "ตรวจภาวะกระดูกพรุน",
    18964: "คัดกรองมะเร็งปากมดลูก",
    18994: "โปรแกรมควบคุมน้ำหนัก",
    19002: "คัดกรองมะเร็ง",
    19006: "คัดกรองมะเร็ง (ชุดเล็ก)",
    19009: "คัดกรองโรคเบาหวาน",
    19016: "ทดสอบสารก่อภูมิแพ้ (ผิวหนัง)",
    19017: "คัดกรองมะเร็งเต้านม",
    19049: "เช็คสุขภาพหัวใจ S-PATCH EX",
    19102: "ตรวจหัวใจเต้นผิดจังหวะ",
    19115: "ส่องกล้องทางเดินอาหาร",
    19124: "MRI สมอง",
}

EXCLUDED_ADDONS = {19115}  # ส่องกล้อง — filter out

TIER_INFO = {
    "standard": {"name": "Standard", "price": 2_900, "color": "#6c757d"},
    "gold": {"name": "Gold", "price": 6_900, "color": "#ffc107"},
    "platinum_m": {"name": "Platinum (ชาย)", "price": 10_900, "color": "#adb5bd"},
    "platinum_f": {"name": "Platinum (หญิง)", "price": 16_900, "color": "#adb5bd"},
    "diamond_m": {"name": "Diamond (ชาย)", "price": 17_900, "color": "#0dcaf0"},
    "diamond_f": {"name": "Diamond (หญิง)", "price": 22_900, "color": "#0dcaf0"},
}

FEATURE_COLS = [
    "gender", "age", "bmi", "marital_status",
    "smoking_status", "exercise_level", "alcohol_use",
    "has_hypertension", "has_diabetes", "has_high_cholesterol",
    "has_heart_disease", "has_cancer_history",
    "family_heart", "family_diabetes", "family_cancer",
    "checkup_frequency", "years_since_last_checkup",
]
CAT_COLS = ["gender", "marital_status", "smoking_status", "exercise_level", "alcohol_use"]

CONFIDENCE_THRESHOLD = 0.5
MIN_POSITIVE = 10
N_CLUSTERS = 5


# ============================================================
# Gemini Schema
# ============================================================
class AddonRecommendation(BaseModel):
    package_id: int = Field(description="Addon package ID")
    package_name: str = Field(description="Addon package name in Thai")
    confidence: float = Field(description="Reranked confidence 0-1")
    reason_for_nurse: str = Field(description="Short Thai explanation for nurse")


class RerankedResult(BaseModel):
    main_tier: str = Field(description="Main package tier")
    addons: list[AddonRecommendation] = Field(description="Reranked addons, best first")
    summary: str = Field(description="1-2 sentence Thai summary for nurse")


# ============================================================
# Load & Train (cached)
# ============================================================
@st.cache_resource
def load_and_train():
    """Load data, train pipeline, return all components."""
    df = pd.read_csv(DATA_DIR / "training_table_gemini_labels.csv")

    with open(DATA_DIR / "gemini_realtime_progress.json", encoding="utf-8") as f:
        llm_raw = json.load(f)

    # Parse LLM labels
    records = []
    for idx_str, val in llm_raw.items():
        parsed = json.loads(val["raw"])
        row = {"patient_idx": int(idx_str)}
        for addon in parsed["addons"]:
            row[f"addon_{addon['parent_id']}"] = addon["confidence"]
        records.append(row)
    llm_df = pd.DataFrame(records).fillna(0.0).sort_values("patient_idx").reset_index(drop=True)

    # Encode
    df_enc = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURE_COLS].values

    # Scaler + KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)

    X_full = np.column_stack([X, clusters])

    # Labels
    addon_cols = [c for c in llm_df.columns if c.startswith("addon_")]
    Y = (llm_df[addon_cols].values >= CONFIDENCE_THRESHOLD).astype(int)
    addon_ids = [int(c.split("_")[1]) for c in addon_cols]

    valid_mask = Y.sum(axis=0) >= MIN_POSITIVE
    Y_filtered = Y[:, valid_mask]
    addon_ids_filtered = [aid for aid, v in zip(addon_ids, valid_mask) if v]

    # Train
    X_train, _, Y_train, _ = train_test_split(X_full, Y_filtered, test_size=0.2, random_state=42)
    base_xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss",
    )
    model = MultiOutputClassifier(base_xgb, n_jobs=-1)
    model.fit(X_train, Y_train)

    return {
        "model": model,
        "encoders": encoders,
        "scaler": scaler,
        "kmeans": kmeans,
        "addon_ids": addon_ids_filtered,
    }


@st.cache_resource
def get_gemini_client():
    # Try: env var → st.secrets → None
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
        st.warning("google-genai not installed — Gemini rerank disabled")
        return None


# ============================================================
# Pipeline Functions
# ============================================================
def assign_tier(age, gender, risk):
    if age >= 40 or risk >= 3:
        return f"diamond_{'f' if gender == 'F' else 'm'}"
    if age >= 35:
        return f"platinum_{'f' if gender == 'F' else 'm'}"
    if age >= 20:
        return "gold"
    return "standard"


def recommend(patient, bundle, top_k=7):
    model = bundle["model"]
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    addon_ids = bundle["addon_ids"]

    # Risk score
    risk = sum([
        patient.get("has_hypertension", 0),
        patient.get("has_diabetes", 0),
        patient.get("has_high_cholesterol", 0),
        patient.get("has_heart_disease", 0),
        patient.get("has_cancer_history", 0),
        patient.get("family_heart", 0),
        patient.get("family_diabetes", 0),
        patient.get("family_cancer", 0),
        1 if patient.get("bmi", 0) >= 30 else 0,
        1 if patient.get("smoking_status", "never") == "current" else 0,
    ])

    tier = assign_tier(patient["age"], patient["gender"], risk)

    # Encode
    row = []
    for col in FEATURE_COLS:
        val = patient[col]
        if col in encoders:
            val = encoders[col].transform([str(val)])[0]
        row.append(val)

    cluster = kmeans.predict(scaler.transform([row]))[0]
    x_input = np.array(row + [cluster]).reshape(1, -1)

    probs = np.array([est.predict_proba(x_input)[0, 1] for est in model.estimators_])
    ranked = np.argsort(probs)[::-1]

    addons = []
    for idx in ranked:
        aid = addon_ids[idx]
        if aid in EXCLUDED_ADDONS:
            continue
        addons.append({
            "package_id": aid,
            "package_name": ADDON_NAMES.get(aid, str(aid)),
            "confidence": float(probs[idx]),
        })
        if len(addons) >= top_k:
            break

    return {
        "main_tier": tier,
        "main_price": TIER_INFO[tier]["price"],
        "cluster": int(cluster),
        "risk_score": risk,
        "addons": addons,
    }


def rerank_gemini(patient, xgb_result, gemini_client):
    gender_th = "หญิง" if patient["gender"] == "F" else "ชาย"
    lines = [
        f"เพศ: {gender_th}", f"อายุ: {patient['age']} ปี",
        f"BMI: {patient['bmi']}", f"สถานภาพ: {patient['marital_status']}",
        f"สูบบุหรี่: {patient['smoking_status']}",
        f"ออกกำลังกาย: {patient['exercise_level']}",
        f"ดื่มแอลกอฮอล์: {patient['alcohol_use']}",
    ]
    conds = []
    if patient.get("has_hypertension"): conds.append("ความดันโลหิตสูง")
    if patient.get("has_diabetes"): conds.append("เบาหวาน")
    if patient.get("has_high_cholesterol"): conds.append("คอเลสเตอรอลสูง")
    if patient.get("has_heart_disease"): conds.append("โรคหัวใจ")
    if patient.get("has_cancer_history"): conds.append("ประวัติมะเร็ง")
    if conds:
        lines.append(f"โรคประจำตัว: {', '.join(conds)}")
    fam = []
    if patient.get("family_heart"): fam.append("โรคหัวใจ")
    if patient.get("family_diabetes"): fam.append("เบาหวาน")
    if patient.get("family_cancer"): fam.append("มะเร็ง")
    if fam:
        lines.append(f"ประวัติครอบครัว: {', '.join(fam)}")
    lines.append(f"ความถี่ตรวจ: {patient['checkup_frequency']:.1f} ครั้ง/ปี")
    lines.append(f"ไม่ได้ตรวจมา: {patient['years_since_last_checkup']:.0f} ปี")

    profile = "\n".join(lines)
    candidates = ""
    for i, a in enumerate(xgb_result["addons"], 1):
        candidates += f"{i}. [{a['package_id']}] {a['package_name']} (ML: {a['confidence']:.0%})\n"

    prompt = (
        "คุณเป็นระบบช่วยพยาบาลแนะนำแพ็คเกจตรวจสุขภาพเสริม (add-on) ให้คนไข้ walk-in\n\n"
        f"## ข้อมูลคนไข้\n{profile}\n\n"
        f"## Main Package\n{xgb_result['main_tier'].upper()} (B{xgb_result['main_price']:,})\n\n"
        f"## Addon Candidates จาก ML\n{candidates}\n"
        "## คำสั่ง\n"
        "1. Rerank addon packages ตามความเหมาะสมกับคนไข้รายนี้\n"
        "2. ให้ confidence score (0-1) ใหม่\n"
        "3. สร้างเหตุผลสั้นๆ ภาษาไทย สำหรับพยาบาลใช้อธิบายให้คนไข้\n"
        "4. สรุป 1-2 ประโยค ว่าทำไมถึงแนะนำ package เหล่านี้\n"
    )

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RerankedResult,
            "temperature": 0.3,
        },
    )
    return RerankedResult.model_validate_json(response.text)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Navavej Package Recommendation", page_icon="🏥", layout="wide")

st.title("🏥 ระบบแนะนำแพ็คเกจตรวจสุขภาพ")
st.caption("POC — 3-Layer Pipeline: Rule-Based → K-Means → XGBoost + Gemini Reranker")

# Load model
with st.spinner("กำลังโหลด model..."):
    bundle = load_and_train()
gemini_client = get_gemini_client()

# --- Sidebar: Patient Form ---
with st.sidebar:
    st.header("📋 ข้อมูลคนไข้ Walk-in")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("เพศ", ["F", "M"], format_func=lambda x: "หญิง" if x == "F" else "ชาย")
    with col2:
        age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=40)

    bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=24.0, step=0.1)
    marital = st.selectbox("สถานภาพ", ["single", "married", "unknown"],
                           format_func=lambda x: {"single": "โสด", "married": "สมรส", "unknown": "ไม่ระบุ"}[x])

    st.subheader("พฤติกรรม")
    smoking = st.selectbox("สูบบุหรี่", ["never", "former", "current"],
                           format_func=lambda x: {"never": "ไม่เคย", "former": "เคย (เลิกแล้ว)", "current": "สูบอยู่"}[x])
    exercise = st.selectbox("ออกกำลังกาย", ["none", "light", "moderate", "vigorous"],
                            format_func=lambda x: {"none": "ไม่ออก", "light": "เบา", "moderate": "ปานกลาง", "vigorous": "หนัก"}[x])
    alcohol = st.selectbox("ดื่มแอลกอฮอล์", ["none", "social", "moderate", "heavy"],
                           format_func=lambda x: {"none": "ไม่ดื่ม", "social": "สังสรรค์", "moderate": "ปานกลาง", "heavy": "หนัก"}[x])

    st.subheader("โรคประจำตัว")
    has_htn = st.checkbox("ความดันโลหิตสูง")
    has_dm = st.checkbox("เบาหวาน")
    has_chol = st.checkbox("คอเลสเตอรอลสูง")
    has_heart = st.checkbox("โรคหัวใจ")
    has_cancer = st.checkbox("ประวัติมะเร็ง")

    st.subheader("ประวัติครอบครัว")
    fam_heart = st.checkbox("ครอบครัว — โรคหัวใจ")
    fam_dm = st.checkbox("ครอบครัว — เบาหวาน")
    fam_cancer = st.checkbox("ครอบครัว — มะเร็ง")

    st.subheader("การตรวจสุขภาพ")
    checkup_freq = st.slider("ความถี่ตรวจ (ครั้ง/ปี)", 0.0, 3.0, 1.0, 0.1)
    years_since = st.slider("ไม่ได้ตรวจมา (ปี)", 0.0, 15.0, 2.0, 0.5)

    use_gemini = st.checkbox("ใช้ Gemini Rerank", value=True, disabled=gemini_client is None,
                             help="ต้องมี GEMINI_API_KEY ใน .env")

    run_btn = st.button("🔍 วิเคราะห์และแนะนำ", type="primary", use_container_width=True)

# --- Main Area ---
if run_btn:
    patient = {
        "gender": gender, "age": age, "bmi": bmi,
        "marital_status": marital, "smoking_status": smoking,
        "exercise_level": exercise, "alcohol_use": alcohol,
        "has_hypertension": int(has_htn), "has_diabetes": int(has_dm),
        "has_high_cholesterol": int(has_chol), "has_heart_disease": int(has_heart),
        "has_cancer_history": int(has_cancer),
        "family_heart": int(fam_heart), "family_diabetes": int(fam_dm),
        "family_cancer": int(fam_cancer),
        "checkup_frequency": checkup_freq, "years_since_last_checkup": years_since,
    }

    # XGBoost recommendation
    with st.spinner("กำลังวิเคราะห์..."):
        xgb_result = recommend(patient, bundle, top_k=7)

    tier = xgb_result["main_tier"]
    tier_data = TIER_INFO[tier]

    # --- Main Package ---
    st.markdown("---")
    st.subheader("📦 Main Package")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Tier", tier_data["name"])
    mc2.metric("ราคา", f"฿{tier_data['price']:,}")
    mc3.metric("Risk Score", f"{xgb_result['risk_score']} / 10")

    # --- Gemini Rerank ---
    gemini_result = None
    if use_gemini and gemini_client:
        with st.spinner("Gemini กำลัง rerank และสร้างคำอธิบาย..."):
            try:
                gemini_result = rerank_gemini(patient, xgb_result, gemini_client)
            except Exception as e:
                st.error(f"Gemini error: {e}")

    # --- Addon Packages ---
    st.markdown("---")
    if gemini_result:
        st.subheader("✨ แนะนำ Add-on Packages (Gemini Reranked)")
        st.info(f"💬 **สรุป:** {gemini_result.summary}")

        for i, addon in enumerate(gemini_result.addons):
            conf_pct = int(addon.confidence * 100)
            with st.container():
                col_rank, col_name, col_bar, col_reason = st.columns([0.5, 2.5, 1.5, 4])
                col_rank.markdown(f"### {i + 1}")
                col_name.markdown(f"**{addon.package_name}**")
                col_bar.progress(addon.confidence, text=f"{conf_pct}%")
                col_reason.markdown(f"🗣️ _{addon.reason_for_nurse}_")
    else:
        st.subheader("📋 แนะนำ Add-on Packages (XGBoost)")
        for i, addon in enumerate(xgb_result["addons"][:5]):
            conf_pct = int(addon["confidence"] * 100)
            with st.container():
                col_rank, col_name, col_bar = st.columns([0.5, 3, 2])
                col_rank.markdown(f"### {i + 1}")
                col_name.markdown(f"**{addon['package_name']}**")
                col_bar.progress(addon["confidence"], text=f"{conf_pct}%")

    # --- XGBoost Raw (expander) ---
    with st.expander("🔬 XGBoost Raw Output (ก่อน Gemini rerank)"):
        for i, addon in enumerate(xgb_result["addons"], 1):
            st.write(f"{i}. **{addon['package_name']}** — {addon['confidence']:.0%}")

    # --- Patient Profile (expander) ---
    with st.expander("👤 Patient Profile"):
        st.json(patient)
        st.write(f"**Cluster:** {xgb_result['cluster']}")
