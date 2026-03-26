import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
import uvicorn

from src.inference_engine import UpsellRecommenderEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PATIENT_LIST = 200

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_engine_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release on shutdown."""
    logger.info("กำลังโหลดโมเดล AI (Model Artifacts)...")
    try:
        _engine_state["engine"] = UpsellRecommenderEngine()
        logger.info("โหลด AI สำเร็จ พร้อมให้บริการ")
    except Exception as exc:
        logger.error("โหลดโมเดลล้มเหลว: %s", exc)
        _engine_state["engine"] = None
    yield
    _engine_state.clear()
    logger.info("Server shutdown: engine released")


app = FastAPI(title="Navavej Upsell API (The Brain)", version="1.0.0", lifespan=lifespan)


def _get_engine() -> UpsellRecommenderEngine:
    engine = _engine_state.get("engine")
    if not engine:
        raise HTTPException(status_code=503, detail="AI Model is down!")
    return engine


# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
def healthcheck():
    engine = _engine_state.get("engine")
    return {
        "status": "ok" if engine else "degraded",
        "model_loaded": engine is not None,
    }


@app.get("/api/v1/patients")
def get_patients():
    engine = _get_engine()

    patient_ids = engine.patient_profiles.index.tolist()[:MAX_PATIENT_LIST]

    patient_list = []
    for pid in patient_ids:
        profile, _ = engine.get_patient_profile(pid)
        if profile:
            g_th = "ชาย" if profile["GENDER"] == "M" else "หญิง"
            cond_txt = str(profile["CONDITIONS"]).split(",")[0][:25]
            if cond_txt in ("nan", "None"):
                cond_txt = "ไม่มีโรค"

            customer_type = profile.get("CUSTOMER_TYPE", "existing")
            suffix = "ลูกค้าใหม่" if customer_type == "new" else "ลูกค้าเก่า"
            label = f"{suffix} {g_th} {profile['AGE']} ปี ({cond_txt}) [{pid[:5]}]"
            patient_list.append({"id": pid, "label": label, "customer_type": customer_type})

    return {"patients": patient_list}


@app.get("/api/v1/recommend")
def get_recommendation(
    patient_id: str = Query(..., min_length=1, max_length=100, strip_whitespace=True),
):
    engine = _get_engine()

    profile, purchased_history = engine.get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail="ค้นหาประวัติคนไข้ไม่พบ (ไม่เคยมาโรงพยาบาล?)")

    top_3 = engine.recommend_with_explanations(patient_id)

    recommendations = []
    for rec in top_3:
        pkg_id = rec["package_id"]
        score = rec["final_score"]
        if score > -99:
            package_matches = engine.packages_df[engine.packages_df["code"] == pkg_id]
            pkg_name = str(package_matches["name"].values[0]) if not package_matches.empty else str(pkg_id)
            recommendations.append(
                {
                    "package_id": str(pkg_id),
                    "package_name": pkg_name,
                    "confidence_score": float(score),
                    "cf_score": float(rec["cf_score"]),
                    "profile_score": float(rec["profile_score"]),
                    "recommendation_mode": str(rec["recommendation_mode"]),
                    "reason": str(rec["reason"]),
                }
            )

    raw_cond = str(profile["CONDITIONS"])
    cond_display = raw_cond[:60] + "..." if raw_cond.lower() != "nan" else "ไม่มีโรคประจำตัว"

    return {
        "status": "success",
        "patient": {
            "id": str(patient_id),
            "age": int(profile["AGE"]),
            "gender": str(profile["GENDER"]),
            "conditions": cond_display,
            "vitals": profile.get("VITALS", {}),
            "customer_type": str(profile.get("CUSTOMER_TYPE", "existing")),
        },
        "purchased_packages": [str(p) for p in purchased_history],
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
