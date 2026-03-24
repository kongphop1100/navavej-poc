from fastapi import FastAPI, HTTPException
import uvicorn

from src.inference_engine import UpsellRecommenderEngine

app = FastAPI(title="Navavej Upsell API (The Brain)", version="1.0.0")

# ==========================================
# 1. โหลดโมเดล 1 ครั้งถ้วนตอนสตาร์ทเซิร์ฟเวอร์
# ==========================================
print("⏳ กำลังโหลดก้อนสมอง AI (Model Artifacts)...")
try:
    engine = UpsellRecommenderEngine()
    print("✅ โหลด AI ทะลุเข้า RAM สำเร็จ พร้อมรับลูกค้า!")
except Exception as e:
    print(f"❌ เซิร์ฟเวอร์ล่ม: {e}")
    engine = None


@app.get("/health")
def healthcheck():
    return {
        "status": "ok" if engine else "degraded",
        "model_loaded": engine is not None,
    }


# ==========================================
# 2. Endpoint: ดึงรายชื่อคนไข้ (ส่งไปให้แอปหน้าบ้านทำ Dropdown)
# ==========================================
@app.get("/api/v1/patients")
def get_patients():
    if not engine:
        raise HTTPException(status_code=503, detail="AI Model is down!")
    
    patient_ids = engine.txn_df['PATIENT'].unique().tolist()[:200]
    
    patient_list = []
    for pid in patient_ids:
        profile, _ = engine.get_patient_profile(pid)
        if profile:
            g_th = 'ชาย' if profile['GENDER'] == 'M' else 'หญิง'
            # ตัดชื่อโรคยาวๆ ให้สั้นลง
            cond_txt = str(profile['CONDITIONS']).split(',')[0][:25] 
            if cond_txt == 'nan' or cond_txt == 'None': cond_txt = 'ไม่มีโรค'
            
            label = f"ผู้ป่วย{g_th} {profile['AGE']} ปี ({cond_txt}) [{pid[:5]}]"
            patient_list.append({"id": pid, "label": label})
            
    return {"patients": patient_list}


# ==========================================
# 3. Endpoint: แม่เหล็กดูดเงิน (ระบบแนะนำ Upsell)
# ==========================================
@app.get("/api/v1/recommend")
def get_recommendation(patient_id: str):
    if not engine:
        raise HTTPException(status_code=503, detail="AI Model is down!")
        
    profile, purchased_history = engine.get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail="ค้นหาประวัติคนไข้ไม่พบ (ไม่เคยมาโรงพยาบาล?)")
        
    # วิ่งทะลุ 3 Layers ของเรา
    top_3 = engine.recommend(patient_id)
    
    # ห่อข้อมูลส่งกลับเป็น JSON (ต้องแปลง type ของ NumPy เป็น Native Python)
    recommendations = []
    for pkg_id, score in top_3.items():
        if score > -99:
            package_matches = engine.packages_df[engine.packages_df['code'] == pkg_id]
            pkg_name = str(package_matches['name'].values[0]) if not package_matches.empty else str(pkg_id)
            recommendations.append({
                "package_id": str(pkg_id),
                "package_name": pkg_name,
                "confidence_score": float(score)
            })
            
    raw_cond = str(profile['CONDITIONS'])
    cond_display = raw_cond[:60] + "..." if raw_cond.lower() != 'nan' else 'ไม่มีโรคประจำตัว'

    return {
        "status": "success",
        "patient": {
            "id": str(patient_id),
            "age": int(profile['AGE']),
            "gender": str(profile['GENDER']),
            "conditions": cond_display,
            "vitals": profile.get("VITALS", {})
        },
        "purchased_packages": [str(p) for p in purchased_history],
        "recommendations": recommendations
    }

if __name__ == "__main__":
    # เปิดเซิร์ฟเวอร์ทิ้งไว้ให้รันเบื้องหลัง (พอร์ต 8000)
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
