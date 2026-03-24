import streamlit as st
import requests
import os

st.set_page_config(page_title="Navavej Upsell App", layout="wide", page_icon="🏥")

# ชี้เป้าไปที่ API Server ก้อนสมองของเรา
API_URL_BASE = os.getenv("NAVAVEJ_API_URL", "http://localhost:8000/api/v1")

st.title("🏥 Navavej AI Recommendation (UI Client)")
st.caption("หน้าจอนี้ไม่มีสมองคำนวณเบื้องหลังเลย! วิ่งไปถาม API ข้าม Network ล้วนๆ 🚀")
st.markdown("---")

# ==========================================
# 1. ยิง API ขอรายชื่อคนไข้มาแสดงใน Dropdown
# ==========================================
@st.cache_data(ttl=300)
def fetch_patients():
    try:
        resp = requests.get(f"{API_URL_BASE}/patients", timeout=3)
        if resp.status_code == 200:
            return resp.json()["patients"]
    except requests.RequestException:
        return []
        
patient_list = fetch_patients()

st.sidebar.header("🔍 ระบบค้นหาลูกค้า")
if patient_list:
    selected_pt = st.sidebar.selectbox(
        "เลือกลูกค้าอ้างอิงจากคิว (200 คนล่าสุด)", 
        patient_list,
        format_func=lambda pt: pt['label']
    )
    patient_id = selected_pt['id']
else:
    patient_id = st.sidebar.text_input("กรอก Patient ID", value="xxxx-xxxx")
    st.sidebar.warning("⚠️ เชื่อมต่อเครือข่าย API ไม่ได้ (รัน uvicorn หรือยัง?) ให้กรอกรหัสแมนนวล")

# ==========================================
# 2. ปล่อยพลัง! ยิง API ขอคำทำนายจากเซิร์ฟเวอร์หลัก
# ==========================================
if st.sidebar.button("🤖 ดึงแพ็กเกจแนะนำจากเซิร์ฟเวอร์", type="primary"):
    with st.spinner('กำลังยิง HTTP GET Request ไปยังเซิร์ฟเวอร์ชั้นบน...'):
        try:
            response = requests.get(f"{API_URL_BASE}/recommend?patient_id={patient_id}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # --- วาด UI ฝั่งข้อมูลลูกค้า ---
                patient = data['patient']
                st.sidebar.markdown("### 📋 ข้อมูลส่วนตัวเบื้องต้น")
                st.sidebar.markdown(f"**อายุ:** {patient['age']} ปี | **เพศ:** {'ชาย' if patient['gender'] == 'M' else 'หญิง'}")
                st.sidebar.markdown(f"**โรคประจำตัว:** {patient['conditions']}")
                
                vitals = patient.get('vitals', {})
                if vitals:
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 🩺 ค่าผลตรวจทางคลินิก (Vitals)")
                    vc1, vc2 = st.sidebar.columns(2)
                    vc1.metric("ดัชนีมวลกาย (BMI)", vitals.get('BMI', 'N/A'))
                    vc2.metric("น้ำตาล (Glucose)", vitals.get('Glucose', 'N/A'))
                    
                    vc3, vc4 = st.sidebar.columns(2)
                    bp_sys = vitals.get('SysBP', 'N/A')
                    bp_dia = vitals.get('DiaBP', 'N/A')
                    bp_text = f"{bp_sys}/{bp_dia}" if bp_sys != 'N/A' else 'N/A'
                    vc3.metric("ความดันโลหิต (BP)", bp_text)
                    vc4.metric("คอเลสเตอรอล", vitals.get('Cholesterol', 'N/A'))
                
                col1, col2 = st.columns([1, 1.2])

                # --- วาด UI ฝั่งตะกร้าสินค้า (Cart) ---
                with col1:
                    st.subheader("🛒 แพ็กเกจตั้งต้น (Cart)")
                    st.info("ของที่ลูกค้าคนนี้มาซื้อไปแล้ว / สนใจจะซื้อ:")
                    for pkg in data['purchased_packages']:
                        st.write(f"✅ **{pkg}**")

                # --- วาด UI โชว์เด็ด AI ทำนายผล ---
                with col2:
                    st.subheader("✨ AI Upsell Suggestions (From API Server)")
                    recs = data['recommendations']
                    
                    if not recs:
                        st.warning("⚠️ ลูกค้ารายนี้ไม่เข้าเกณฑ์ให้คุณหมอเสนอแพ็กเกจเพิ่มครับ")
                        
                    for idx, rec in enumerate(recs, 1):
                        st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; border:2px solid #5C6BC0; margin-bottom:15px; background-color:#E8EAF6;">
                            <strong style="color:#283593; font-size:18px;">💡 โอกาสเชียร์ขายอันดับ {idx}: {rec['package_name']}</strong><br/>
                            <span style="color:#3F51B5;">รหัส: {rec['package_id']} | AI มั่นใจ: {(rec['confidence_score'] * 100):.1f}% </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
            elif response.status_code == 404:
                st.error("❌ ไม่พบประวัติผู้ป่วยนี้ในเซิร์ฟเวอร์ครับ")
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ ไม่สามารถเชื่อมต่อกับ API ได้!\n\nโปรดเปิด Terminal แล้วรัน `uvicorn main:app --reload` หรือ `python main.py` ไว้ก่อนครับ")
