import uvicorn
from src.api import app

# ==========================================
# ไฟล์ระบุจุดเริ่มต้นรับส่ง HTTP API (Entry Point)
# ให้รันไฟล์นี้ หรือใช้ Uvicorn สั่งรันได้เลยเพื่อสตาร์ตแอป
# ==========================================

if __name__ == "__main__":
    # รองรับการรันผ่าน python main.py ธรรมดา
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
