# Navavej Demo

Navavej Demo คือโปรเจคตัวอย่างสำหรับแนะนำแพ็กเกจสุขภาพเพิ่มเติมจากประวัติผู้ป่วย โดยใช้แนวคิด recommendation แบบผสมระหว่าง:

- item-based collaborative filtering
- deterministic rule-based transaction generation
- business / medical guardrails

โปรเจคนี้ถูกออกแบบมาเพื่อใช้เดโมแนวคิดของระบบ ไม่ใช่ production-ready clinical recommendation system โดยตรง

## What This Project Does

Clinical-rule note:

- The training snapshot now uses deterministic clinical-style package assignment instead of probabilistic package sampling.
- Missing vitals in the snapshot fallback path are filled with deterministic midpoint values from condition-aware ranges.

ระบบนี้มี flow หลักดังนี้:

1. ใช้ข้อมูลผู้ป่วยจากไฟล์ encounter ใน `data/`
2. สร้าง transaction จำลองของการซื้อแพ็กเกจด้วย rule-based logic
3. คำนวณ similarity ระหว่างแพ็กเกจและบันทึกเป็น model artifact `.pkl`
4. เปิด API เพื่อให้ UI ขอข้อมูลผู้ป่วยและ recommendations
5. แสดงผลผ่านหน้า Streamlit

ผลลัพธ์ที่หน้าเดโมจะแสดงคือ:

- ข้อมูลผู้ป่วยเบื้องต้น
- รายการแพ็กเกจที่มีอยู่ในประวัติ
- แพ็กเกจแนะนำ 3 อันดับแรก

## Project Structure

ไฟล์และโฟลเดอร์สำคัญ:

- `main.py`
  entry point สำหรับเปิด API
- `src/api.py`
  FastAPI app และ endpoints หลัก
- `src/inference_engine.py`
  โหลด model artifact และคำนวณ recommendations
- `src/train_job.py`
  ใช้สร้าง transaction จำลองและ train model
- `strict_evaluation.py`
  ใช้ประเมิน recommendation logic แบบ offline
- `ui/app.py`
  Streamlit UI
- `src/config.py`
  รวม path configuration ของระบบ
- `data/`
  ข้อมูล input สำหรับเดโม
- `models/`
  เก็บไฟล์ `model_artifacts.pkl`
- `docs/`
  เอกสารอธิบายเชิงเทคนิค

## Architecture Summary

ภาพรวมการไหลของข้อมูล:

```text
data/merged_encounters.csv
        |
        v
src/train_job.py
        |
        v
models/model_artifacts.pkl
        |
        v
src/inference_engine.py
        |
        v
src/api.py
        |
        v
ui/app.py
```

สรุปสั้นๆ:

- `train_job.py` สร้าง `.pkl`
- `main.py` และ `src/api.py` โหลด `.pkl`
- `ui/app.py` เป็น client ที่เรียก API

## Project Architecture

สถาปัตยกรรมของโปรเจคนี้แบ่งได้เป็น 4 ชั้นหลัก:

### 1. Data Layer

ชั้นนี้รับผิดชอบข้อมูลต้นทางและไฟล์ที่ระบบใช้เป็น input/output หลัก

องค์ประกอบสำคัญ:

- `data/merged_encounters.csv`
  ข้อมูล encounter และ clinical fields ที่ใช้เป็นฐานของการสร้าง transaction
- `models/model_artifacts.pkl`
  artifact ที่ผ่านการ train แล้วและถูกใช้ตอน inference

หน้าที่ของชั้นนี้:

- เก็บข้อมูลต้นทาง
- เป็นแหล่งข้อมูลสำหรับ training
- เป็นจุดเก็บ model ที่ API ต้องโหลดตอนเริ่มทำงาน

### 2. Training Layer

ชั้นนี้รับผิดชอบการแปลงข้อมูลต้นทางให้เป็น recommendation model

องค์ประกอบสำคัญ:

- `src/train_job.py`

หน้าที่ของชั้นนี้:

- อ่าน encounter data
- สร้าง transaction จำลองจาก heuristic rules
- ดึง/เติมข้อมูล vitals ที่ใช้ในเดโม
- สร้าง patient-package matrix
- คำนวณ item similarity
- บันทึกผลลัพธ์เป็น `.pkl`

ผลลัพธ์ของชั้นนี้คือ:

- `models/model_artifacts.pkl`

### 3. Inference and API Layer

ชั้นนี้รับผิดชอบโหลด model, ประมวลผลคำแนะนำ, และเปิดบริการผ่าน HTTP

องค์ประกอบสำคัญ:

- `main.py`
- `src/api.py`
- `src/inference_engine.py`
- `src/config.py`

หน้าที่ของแต่ละไฟล์:

- `main.py`
  ใช้เป็น entry point สำหรับเปิด API server
- `src/api.py`
  นิยาม endpoints และจัดการ request/response
- `src/inference_engine.py`
  โหลด artifact, ดึง profile ผู้ป่วย, และคำนวณ recommendations
- `src/config.py`
  จัดการ default paths และ environment-based configuration

หน้าที่ของชั้นนี้:

- โหลด model เข้า memory
- รับ request จาก UI
- ดึงข้อมูลผู้ป่วยจาก transaction dataframe
- คำนวณ score ของแพ็กเกจ
- ใช้ medical/business rules กรอง recommendation
- ส่งผลลัพธ์กลับเป็น JSON

### 4. Presentation Layer

ชั้นนี้รับผิดชอบการแสดงผลให้ผู้ใช้เห็น

องค์ประกอบสำคัญ:

- `ui/app.py`

หน้าที่ของชั้นนี้:

- เรียก API เพื่อดึงรายชื่อผู้ป่วย
- เรียก API เพื่อขอ recommendation
- แสดงข้อมูลผู้ป่วย, purchased packages, และ recommendation cards

จุดสำคัญ:

- UI ไม่มี model อยู่ในตัวเอง
- UI ไม่คำนวณ recommendation เอง
- UI ทำหน้าที่เป็น thin client ที่พึ่งพา API

### Component Relationship

ความสัมพันธ์ของแต่ละส่วนสามารถสรุปได้แบบนี้:

```text
+-------------------------+
|      Data Layer         |
| merged_encounters.csv   |
| model_artifacts.pkl     |
+-----------+-------------+
            |
            v
+-------------------------+
|    Training Layer       |
|    src/train_job.py     |
+-----------+-------------+
            |
            v
+-------------------------+
| Inference / API Layer   |
| main.py                 |
| src/api.py              |
| src/inference_engine.py |
| src/config.py           |
+-----------+-------------+
            |
            v
+-------------------------+
|  Presentation Layer     |
|      ui/app.py          |
+-------------------------+
```

### Request Lifecycle

ตัวอย่าง flow ตอนผู้ใช้ขอ recommendation:

1. ผู้ใช้เปิด `ui/app.py` ผ่าน Streamlit
2. UI เรียก `GET /api/v1/patients` เพื่อโหลดรายการผู้ป่วย
3. ผู้ใช้เลือก patient และกดปุ่มขอ recommendation
4. UI เรียก `GET /api/v1/recommend?patient_id=...`
5. `src/api.py` ส่ง request ต่อไปยัง `UpsellRecommenderEngine`
6. engine ดึง profile และประวัติ package ของผู้ป่วย
7. engine คำนวณ candidate scores จาก similarity matrix
8. engine ใช้ rule-based filtering ตัดแพ็กเกจที่ไม่เหมาะสม
9. API สร้าง response JSON
10. UI นำ response ไปแสดงผลบนหน้าเดโม

### Separation of Responsibilities

แนวคิดการแยกความรับผิดชอบในโปรเจคนี้คือ:

- training logic อยู่ใน `src/train_job.py`
- inference logic อยู่ใน `src/inference_engine.py`
- transport layer อยู่ใน `src/api.py`
- presentation logic อยู่ใน `ui/app.py`
- configuration อยู่ใน `src/config.py`

ข้อดีของการแยกแบบนี้:

- อ่านโค้ดง่ายขึ้น
- แก้ส่วนใดส่วนหนึ่งได้โดยกระทบอีกส่วนน้อยลง
- ช่วยให้ขยายต่อเป็น production architecture ได้ง่ายกว่าเขียนทุกอย่างรวมในไฟล์เดียว

## Requirements

- Python `3.12+`
- แนะนำให้ใช้ `uv`

dependency หลักของโปรเจค:

- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `requests`

## Setup

ติดตั้ง dependency:

```powershell
uv sync
```

ถ้าคุณไม่ได้ใช้ `uv` สามารถติดตั้งด้วย `pip` ได้เช่นกัน:

```powershell
pip install -e .
```

## Quick Start

### 1. เปิด API

```powershell
uv run uvicorn main:app --reload
```

หรือ:

```powershell
uv run python main.py
```

ค่า default:

- API URL: `http://localhost:8000`
- health check: `http://localhost:8000/health`

### 2. เปิด UI

```powershell
uv run streamlit run ui/app.py
```

จากนั้นเปิด:

- `http://localhost:8501`

### 3. ทดลองใช้งาน

เมื่อ API เปิดอยู่แล้ว:

1. เปิด UI
2. เลือกผู้ป่วยจาก sidebar
3. กดปุ่มดึงแพ็กเกจแนะนำ
4. ดูข้อมูลผู้ป่วยและ recommendation cards

## How Model Loading Works

จุดที่สำคัญมาก:

- `uv run uvicorn main:app --reload` ไม่ได้สร้าง `.pkl`
- `uv run python main.py` ก็ไม่ได้สร้าง `.pkl` เช่นกัน
- ทั้งสองคำสั่งมีหน้าที่เปิด API และโหลด `.pkl` ที่มีอยู่แล้ว

ดังนั้น:

- ถ้ามี `models/model_artifacts.pkl` อยู่แล้ว ระบบจะโหลดและพร้อมให้บริการ
- ถ้าไม่มีไฟล์นี้ API จะเปิดได้แต่ endpoint หลักจะตอบกลับแบบ degraded / ใช้งานจริงไม่ได้

## How To Retrain The Model

Additional note:

- `src.train_job` builds from a snapshot that is generated with deterministic rules.
- If you want to rebuild the snapshot as well, set `NAVAVEJ_REBUILD_SNAPSHOT=true` before running the training command.

ถ้าต้องการสร้างหรืออัปเดต model artifact ใหม่:

```powershell
uv run python -m src.train_job
```

สิ่งที่ script นี้ทำ:

1. อ่านข้อมูล encounter
2. สร้าง transaction จำลองตาม rules
3. คำนวณ package similarity
4. บันทึกผลเป็น `models/model_artifacts.pkl`

หลังจาก train เสร็จแล้ว ให้เปิด API ใหม่อีกครั้งเพื่อโหลด model เวอร์ชันล่าสุด

## Data and Model Files

ไฟล์หลักที่ระบบใช้:

- `data/merged_encounters.csv`
  ข้อมูล encounter ต้นทาง
- `models/model_artifacts.pkl`
  model artifact ที่ inference ใช้งาน

หมายเหตุ:

- ถ้าไม่มี package catalog ภายนอก ระบบจะ fallback ไปใช้ `PACKAGE code` เป็นชื่อชั่วคราว
- โปรเจคนี้มี model artifact อยู่แล้วใน repo เพื่อให้เดโมเริ่มต้นได้ง่ายขึ้น

## Environment Variables

Additional environment variable:

- `NAVAVEJ_SNAPSHOT_PATH`
  Custom path for the generated training snapshot `.pkl`.

ระบบรองรับ environment variables ต่อไปนี้:

- `NAVAVEJ_MODEL_PATH`
  กำหนด path ของไฟล์ `model_artifacts.pkl`
- `NAVAVEJ_ENCOUNTERS_PATH`
  กำหนด path ของไฟล์ encounter data
- `NAVAVEJ_PACKAGES_PATH`
  กำหนด path ของ package catalog
- `NAVAVEJ_API_URL`
  กำหนด base URL ของ API สำหรับ UI

ถ้าไม่กำหนด ระบบจะใช้ path default ภายใน repo นี้

ตัวอย่าง:

```powershell
$env:NAVAVEJ_API_URL="http://localhost:8000/api/v1"
uv run streamlit run ui/app.py
```

## API Endpoints

endpoint หลักที่มีตอนนี้:

- `GET /health`
  เช็กว่า API ทำงานอยู่และ model ถูกโหลดหรือไม่
- `GET /api/v1/patients`
  ดึงรายการผู้ป่วยสำหรับ dropdown ใน UI
- `GET /api/v1/recommend?patient_id=...`
  ดึงคำแนะนำแพ็กเกจสำหรับผู้ป่วยที่ระบุ

## Evaluation

มี script สำหรับประเมิน recommendation logic แบบ offline:

```powershell
uv run python strict_evaluation.py
```

สิ่งที่ script นี้วัดคร่าวๆ:

- hit rate ของการทำนาย package ที่ซ่อนไว้
- จำนวน recommendation ที่ผิดกฎทางการแพทย์ตาม rule ที่กำหนด

ข้อควรเข้าใจ:

- evaluation นี้เป็น sanity check สำหรับเดโม
- ยังไม่ใช่ production-grade model validation

## Demo Assumptions and Limitations

Current rule behavior:

- Synthetic transactions are still generated, but now from deterministic clinical-style rules rather than random probabilities.
- Missing vitals may still be synthesized when source data is absent, but the fallback is now a deterministic midpoint instead of a random value.
- Medical and business rules remain hard-coded in source and should still be reviewed with domain experts before production use.

จุดที่ควรทราบก่อนใช้เดโมหรืออธิบายต่อ stakeholder:

- transaction หลายส่วนถูกสร้างจาก heuristic rules ไม่ใช่ purchase history จริงทั้งหมด
- recommendation จึงสะท้อนสมมติฐานของเดโมร่วมด้วย
- vitals บางตัวอาจถูก mock เมื่อข้อมูลต้นทางไม่มีค่า
- medical/business rules ยัง hard-code อยู่ใน source code
- ระบบยังไม่มี automated test suite แบบครบถ้วน
- environment ที่มีปัญหา native dependency อาจทำให้ `numpy`, `pandas`, `fastapi` หรือ `pydantic` import ไม่ได้แม้โค้ดจะถูกต้อง

## Troubleshooting

### เปิด API แล้วใช้งานไม่ได้

เช็ก:

- มีไฟล์ `models/model_artifacts.pkl` หรือยัง
- environment import `numpy`, `pandas`, `fastapi` ได้หรือไม่
- path ที่ตั้งผ่าน env vars ถูกต้องหรือไม่

### UI เชื่อม API ไม่ได้

เช็ก:

- API รันอยู่หรือไม่
- `NAVAVEJ_API_URL` ถูกต้องหรือไม่
- health endpoint ตอบหรือไม่:

```powershell
curl http://localhost:8000/health
```

### ไม่มี `.pkl`

ให้รัน:

```powershell
uv run python -m src.train_job
```

แล้วค่อยเปิด API ใหม่

## Recommended Reading Order

ถ้าคุณเพิ่งเข้ามาดูโปรเจคนี้ แนะนำลำดับการอ่าน:

1. `README.md`
2. `main.py`
3. `src/api.py`
4. `src/inference_engine.py`
5. `src/train_job.py`
6. `ui/app.py`
7. `strict_evaluation.py`

ลำดับนี้จะช่วยให้เห็นภาพการทำงานของ runtime ก่อน แล้วค่อยลงรายละเอียดของ training และ evaluation

## Technical Documentation

เอกสารเชิงเทคนิคแบบละเอียดอยู่ที่:

- `docs/TECHNICAL_OVERVIEW.md`

ไฟล์นี้อธิบายเพิ่มเรื่อง:

- architecture
- runtime flow
- training flow
- inference logic
- rule-based filtering
- evaluation limitations

## Current Status

สถานะปัจจุบันของโปรเจคนี้เหมาะสำหรับ:

- เดโมแนวคิดระบบ recommendation
- เดโม flow API + UI
- ใช้เป็น prototype สำหรับคุย requirement หรือพัฒนาต่อ

ยังไม่ควรมองเป็น:

- production-ready recommendation platform
- clinical decision support system ที่พร้อมใช้งานจริงโดยไม่มีการ validate เพิ่ม
