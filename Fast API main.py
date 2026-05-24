from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import json
from datetime import datetime
from darkhorse_scorer import load_major_profiles, rank_majors

# ------------------------------------------------------------
# مدل‌های داده
# ------------------------------------------------------------
class MicroMotivesInput(BaseModel):
    tags: List[str] = Field(..., description="لیست تگ‌های انگیزشی کاربر")

class SJTInput(BaseModel):
    answers: Dict[str, str] = Field(..., description="پاسخ به ۸ سناریو: sjt_1 تا sjt_8")

class ConjointInput(BaseModel):
    choices: Dict[str, str] = Field(..., description="انتخاب در ۱۰ مقایسه: conj_1 تا conj_10")

class RealityInput(BaseModel):
    pressure_tolerance: int = Field(3, ge=1, le=5)
    prestige_sensitivity: int = Field(3, ge=1, le=5)
    financial_constraint: int = Field(3, ge=1, le=5)

class DarkHorseRequest(BaseModel):
    micro_motives: MicroMotivesInput
    sjt: SJTInput
    conjoint: ConjointInput
    reality: Optional[RealityInput] = None

class RecommendationDetail(BaseModel):
    mm_score: float
    sjt_score: float
    value_score: float
    raw_score: float
    penalty: float
    darkhorse_fit_score: float
    non_linear_pathways: List[str]

class RecommendationOutput(BaseModel):
    major_id: str
    score: float
    details: RecommendationDetail

# مدل برای گزارش عدم تطابق
class MismatchReport(BaseModel):
    user_id: Optional[str] = None
    top_score: float
    mm_score: float
    sjt_score: float
    value_score: float
    top_majors: List[str]

# ------------------------------------------------------------
# راه‌اندازی اپ
# ------------------------------------------------------------
app = FastAPI(title="رشد هوشمند - موتور اسب سیاه", version="1.0")

# بارگذاری پروفایل رشته‌ها (مسیر فایل را در صورت نیاز اصلاح کنید)
try:
    MAJOR_PROFILES = load_major_profiles("major_profiles_80.json")
    print(f"✅ پروفایل رشته‌ها بارگذاری شد: {len(MAJOR_PROFILES)} رشته")
except Exception as e:
    print(f"❌ خطا در بارگذاری پروفایل: {e}")
    MAJOR_PROFILES = {}

# مسیر فایل برای ذخیره گزارش‌های عدم تطابق (در همان پوشه)
MISMATCH_FILE = "mismatch_reports.json"

def save_mismatch_report(report: dict):
    """ذخیره گزارش در فایل JSON (برای MVP)"""
    try:
        # خواندن گزارش‌های قبلی
        if os.path.exists(MISMATCH_FILE):
            with open(MISMATCH_FILE, 'r', encoding='utf-8') as f:
                reports = json.load(f)
        else:
            reports = []
        # اضافه کردن گزارش جدید
        reports.append(report)
        # ذخیره مجدد
        with open(MISMATCH_FILE, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"📝 گزارش جدید ذخیره شد. تعداد کل: {len(reports)}")
    except Exception as e:
        print(f"⚠️ خطا در ذخیره گزارش: {e}")

# ------------------------------------------------------------
# اندپوینت‌ها
# ------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "profiles_loaded": len(MAJOR_PROFILES)}

@app.post("/api/darkhorse/recommend", response_model=List[RecommendationOutput])
async def get_darkhorse_recommendations(request: DarkHorseRequest):
    # اعتبارسنجی تعداد پاسخ‌ها
    if len(request.sjt.answers) != 8:
        raise HTTPException(400, "پاسخ به تمام ۸ سناریوی SJT الزامی است")
    if len(request.conjoint.choices) != 10:
        raise HTTPException(400, "پاسخ به تمام ۱۰ مقایسه conjoint الزامی است")
    if not MAJOR_PROFILES:
        raise HTTPException(500, "پروفایل رشته‌ها بارگذاری نشده است")
    
    user_reality = None
    if request.reality:
        user_reality = request.reality.dict()
    
    ranked = rank_majors(
        MAJOR_PROFILES,
        request.micro_motives.tags,
        request.sjt.answers,
        request.conjoint.choices,
        user_reality,
        top_n=20
    )
    
    results = []
    for major_id, score, details in ranked:
        results.append(RecommendationOutput(
            major_id=major_id,
            score=score,
            details=RecommendationDetail(
                mm_score=details["mm_score"],
                sjt_score=details["sjt_score"],
                value_score=details["value_score"],
                raw_score=details["raw_score"],
                penalty=details["penalty"],
                darkhorse_fit_score=details["darkhorse_fit_score"],
                non_linear_pathways=details["non_linear_pathways"]
            )
        ))
    return results

@app.post("/api/feedback/mismatch")
async def report_mismatch(report: MismatchReport):
    """دریافت گزارش عدم تطابق از فرانت‌اند"""
    report_dict = report.dict()
    report_dict["created_at"] = datetime.now().isoformat()
    save_mismatch_report(report_dict)
    return {"status": "ok", "message": "گزارش با موفقیت ثبت شد"}
