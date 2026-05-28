"""
Dark Horse API - Backend اصلی پروژه رشد پیلکینک
نسخه: 1.0.0
Deploy: Hugging Face Spaces
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from darkhorse_scorer_v4 import (
    load_major_profiles,
    rank_majors,
    compute_darkhorse_fit_score_for_major
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('darkhorse-api')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="Dark Horse API",
    description="سامانه هوشمند انتخاب رشته - رشد پیلکینک",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load Data at Startup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAJOR_PROFILES_PATH = Path("major_profiles_89_v3.json")
MISMATCH_REPORTS_PATH = Path("mismatch_reports.json")

try:
    MAJOR_PROFILES = load_major_profiles(str(MAJOR_PROFILES_PATH))
    logger.info(f"✅ {len(MAJOR_PROFILES)} رشته بارگذاری شد")
except Exception as e:
    logger.error(f"❌ خطا در بارگذاری پروفایل‌ها: {e}")
    MAJOR_PROFILES = {}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserReality(BaseModel):
    pressure_tolerance: int = Field(3, ge=1, le=5)
    prestige_sensitivity: int = Field(3, ge=1, le=5)
    financial_constraint: int = Field(3, ge=1, le=5)


class DarkHorseRequest(BaseModel):
    micro_motives: List[str] = Field(..., min_length=1, max_length=3)
    sjt_answers: Dict[str, str]
    conjoint_choices: Dict[str, str]
    reality: Optional[UserReality] = None
    user_id: Optional[str] = None
    top_n: int = Field(20, ge=5, le=50)


class StandardRequest(BaseModel):
    exam_group: str
    rank_in_quota: int
    quota_type: str
    gpa_12: Optional[float] = None
    province: Optional[str] = None


class AcademicRequest(BaseModel):
    diploma_type: str
    gpa_written_final: float
    province: Optional[str] = None


class MismatchFeedback(BaseModel):
    top_score: float
    top_majors: List[str]
    mm_score: float    sjt_score: float
    value_score: float
    user_id: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
async def root():
    return {
        "message": "🐴 Dark Horse API - رشد پیلکینک",
        "version": "1.0.0",
        "majors_loaded": len(MAJOR_PROFILES),
        "endpoints": ["/api/health", "/api/darkhorse/recommend", 
                     "/api/standard/recommend", "/api/academic/recommend",
                     "/api/feedback/mismatch", "/docs"]
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "majors_loaded": len(MAJOR_PROFILES),
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/darkhorse/recommend")
async def darkhorse_recommend(request: DarkHorseRequest):
    """محاسبه ۲۰ رشته برتر با الگوریتم DDHFS"""
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"🔵 DarkHorse request: session={session_id}")

        reality_dict = request.reality.dict() if request.reality else None

        ranked = rank_majors(
            major_profiles=MAJOR_PROFILES,
            user_micro_motives=request.micro_motives,
            user_sjt_answers=request.sjt_answers,
            user_conjoint_choices=request.conjoint_choices,
            user_reality=reality_dict,
            top_n=request.top_n
        )

        if not ranked:            raise HTTPException(status_code=500, detail="خطا در محاسبه نمرات")

        top_major = ranked[0]
        user_profile = {
            "cms": top_major["cms"],
            "regime": top_major["regime"],
            "value_weights": top_major.get("user_value_weights", {}),
            "traits": top_major.get("user_traits", [])
        }

        avg_score = sum(r["darkhorse_fit_score"] for r in ranked) / len(ranked)

        logger.info(f"✅ Success: top={ranked[0]['major_name_fa']} ({ranked[0]['darkhorse_fit_score']})")

        return {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_profile": user_profile,
            "ranked_majors": ranked,
            "summary": {
                "total_analyzed": len(MAJOR_PROFILES),
                "returned": len(ranked),
                "average_score": round(avg_score, 2),
                "highest_score": ranked[0]["darkhorse_fit_score"],
                "lowest_score": ranked[-1]["darkhorse_fit_score"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DarkHorse error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/standard/recommend")
async def standard_recommend(request: StandardRequest):
    """تب با آزمون (MVP - mock data)"""
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"🔵 Standard: rank={request.rank_in_quota}")

        def calc_prob(user_rank, last_rank):
            return max(0, min(100, 100 - (user_rank / max(last_rank, 1)) * 100))

        mock_data = [
            {"university": "دانشگاه تهران", "major": "مهندسی کامپیوتر", "last_rank": 800, "type": "روزانه"},
            {"university": "دانشگاه شریف", "major": "مهندسی کامپیوتر", "last_rank": 400, "type": "روزانه"},
            {"university": "دانشگاه امیرکبیر", "major": "مهندسی کامپیوتر", "last_rank": 700, "type": "روزانه"},
            {"university": "دانشگاه آزاد تهران", "major": "مهندسی کامپیوتر", "last_rank": 8000, "type": "آزاد"},            {"university": "دانشگاه اصفهان", "major": "روانشناسی", "last_rank": 5000, "type": "روزانه"},
        ]

        results = []
        for uni in mock_data:
            prob = calc_prob(request.rank_in_quota, uni["last_rank"])
            color = "سبز" if prob >= 75 else ("نارنجی" if prob >= 45 else "قرمز")
            results.append({
                "university": uni["university"],
                "major": uni["major"],
                "type": uni["type"],
                "probability": round(prob, 2),
                "color": color
            })

        results.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "session_id": session_id,
            "input": request.dict(),
            "results": results,
            "note": "MVP - در فاز ۲ با XGBoost جایگزین می‌شود"
        }

    except Exception as e:
        logger.error(f"❌ Standard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/academic/recommend")
async def academic_recommend(request: AcademicRequest):
    """تب سوابق تحصیلی (MVP - mock data)"""
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"🔵 Academic: gpa={request.gpa_written_final}")

        def calc_prob(user_gpa, cutoff):
            return max(0, min(100, (user_gpa / max(cutoff, 1)) * 100))

        mock_data = [
            {"university": "پیام نور تهران", "major": "مدیریت", "cutoff": 14.5, "type": "پیام نور"},
            {"university": "آزاد تهران جنوب", "major": "حسابداری", "cutoff": 13.0, "type": "آزاد"},
            {"university": "غیرانتفاعی سوره", "major": "گرافیک", "cutoff": 15.0, "type": "غیرانتفاعی"},
        ]

        results = []
        for uni in mock_data:
            prob = calc_prob(request.gpa_written_final, uni["cutoff"])
            color = "سبز" if prob >= 75 else ("نارنجی" if prob >= 45 else "قرمز")
            results.append({                "university": uni["university"],
                "major": uni["major"],
                "type": uni["type"],
                "probability": round(prob, 2),
                "color": color
            })

        results.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "session_id": session_id,
            "input": request.dict(),
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ Academic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/mismatch")
async def feedback_mismatch(feedback: MismatchFeedback):
    """ذخیره گزارش عدم تطابق"""
    try:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_id": feedback.user_id,
            "top_score": feedback.top_score,
            "top_majors": feedback.top_majors,
            "scores": {
                "mm": feedback.mm_score,
                "sjt": feedback.sjt_score,
                "value": feedback.value_score
            }
        }

        existing = []
        if MISMATCH_REPORTS_PATH.exists():
            try:
                with open(MISMATCH_REPORTS_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except:
                existing = []

        existing.append(record)

        with open(MISMATCH_REPORTS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Feedback saved: top_score={feedback.top_score}")

        return {"status": "saved", "id": record["id"]}

    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)