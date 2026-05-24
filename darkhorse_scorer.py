import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. بارگذاری پروفایل رشته‌ها از فایل JSON
# ------------------------------------------------------------
def load_major_profiles(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['major_profiles']['majors']


# ------------------------------------------------------------
# 2. لایه اول: انگیزه‌های ریز (Micro-Motives)
# ------------------------------------------------------------
MICRO_MOTIVES_LEXICON = [
    "حل معماهای منطقی", "خودکارسازی کارهای تکراری", "کشف الگوهای پنهان در داده",
    "کمک به دیگران", "آموزش و یاددهی", "طراحی خلاقانه", "سازماندهی و نظم",
    "تحلیل سیستم‌های پیچیده", "کار عملی با دست", "برنامه‌ریزی استراتژیک"
]

def vectorize_micro_motives(selected_tags: List[str]) -> np.ndarray:
    clean_tags = [tag.strip() for tag in selected_tags]
    vec = np.zeros(len(MICRO_MOTIVES_LEXICON))
    for tag in clean_tags:
        if tag in MICRO_MOTIVES_LEXICON:
            idx = MICRO_MOTIVES_LEXICON.index(tag)
            vec[idx] = 1
    return vec.reshape(1, -1)

def compute_mm_score(user_tags: List[str], major_tags: List[str]) -> float:
    if not user_tags or not major_tags:
        return 0.0
    clean_major = [tag.strip() for tag in major_tags]
    major_vec = np.zeros(len(MICRO_MOTIVES_LEXICON))
    for tag in clean_major:
        if tag in MICRO_MOTIVES_LEXICON:
            idx = MICRO_MOTIVES_LEXICON.index(tag)
            major_vec[idx] = 1
    user_vec = vectorize_micro_motives(user_tags)
    if np.linalg.norm(user_vec) == 0 or np.linalg.norm(major_vec) == 0:
        return 0.0
    sim = cosine_similarity(user_vec, major_vec.reshape(1, -1))[0][0]
    return float(sim)


# ------------------------------------------------------------
# 3. لایه دوم: SJT (تست قضاوت موقعیتی)
# ------------------------------------------------------------
SJT_MAPPING = {
    "sjt_1": {"A": ["کمال‌گرایی", "ساختارگرایی"],
              "B": ["انسان‌گرایی", "مذاکره‌گری"],
              "C": ["ریسک‌پذیری", "ابهام‌پذیری"],
              "D": ["سلسله‌مراتبی", "قانون‌مداری"]},
    "sjt_2": {"A": ["فرصت‌طلبی"],
              "B": ["خودمختاری"],
              "C": ["کنجکاوی"],
              "D": ["اولویت درآمد"]},
    "sjt_3": {"A": ["صداقت", "کمال‌گرایی"],
              "B": ["اجتناب از تعارض"],
              "C": ["قانون‌مداری"],
              "D": ["خلاقیت", "ریسک‌پذیری"]},
    "sjt_4": {"A": ["اولویت درآمد", "ساختارگرایی"],
              "B": ["اولویت معنا", "خودمختاری"],
              "C": ["کنجکاوی", "ابهام‌پذیری"],
              "D": ["ریسک‌پذیری", "استقلال"]},
    "sjt_5": {"A": ["ساختارگرایی"],
              "B": ["ابهام‌پذیری", "کنجکاوی"],
              "C": ["انسان‌گرایی"],
              "D": ["سلسله‌مراتبی"]},
    "sjt_6": {"A": ["یادگیری‌محور"],
              "B": ["اجتناب از تعارض"],
              "C": ["خودمختاری", "ریسک‌پذیری"],
              "D": ["کمال‌گرایی", "قانون‌مداری"]},
    "sjt_7": {"A": ["اولویت شغلی"],
              "B": ["اولویت روابط"],
              "C": ["خلاقیت", "فشارپذیری"],
              "D": ["خودمختاری", "ریسک‌پذیری"]},
    "sjt_8": {"A": ["ساختارگرایی", "تحمل فشار"],
              "B": ["خودمختاری", "ریسک‌پذیری"],
              "C": ["فرصت‌طلبی", "انعطاف‌پذیری"],
              "D": ["انسان‌گرایی", "کنجکاوی"]}
}

def get_user_traits_from_sjt(answers: Dict[str, str]) -> set:
    traits = set()
    for scenario, choice in answers.items():
        if scenario in SJT_MAPPING and choice in SJT_MAPPING[scenario]:
            traits.update(SJT_MAPPING[scenario][choice])
    return traits

def compute_sjt_score(user_traits: set, major_env_profile: List[str]) -> float:
    if not user_traits or not major_env_profile:
        return 0.0
    set_user = set(user_traits)
    set_major = set([p.strip() for p in major_env_profile])
    inter = len(set_user & set_major)
    union = len(set_user | set_major)
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------
# 4. لایه سوم: Conjoint (ترجیحات ارزشی)
# ------------------------------------------------------------
VALUE_DIMS = ["income", "prestige", "meaning", "global_opportunity"]

def compute_user_value_weights_simple(conjoint_choices: Dict[str, str]) -> Dict[str, float]:
    CONJOINT_TO_VALUE = {
        "conj_1": {"A": ["meaning"], "B": ["income"]},
        "conj_2": {"A": ["prestige"], "B": ["global_opportunity"]},
        "conj_3": {"A": ["meaning"], "B": ["income"]},
        "conj_4": {"A": ["income"], "B": ["global_opportunity"]},
        "conj_5": {"A": ["income"], "B": ["meaning"]},
        "conj_6": {"A": ["income"], "B": ["meaning"]},
        "conj_7": {"A": ["income"], "B": ["meaning"]},
        "conj_8": {"A": ["global_opportunity"], "B": ["income"]},
        "conj_9": {"A": ["income"], "B": ["meaning"]},
        "conj_10": {"A": ["meaning"], "B": ["income"]}
    }
    counts = {dim: 0 for dim in VALUE_DIMS}
    for comp, choice in conjoint_choices.items():
        if comp in CONJOINT_TO_VALUE and choice in CONJOINT_TO_VALUE[comp]:
            for dim in CONJOINT_TO_VALUE[comp][choice]:
                if dim in counts:
                    counts[dim] += 1
    total = sum(counts.values())
    if total == 0:
        return {dim: 0.25 for dim in VALUE_DIMS}
    return {dim: counts[dim] / total for dim in VALUE_DIMS}

def compute_value_score(user_weights: Dict[str, float], major_values: Dict[str, int]) -> float:
    norm_major = {k: (v - 1) / 4.0 for k, v in major_values.items() if k in VALUE_DIMS}
    score = 0.0
    for dim in VALUE_DIMS:
        score += user_weights.get(dim, 0) * norm_major.get(dim, 0)
    return score


# ------------------------------------------------------------
# 5. جریمه‌ها (ساده برای MVP)
# ------------------------------------------------------------
def apply_penalties(user_reality: Optional[Dict[str, Any]], major_profile: Dict[str, Any]) -> float:
    if not user_reality:
        return 0.0
    penalty = 0.0
    # Burnout (فشار)
    if "pressure_tolerance" in user_reality:
        pressure_required = 3
        env = major_profile.get("environment_profile", [])
        if "فشار بالا" in env:
            pressure_required = 5
        elif "کم‌فشار" in env:
            pressure_required = 2
        gap = max(0, (pressure_required - user_reality["pressure_tolerance"]) / 4)
        penalty += min(0.12, 0.15 * gap)
    # PrestigeTrap
    if "prestige_sensitivity" in user_reality:
        prestige_level = major_profile.get("value_metrics", {}).get("prestige", 3)
        gap = max(0, (user_reality["prestige_sensitivity"] - prestige_level) / 4)
        penalty += min(0.03, 0.03 * gap)
    # FinancialStress
    if "financial_constraint" in user_reality:
        income_stability = major_profile.get("value_metrics", {}).get("income", 3)
        fin_gap = max(0, (5 - income_stability) * (user_reality["financial_constraint"] / 5))
        penalty += min(0.08, 0.10 * fin_gap)
    return min(penalty, 0.35)


# ------------------------------------------------------------
# 6. تابع اصلی محاسبه برای یک رشته
# ------------------------------------------------------------
def compute_darkhorse_fit_score_for_major(
    major_data: Dict[str, Any],
    user_micro_tags: List[str],
    user_sjt_answers: Dict[str, str],
    user_conjoint_choices: Dict[str, str],
    user_reality: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    mm_score = compute_mm_score(user_micro_tags, major_data.get("micro_motives", []))
    user_traits = get_user_traits_from_sjt(user_sjt_answers)
    sjt_score = compute_sjt_score(user_traits, major_data.get("environment_profile", []))
    user_weights = compute_user_value_weights_simple(user_conjoint_choices)
    value_score = compute_value_score(user_weights, major_data.get("value_metrics", {}))
    raw = 0.50 * mm_score + 0.30 * sjt_score + 0.20 * value_score
    penalty = apply_penalties(user_reality, major_data)
    final_score = max(0.0, raw - penalty) * 100
    return {
        "mm_score": mm_score,
        "sjt_score": sjt_score,
        "value_score": value_score,
        "raw_score": raw,
        "penalty": penalty,
        "darkhorse_fit_score": final_score,
        "non_linear_pathways": major_data.get("non_linear_pathways", [])
    }


# ------------------------------------------------------------
# 7. رتبه‌بندی همه رشته‌ها
# ------------------------------------------------------------
def rank_majors(
    major_profiles: Dict[str, Dict],
    user_micro_tags: List[str],
    user_sjt_answers: Dict[str, str],
    user_conjoint_choices: Dict[str, str],
    user_reality: Optional[Dict[str, Any]] = None,
    top_n: int = 20
) -> List[Tuple[str, float, Dict]]:
    results = []
    for major_id, major_data in major_profiles.items():
        details = compute_darkhorse_fit_score_for_major(
            major_data, user_micro_tags, user_sjt_answers, user_conjoint_choices, user_reality
        )
        results.append((major_id, details["darkhorse_fit_score"], details))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
