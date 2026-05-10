# agent/screener.py
# ─────────────────────────────────────────────────────────────────────────────
# PHQ-9 and GAD-7 questionnaire state management.
# Tracks which questions have been answered, allows partial/resumable sessions,
# and enforces the roadmap rule: PHQ-9 Q9 must be asked every session.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Question banks
# ─────────────────────────────────────────────────────────────────────────────
PHQ9_QUESTIONS = {
    1: "Over the last two weeks, how often have you had little interest or pleasure in doing things?",
    2: "How often have you felt down, depressed, or hopeless?",
    3: "How often have you had trouble falling or staying asleep, or sleeping too much?",
    4: "How often have you felt tired or had little energy?",
    5: "How often have you had poor appetite, or been overeating?",
    6: "How often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?",
    7: "How often have you had trouble concentrating on things, such as reading or watching television?",
    8: "How often have you been moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you moved around more than usual?",
    9: "How often have you had thoughts that you would be better off dead, or thoughts of hurting yourself in some way?"  # MUST be asked every session
}

GAD7_QUESTIONS = {
    1: "Over the last two weeks, how often have you felt nervous, anxious, or on edge?",
    2: "How often have you not been able to stop or control worrying?",
    3: "How often have you been worrying too much about different things?",
    4: "How often have you had trouble relaxing?",
    5: "How often have you been so restless that it is hard to sit still?",
    6: "How often have you become easily annoyed or irritable?",
    7: "How often have you felt afraid, as if something awful might happen?"
}

# Answer options shown to the user for each question
SCORE_OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

SCORE_OPTIONS_TEXT = "\n".join([f"  {k} = {v}" for k, v in SCORE_OPTIONS.items()])


# ─────────────────────────────────────────────────────────────────────────────
# State initializer
# ─────────────────────────────────────────────────────────────────────────────
def fresh_screening_state() -> dict:
    """Returns a clean screening state for a new or resumed session."""
    return {
        "phq9_answers":       {},       # {q_num (int): score 0-3}
        "gad7_answers":       {},
        "current_instrument": "phq9",   # "phq9" or "gad7"
        "questions_asked":    set(),    # {"phq9_1", "phq9_2", ...}
        "screening_complete": False,
        "phq9_total":         None,
        "gad7_total":         None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Next question selector
# ─────────────────────────────────────────────────────────────────────────────
def get_next_question(state: dict) -> tuple:
    """
    Returns (instrument, q_num, question_text) for the next unanswered question.
    Returns (None, None, None) when all questions are answered.

    Enforces: PHQ-9 Q9 is never skipped — it is always included in every session.
    """
    instrument = state["current_instrument"]
    asked      = state["questions_asked"]

    questions = PHQ9_QUESTIONS if instrument == "phq9" else GAD7_QUESTIONS

    for q_num, q_text in questions.items():
        key = f"{instrument}_{q_num}"
        if key not in asked:
            return (instrument, q_num, q_text)

    # Finished current instrument — switch or mark complete
    if instrument == "phq9":
        state["current_instrument"] = "gad7"
        return get_next_question(state)

    # All questions done
    state["screening_complete"] = True
    return (None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Answer recorder
# ─────────────────────────────────────────────────────────────────────────────
def record_answer(state: dict, instrument: str, q_num: int, score: int) -> dict:
    """
    Records a validated answer (0-3) for a question.
    Marks that question as asked so it won't be repeated.
    """
    if score not in SCORE_OPTIONS:
        raise ValueError(f"Score must be 0-3, got {score}")

    key = f"{instrument}_{q_num}"
    state["questions_asked"].add(key)

    if instrument == "phq9":
        state["phq9_answers"][q_num] = score
    else:
        state["gad7_answers"][q_num] = score

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Score computation with roadmap severity thresholds
# ─────────────────────────────────────────────────────────────────────────────
def compute_phq9_score(answers: dict) -> dict:
    """
    PHQ-9 severity thresholds (from roadmap):
    0-4: Minimal | 5-9: Mild | 10-14: Moderate | 15-19: Mod. Severe | 20-27: Severe
    """
    total = sum(answers.values()) if answers else 0

    if total <= 4:    severity = "Minimal"
    elif total <= 9:  severity = "Mild"
    elif total <= 14: severity = "Moderate"
    elif total <= 19: severity = "Moderately Severe"
    else:             severity = "Severe"

    return {"total": total, "severity": severity, "max": 27}


def compute_gad7_score(answers: dict) -> dict:
    """
    GAD-7 severity thresholds (from roadmap):
    0-4: Minimal | 5-9: Mild | 10-14: Moderate | 15-21: Severe
    """
    total = sum(answers.values()) if answers else 0

    if total <= 4:    severity = "Minimal"
    elif total <= 9:  severity = "Mild"
    elif total <= 14: severity = "Moderate"
    else:             severity = "Severe"

    return {"total": total, "severity": severity, "max": 21}


# ─────────────────────────────────────────────────────────────────────────────
# Fused risk score (roadmap: 0.4 clinical + 0.6 ML, tune on validation data)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fused_risk(phq9_total: int, gad7_total: int,
                       ml_dep: float, ml_anx: float) -> dict:
    """
    Weighted fusion of clinical scores and ML predictions.
    Weights from roadmap: start at 0.4/0.6, tune on your validation data.
    """
    depression_risk = 0.4 * (phq9_total / 27) + 0.6 * ml_dep
    anxiety_risk    = 0.4 * (gad7_total / 21) + 0.6 * ml_anx

    return {
        "depression_risk": round(depression_risk, 4),
        "anxiety_risk":    round(anxiety_risk, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parse user's numeric answer from free text
# ─────────────────────────────────────────────────────────────────────────────
def parse_score_from_text(text: str) -> int | None:
    """
    Tries to extract a 0-3 integer from the user's reply.
    Returns None if no valid score found (agent should re-prompt).
    """
    text = text.strip()

    # Direct digit
    if text in {"0", "1", "2", "3"}:
        return int(text)

    # Written words
    mapping = {
        "not at all": 0, "never": 0, "zero": 0,
        "several days": 1, "sometimes": 1, "one": 1,
        "more than half": 2, "often": 2, "two": 2, "most days": 2,
        "nearly every day": 3, "always": 3, "three": 3, "every day": 3
    }
    lower = text.lower()
    for phrase, score in mapping.items():
        if phrase in lower:
            return score

    return None  # Couldn't parse — agent will re-ask
