# agent/escalation.py
# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC escalation rules — no LLM involvement here.
# This is the most critical safety component. Hard-coded Python rules
# always override the LLM for crisis and clinical decisions.
# Bangladesh crisis resource included as specified in roadmap.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Crisis resource (from roadmap)
# ─────────────────────────────────────────────────────────────────────────────
CRISIS_HOTLINE_NAME   = "Kaan Pete Roi"
CRISIS_HOTLINE_NUMBER = "01779-554391"
CRISIS_HOTLINE_FULL   = f"{CRISIS_HOTLINE_NAME} — {CRISIS_HOTLINE_NUMBER}"


# ─────────────────────────────────────────────────────────────────────────────
# Escalation rule lambdas (exactly from roadmap)
# ─────────────────────────────────────────────────────────────────────────────
ESCALATION_RULES = {
    # Immediate: PHQ-9 Q9 (suicidal ideation) score >= 1 — NEVER LLM-mediated
    "immediate": lambda s: s.get("phq9_answers", {}).get(9, 0) >= 1,

    # Urgent: severe total scores
    "urgent": lambda s: (
        s.get("phq9_total", 0) >= 20 or
        s.get("gad7_total",  0) >= 15
    ),

    # Recommend: moderate scores
    "recommend": lambda s: (
        s.get("phq9_total", 0) >= 10 or
        s.get("gad7_total",  0) >= 10
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixed response messages per escalation level
# The LLM does NOT generate or modify these messages.
# ─────────────────────────────────────────────────────────────────────────────
ESCALATION_MESSAGES = {
    "immediate": (
        f"⚠️ I'm genuinely concerned about your safety right now.\n\n"
        f"Please reach out to a crisis support line immediately:\n"
        f"📞 {CRISIS_HOTLINE_FULL}\n\n"
        f"If you are in immediate danger, please go to your nearest hospital emergency department "
        f"or ask someone nearby for help. You are not alone."
    ),
    "urgent": (
        "Your responses suggest you are going through something very difficult right now.\n\n"
        "I strongly encourage you to speak with a mental health professional this week. "
        "Please consider contacting a counselor, psychologist, or psychiatrist as soon as possible. "
        f"If you need immediate support, you can also call {CRISIS_HOTLINE_FULL}."
    ),
    "recommend": (
        "It sounds like you have been struggling lately, and that matters.\n\n"
        "Based on your responses, it would be helpful to talk to a mental health professional. "
        "Consider reaching out to a counselor or therapist. "
        "We will check back in with you in two weeks."
    ),
    "low": (
        "Thank you for taking the time to check in with yourself today. "
        "Your responses suggest you are doing reasonably well. "
        "Continue to take care of yourself, and we will check in again next session."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main escalation router — returns level + message
# ─────────────────────────────────────────────────────────────────────────────
def apply_escalation(screening_state: dict) -> dict:
    """
    Evaluates deterministic rules in priority order.
    Returns a dict with:
      - level   : "immediate" | "urgent" | "recommend" | "low"
      - message : Fixed hard-coded message (not LLM-generated)
      - crisis  : bool — True if immediate crisis response needed
    """
    if ESCALATION_RULES["immediate"](screening_state):
        level = "immediate"
    elif ESCALATION_RULES["urgent"](screening_state):
        level = "urgent"
    elif ESCALATION_RULES["recommend"](screening_state):
        level = "recommend"
    else:
        level = "low"

    return {
        "level":   level,
        "message": ESCALATION_MESSAGES[level],
        "crisis":  level == "immediate",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: check only for immediate crisis (called mid-screening)
# PHQ-9 Q9 must be checked the moment it is answered — do not wait for
# screening to complete before routing to crisis response.
# ─────────────────────────────────────────────────────────────────────────────
def check_immediate_crisis(screening_state: dict) -> bool:
    """Returns True if PHQ-9 Q9 was answered with score >= 1."""
    return ESCALATION_RULES["immediate"](screening_state)
