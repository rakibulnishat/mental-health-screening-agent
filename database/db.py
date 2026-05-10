# database/db.py
# ─────────────────────────────────────────────────────────────────────────────
# SQLite + SQLModel session storage.
# Stores only structured scores and dates — NEVER raw conversation text.
# Supports longitudinal tracking and trend detection across sessions.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import date
from typing import Optional, List

from sqlmodel import SQLModel, Field, create_engine, Session, select

# ─────────────────────────────────────────────────────────────────────────────
# Schema (from roadmap)
# ─────────────────────────────────────────────────────────────────────────────
class SessionRecord(SQLModel, table=True):
    id:               Optional[int] = Field(default=None, primary_key=True)
    user_id:          str
    session_date:     date
    phq9_score:       int
    gad7_score:       int
    depression_risk:  float
    anxiety_risk:     float
    escalation_level: str


# ─────────────────────────────────────────────────────────────────────────────
# Engine — SQLite file in project root
# ─────────────────────────────────────────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine("sqlite:///mental_health.db", echo=False)
        SQLModel.metadata.create_all(_engine)
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Save a completed session
# ─────────────────────────────────────────────────────────────────────────────
def save_session(
    user_id:          str,
    phq9_score:       int,
    gad7_score:       int,
    depression_risk:  float,
    anxiety_risk:     float,
    escalation_level: str,
) -> SessionRecord:
    """
    Persists one session's scores.
    Called at the end of every completed screening.
    Raw conversation text is NEVER stored (roadmap privacy rule).
    """
    record = SessionRecord(
        user_id          = user_id,
        session_date     = date.today(),
        phq9_score       = phq9_score,
        gad7_score       = gad7_score,
        depression_risk  = round(depression_risk, 4),
        anxiety_risk     = round(anxiety_risk, 4),
        escalation_level = escalation_level,
    )
    with Session(get_engine()) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Load user history (oldest first)
# ─────────────────────────────────────────────────────────────────────────────
def get_user_history(user_id: str) -> List[SessionRecord]:
    """Returns all past sessions for a user, ordered by date ascending."""
    with Session(get_engine()) as session:
        records = session.exec(
            select(SessionRecord)
            .where(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.session_date)
        ).all()
    return list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Trend detection (roadmap: alert when scores worsen over 2+ consecutive sessions)
# ─────────────────────────────────────────────────────────────────────────────
def detect_worsening(records: List[SessionRecord]) -> bool:
    """
    Returns True if PHQ-9 score increased across the last two sessions.
    Triggers a warning in the agent's opening message.
    """
    if len(records) < 2:
        return False
    return records[-1].phq9_score > records[-2].phq9_score


def build_trend_message(records: List[SessionRecord]) -> str:
    """
    Returns a natural-language trend summary to surface in conversation.
    Examples from roadmap: 'Your PHQ-9 dropped from 14 to 9 — real progress.'
    """
    if len(records) < 2:
        return ""

    prev = records[-2]
    curr = records[-1]
    diff = curr.phq9_score - prev.phq9_score

    if diff < 0:
        return (
            f"Your PHQ-9 score dropped from {prev.phq9_score} to {curr.phq9_score} "
            f"since your last session — that is real progress. Keep going."
        )
    elif diff > 0:
        return (
            f"Your PHQ-9 score increased from {prev.phq9_score} to {curr.phq9_score} "
            f"since your last session. Let us talk about how you have been feeling."
        )
    else:
        return (
            f"Your PHQ-9 score is unchanged at {curr.phq9_score} since your last session."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats for a user (used in PDF report)
# ─────────────────────────────────────────────────────────────────────────────
def get_user_summary(user_id: str) -> dict:
    """Returns aggregate stats for reporting."""
    records = get_user_history(user_id)
    if not records:
        return {}

    phq9_scores = [r.phq9_score for r in records]
    gad7_scores = [r.gad7_score for r in records]

    return {
        "total_sessions":    len(records),
        "first_session":     records[0].session_date,
        "latest_session":    records[-1].session_date,
        "latest_phq9":       records[-1].phq9_score,
        "latest_gad7":       records[-1].gad7_score,
        "avg_phq9":          round(sum(phq9_scores) / len(phq9_scores), 1),
        "avg_gad7":          round(sum(gad7_scores) / len(gad7_scores), 1),
        "min_phq9":          min(phq9_scores),
        "max_phq9":          max(phq9_scores),
        "worsening_trend":   detect_worsening(records),
        "trend_message":     build_trend_message(records),
        "escalation_levels": [r.escalation_level for r in records],
    }
