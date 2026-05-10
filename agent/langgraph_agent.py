# agent/langgraph_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Full LangGraph state machine:
# greet → assess → screen → score → escalation → respond → [loop or end]
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()  # ← add this line

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from .screener import (
    fresh_screening_state,
    get_next_question,
    record_answer,
    compute_phq9_score,
    compute_gad7_score,
    compute_fused_risk,
    parse_score_from_text,
    SCORE_OPTIONS_TEXT,
)
from .escalation import apply_escalation, check_immediate_crisis
from models.mental_health_model import detect_mental_health_signal

# ─────────────────────────────────────────────────────────────────────────────
# Audit logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="agent_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

def _log(event: str, state: "AgentState"):
    logging.info(
        f"{event} | user={state['user_id']} | "
        f"phq9={state['screening'].get('phq9_total', '?')} | "
        f"gad7={state['screening'].get('gad7_total', '?')} | "
        f"escalation={state.get('escalation_level', '?')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_id:          str
    messages:         list
    screening:        dict
    ml_result:        dict
    fused_risk:       dict
    escalation_level: str
    session_complete: bool
    current_question: dict
    awaiting_answer:  bool
    greeted:          bool   # True after greeting is done


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Gemini
# ─────────────────────────────────────────────────────────────────────────────
def _get_llm():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Check your .env file.")
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Node 1: greet — ONLY shows greeting, does NOT ask Q1 yet
# ─────────────────────────────────────────────────────────────────────────────
def greet_node(state: AgentState) -> AgentState:
    _log("NODE:greet", state)

    # Only greet once
    if not state.get("greeted"):
        state["messages"].append({
            "role": "assistant",
            "content": (
                "Hello. I am here to check in with you today. "
                "This is a safe space — please feel free to share how you have been feeling.\n\n"
                "I will now ask you 16 short questions (PHQ-9 and GAD-7) to better understand "
                "your wellbeing. For each question, please reply with a number:\n"
                f"{SCORE_OPTIONS_TEXT}\n\n"
                "Let us begin."
            )
        })
        state["greeted"] = True

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: assess — run ML tool on latest user message
# ─────────────────────────────────────────────────────────────────────────────
def assess_node(state: AgentState) -> AgentState:
    _log("NODE:assess", state)
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    if user_msgs:
        last_msg = user_msgs[-1]["content"]
        result = detect_mental_health_signal.invoke(last_msg)
        state["ml_result"] = result
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: screen
# ─────────────────────────────────────────────────────────────────────────────
def screen_node(state: AgentState) -> AgentState:
    _log("NODE:screen", state)

    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    last_user = user_msgs[-1]["content"] if user_msgs else ""

    # ── If we are waiting for an answer to a specific question ───────────────
    if (state.get("awaiting_answer")
            and state.get("current_question")
            and len(state["current_question"]) > 0):

        score = parse_score_from_text(last_user)
        cq    = state["current_question"]

        if score is None:
            # Cannot parse — re-ask same question
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"I didn't quite catch that. Please reply with just a number "
                    f"(0, 1, 2, or 3) for the question:\n\n"
                    f"\"{cq['text']}\"\n\n"
                    f"{SCORE_OPTIONS_TEXT}"
                )
            })
            return state

        # Record valid answer
        record_answer(state["screening"], cq["instrument"], cq["q_num"], score)

        # CRITICAL: check PHQ-9 Q9 immediately
        if cq["instrument"] == "phq9" and cq["q_num"] == 9:
            if check_immediate_crisis(state["screening"]):
                state["awaiting_answer"] = False
                state["current_question"] = {}
                return state

    # ── Ask the next question ─────────────────────────────────────────────────
    instrument, q_num, q_text = get_next_question(state["screening"])

    if q_text:
        state["current_question"] = {
            "instrument": instrument,
            "q_num":      q_num,
            "text":       q_text,
        }
        state["awaiting_answer"] = True

        # Show which question number out of total
        if instrument == "phq9":
            q_label = f"PHQ-9 Question {q_num} of 9"
        else:
            q_label = f"GAD-7 Question {q_num} of 7"

        state["messages"].append({
            "role": "assistant",
            "content": (
                f"**{q_label}**\n\n"
                f"{q_text}\n\n"
                f"{SCORE_OPTIONS_TEXT}"
            )
        })
    else:
        # All questions answered
        state["awaiting_answer"]                 = False
        state["current_question"]                = {}
        state["screening"]["screening_complete"] = True

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: score
# ─────────────────────────────────────────────────────────────────────────────
def score_node(state: AgentState) -> AgentState:
    _log("NODE:score", state)

    phq9_result = compute_phq9_score(state["screening"]["phq9_answers"])
    gad7_result = compute_gad7_score(state["screening"]["gad7_answers"])

    state["screening"]["phq9_total"]    = phq9_result["total"]
    state["screening"]["gad7_total"]    = gad7_result["total"]
    state["screening"]["phq9_severity"] = phq9_result["severity"]
    state["screening"]["gad7_severity"] = gad7_result["severity"]

    ml_dep = state["ml_result"].get("depression_prob") or 0.0
    ml_anx = state["ml_result"].get("anxiety_prob")    or 0.0

    state["fused_risk"] = compute_fused_risk(
        phq9_result["total"],
        gad7_result["total"],
        ml_dep,
        ml_anx,
    )

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: escalation — deterministic, NO LLM
# ─────────────────────────────────────────────────────────────────────────────
def escalation_node(state: AgentState) -> AgentState:
    _log("NODE:escalation", state)

    result = apply_escalation(state["screening"])
    state["escalation_level"] = result["level"]

    state["messages"].append({
        "role": "assistant",
        "content": result["message"]
    })

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: respond — LLM empathetic closing only
# ─────────────────────────────────────────────────────────────────────────────
def respond_node(state: AgentState) -> AgentState:
    _log("NODE:respond", state)

    if state["escalation_level"] == "immediate":
        state["session_complete"] = True
        return state

    llm = _get_llm()

    system_prompt = (
        "You are a compassionate mental health support assistant. "
        "Your role is ONLY to provide empathetic, supportive conversation. "
        "You are NOT a doctor, therapist, or diagnostician. "
        "NEVER make a diagnosis. NEVER give medical advice. "
        "NEVER override or contradict the clinical safety messages already sent. "
        "Keep your response warm, brief (2-3 sentences), and supportive."
    )

    user_context = (
        f"The user just completed a mental health screening.\n"
        f"PHQ-9 score: {state['screening'].get('phq9_total', 'N/A')} "
        f"({state['screening'].get('phq9_severity', '')})\n"
        f"GAD-7 score: {state['screening'].get('gad7_total', 'N/A')} "
        f"({state['screening'].get('gad7_severity', '')})\n"
        f"Escalation level: {state['escalation_level']}\n\n"
        f"Write a brief, warm closing message. "
        f"Do NOT repeat the clinical recommendation already given. "
        f"Do NOT diagnose. Simply acknowledge their effort and offer encouragement."
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_context},
    ])

    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })

    state["session_complete"] = True
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────
def route_after_greet(state: AgentState) -> str:
    return "screen"


def route_after_assess(state: AgentState) -> str:
    if check_immediate_crisis(state["screening"]):
        return "escalation"
    return "screen"


def route_after_screen(state: AgentState) -> str:
    if check_immediate_crisis(state["screening"]):
        return "score"
    if state["screening"].get("screening_complete"):
        return "score"
    return END


def route_after_score(state: AgentState) -> str:
    return "escalation"


def route_after_escalation(state: AgentState) -> str:
    if state["escalation_level"] == "immediate":
        return END
    return "respond"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("greet",      greet_node)
    g.add_node("assess",     assess_node)
    g.add_node("screen",     screen_node)
    g.add_node("score",      score_node)
    g.add_node("escalation", escalation_node)
    g.add_node("respond",    respond_node)

    g.set_entry_point("greet")

    # On first load: greet → screen (ask Q1, then END and wait for user)
    g.add_conditional_edges("greet", route_after_greet)

    # On each user reply: assess → screen (record answer, ask next Q)
    g.add_conditional_edges("assess",     route_after_assess)
    g.add_conditional_edges("screen",     route_after_screen)
    g.add_conditional_edges("score",      route_after_score)
    g.add_conditional_edges("escalation", route_after_escalation)
    g.add_edge("respond", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Fresh state factory
# ─────────────────────────────────────────────────────────────────────────────
def fresh_agent_state(user_id: str) -> AgentState:
    return {
        "user_id":          user_id,
        "messages":         [],
        "screening":        fresh_screening_state(),
        "ml_result":        {},
        "fused_risk":       {},
        "escalation_level": "",
        "session_complete": False,
        "current_question": {},
        "awaiting_answer":  False,
        "greeted":          False,
    }