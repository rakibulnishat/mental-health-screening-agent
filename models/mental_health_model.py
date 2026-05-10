# models/mental_health_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Wraps the trained MentalRoBERTa ONNX model as a LangChain tool.
#
# Expects these files in the project ROOT (same folder as app.py):
#   mental_health_model.onnx
#   thresholds.json
#   mental_health_tokenizer/
#       tokenizer_config.json
#       vocab.json
#       merges.txt
#
# Architecture (from your Kaggle notebook):
#   mental/mental-roberta-base backbone
#   → [CLS] token → Dropout(0.3)
#   → three independent heads (768→256→1 each)
#   ONNX input names  : 'input_ids', 'attention_mask'  (int64, shape [batch,128])
#   ONNX output names : 'depression', 'anxiety', 'suicidal'  (raw logits)
#   Sigmoid is applied HERE, not inside ONNX graph
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import numpy as np
from langchain.tools import tool

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ONNX_PATH       = os.path.join(_ROOT, "mental_health_model.onnx")
_TOKENIZER_PATH  = os.path.join(_ROOT, "mental_health_tokenizer")
_THRESHOLDS_PATH = os.path.join(_ROOT, "thresholds.json")

MAX_LEN = 128   # must match Kaggle notebook

# ── Crisis keywords — same list as your notebook ──────────────────────────────
CRISIS_KEYWORDS = [
    'kill myself', 'want to die', 'end my life', 'take my life',
    'suicide', 'suicidal', 'no reason to live', 'better off dead',
    "can't go on", 'cannot go on', 'life is not worth', 'end it all',
    'want to end', 'plan to die',
]

MIN_CONFIDENCE = 0.45   # below this probability → UNCERTAIN

# ── Sigmoid (ONNX outputs raw logits) ─────────────────────────────────────────
def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-float(x))))

def _clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Lazy singletons ───────────────────────────────────────────────────────────
_tokenizer   = None
_ort_session = None
_thresholds  = None
_dummy_mode  = False

def _load_resources():
    global _tokenizer, _ort_session, _thresholds, _dummy_mode

    missing = []
    if not os.path.exists(_ONNX_PATH):       missing.append("mental_health_model.onnx")
    if not os.path.exists(_TOKENIZER_PATH):  missing.append("mental_health_tokenizer/")
    if not os.path.exists(_THRESHOLDS_PATH): missing.append("thresholds.json")

    if missing:
        print(f"[Model] WARNING: Missing: {missing}")
        print("[Model] Running in DUMMY mode.")
        _dummy_mode = True
        _thresholds = {"depression": 0.50, "anxiety": 0.70, "suicidal": 0.55}
        return

    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH)
        print(f"[Model] Tokenizer loaded.")
    except Exception as e:
        print(f"[Model] ERROR loading tokenizer: {e}")
        print("[Model] Run: pip install transformers")
        _dummy_mode = True
        _thresholds = {"depression": 0.50, "anxiety": 0.70, "suicidal": 0.55}
        return

    try:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(
            _ONNX_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"[Model] ONNX model loaded.")
    except Exception as e:
        print(f"[Model] ERROR loading ONNX: {e}")
        print("[Model] Run: pip install onnxruntime")
        _dummy_mode = True
        _thresholds = {"depression": 0.50, "anxiety": 0.70, "suicidal": 0.55}
        return

    try:
        with open(_THRESHOLDS_PATH) as f:
            _thresholds = json.load(f)
        print(f"[Model] Thresholds: {_thresholds}")
    except Exception as e:
        print(f"[Model] WARNING: thresholds.json failed ({e}). Using defaults.")
        _thresholds = {"depression": 0.50, "anxiety": 0.70, "suicidal": 0.55}

    _dummy_mode = False

def _ensure_loaded():
    if _thresholds is None:
        _load_resources()

# ── Core predict — mirrors notebook predict() exactly ─────────────────────────
def _predict(text: str) -> dict:
    _ensure_loaded()
    text_lower = text.lower()

    # Step 1: Crisis keyword check
    if any(kw in text_lower for kw in CRISIS_KEYWORDS):
        return {
            "depression_prob": None,
            "anxiety_prob":    None,
            "suicidal_prob":   None,
            "risk_level":      "CRISIS",
            "flags":           ["SUICIDAL"],
            "crisis_override": True,
            "message": (
                "Crisis detected. Please reach out for help immediately. "
                "Bangladesh: Kaan Pete Roi — 01779-554391. "
                "International: findahelpline.com"
            ),
        }

    # Step 2: Dummy mode
    if _dummy_mode:
        return _dummy_predict(text)

    # Step 3: Tokenize
    cleaned = _clean_text(text)
    enc = _tokenizer(
        cleaned,
        max_length     = MAX_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "np",
    )
    input_ids      = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    # Step 4: ONNX inference — output names: 'depression', 'anxiety', 'suicidal'
    dep_logit, anx_logit, sui_logit = _ort_session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    dep_prob = _sigmoid(dep_logit[0])
    anx_prob = _sigmoid(anx_logit[0])
    sui_prob = _sigmoid(sui_logit[0])

    # Step 5: Apply calibrated thresholds
    flags = []
    if sui_prob >= _thresholds["suicidal"]:    flags.append("SUICIDAL")
    if dep_prob >= _thresholds["depression"]:  flags.append("DEPRESSION")
    if anx_prob >= _thresholds["anxiety"]:     flags.append("ANXIETY")

    max_prob = max(dep_prob, anx_prob, sui_prob)

    # Step 6: Risk level
    if "SUICIDAL" in flags:
        risk    = "CRISIS"
        message = "Suicidal ideation detected. Kaan Pete Roi (BD): 01779-554391"
    elif max_prob < MIN_CONFIDENCE:
        risk    = "UNCERTAIN"
        message = "Not enough signal to classify. Please consult a professional."
    elif max_prob >= 0.75:
        risk    = "HIGH"
        message = "High risk detected. Consider reaching out to a mental health professional."
    elif max_prob >= _thresholds.get("depression", 0.5):
        risk    = "MODERATE"
        message = "Moderate risk. Consider speaking to someone you trust."
    else:
        risk    = "LOW"
        message = "Low risk detected."

    return {
        "depression_prob": round(dep_prob, 4),
        "anxiety_prob":    round(anx_prob, 4),
        "suicidal_prob":   round(sui_prob, 4),
        "risk_level":      risk,
        "flags":           flags,
        "crisis_override": False,
        "message":         message,
    }

# ── Dummy fallback ────────────────────────────────────────────────────────────
def _dummy_predict(text: str) -> dict:
    kw_dep = ["hopeless","worthless","sad","empty","tired","failure","alone","cry","numb"]
    kw_anx = ["anxious","worry","nervous","fear","panic","restless","stress","dread","tense"]
    t = text.lower()
    dep = min(sum(1 for k in kw_dep if k in t) / 4.0, 1.0)
    anx = min(sum(1 for k in kw_anx if k in t) / 4.0, 1.0)
    sui = 0.0
    flags = []
    if dep >= 0.5: flags.append("DEPRESSION")
    if anx >= 0.7: flags.append("ANXIETY")
    max_p = max(dep, anx, sui)
    risk = "HIGH" if max_p >= 0.75 else "MODERATE" if max_p >= 0.5 else "LOW" if max_p >= 0.45 else "UNCERTAIN"
    return {
        "depression_prob": round(dep, 4), "anxiety_prob": round(anx, 4),
        "suicidal_prob": round(sui, 4), "risk_level": risk,
        "flags": flags, "crisis_override": False,
        "message": f"{risk} risk (dummy mode — real model not loaded).",
    }

# ── LangChain Tool ────────────────────────────────────────────────────────────
@tool
def detect_mental_health_signal(text: str) -> dict:
    """
    Detects depression, anxiety, and suicidal ideation from free text.
    Uses fine-tuned MentalRoBERTa with three independent output heads.
    Returns structured probabilities and risk level.
    Used as grounded evidence by the agent — does NOT diagnose.
    """
    return _predict(text)

# ── Self-test: python models/mental_health_model.py ──────────────────────────
if __name__ == "__main__":
    tests = [
        ("I feel completely hopeless and can't sleep.",  "Depression expected"),
        ("I can't stop worrying, heart racing, panic.",  "Anxiety expected"),
        ("I had a good day, feeling fine overall.",      "Low/Uncertain expected"),
        ("I want to kill myself.",                       "CRISIS keyword expected"),
        ("Not depressed, just tired from work.",         "Negation test"),
    ]
    print("=" * 60)
    print("Mental Health Model — Self Test")
    print("=" * 60)
    for text, note in tests:
        r = _predict(text)
        print(f"\nText  : {text}")
        print(f"Note  : {note}")
        if r["crisis_override"]:
            print("Result: CRISIS (keyword triggered)")
        else:
            print(f"Result: dep={r['depression_prob']}  anx={r['anxiety_prob']}  "
                  f"sui={r['suicidal_prob']}  risk={r['risk_level']}  flags={r['flags']}")
