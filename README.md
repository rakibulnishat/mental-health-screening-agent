# 🧠 Mental Health Screening Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?style=for-the-badge&logo=pytorch)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-green?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A conversational AI agent for clinical mental health screening using PHQ-9 and GAD-7**

[Live API](https://nishaatt-mental-health-screening-agent.hf.space/docs) · [Live UI](https://nishaatt-mental-health-screening-ui.hf.space) · [Report Bug](https://github.com/nishaatt/mental-health-screening-agent/issues)

</div>

---

> ⚠️ **Disclaimer:** This tool is a **screening and support system — NOT a clinical diagnostic tool**. All results must be discussed with a qualified mental health professional. Crisis escalation paths have not yet been validated by a mental health professional for production use.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Design Principle](#key-design-principle)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Screening Instruments](#screening-instruments)
- [Escalation Rules](#escalation-rules)
- [Model Files](#model-files)
- [Publication Potential](#publication-potential)
- [Author](#author)

---

## Overview

The Mental Health Screening Agent is a full-stack conversational AI system that conducts clinical mental health screenings through natural conversation. It combines:

- **Validated clinical instruments** — PHQ-9 (depression, 9 questions) and GAD-7 (anxiety, 7 questions)
- **Fine-tuned ML model** — MentalRoBERTa with three independent output heads (depression, anxiety, suicidal ideation)
- **LangGraph state machine** — auditable, deterministic conversation flow
- **Safety-first escalation** — hard-coded Python rules for all crisis decisions, never delegated to the LLM
- **Longitudinal tracking** — SQLite session storage with trend detection across sessions
- **PDF report generation** — downloadable reports for sharing with doctors

---

## Key Design Principle

```
The LLM acts as a conversational wrapper — not a clinical decision maker.
Scores, thresholds, and escalation paths are deterministic Python code.
This keeps the system auditable and safe.
```

The LLM (Groq LLaMA 3) is used **only** for generating empathetic closing messages. All clinical scoring, risk assessment, and crisis escalation are handled by deterministic Python code with no LLM involvement.

---

## System Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                │
│                                                         │
│  greet → assess → screen → score → escalation → respond │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  ML Model   │  │ PHQ-9 / GAD-7│  │  Escalation   │  │
│  │ MentalRoBERTa│  │  Screener    │  │  Rules        │  │
│  │ (ONNX)      │  │ (Deterministic)│  │ (Hard-coded)  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
ML Prediction      Clinical Score        Safety Action
(40% weight)       (60% weight)         (No LLM)
    │                    │
    └────────┬───────────┘
             ▼
        Fused Risk Score
             │
             ▼
    ┌────────────────┐
    │   SQLite DB    │  ← Session history, trend tracking
    └────────────────┘
             │
             ▼
      PDF Report
```

---

## Features

- ✅ **Conversational PHQ-9 screening** — 9 questions, scores 0–27, 5 severity levels
- ✅ **Conversational GAD-7 screening** — 7 questions, scores 0–21, 4 severity levels
- ✅ **MentalRoBERTa ML model** — fine-tuned on 51,000+ real mental health posts
- ✅ **Score fusion** — weighted combination of clinical scores (40%) and ML predictions (60%)
- ✅ **Deterministic escalation** — 4 levels: immediate, urgent, recommend, low
- ✅ **PHQ-9 Q9 safety rule** — suicidal ideation triggers immediate crisis response, never LLM-mediated
- ✅ **Crisis keyword detection** — bypasses model entirely for explicit crisis language
- ✅ **Longitudinal tracking** — trends surfaced across sessions ("Your PHQ-9 dropped from 14 to 9")
- ✅ **PDF report generation** — printable report for sharing with doctors
- ✅ **Full audit logging** — every state transition logged for auditability
- ✅ **Deployed on HuggingFace** — FastAPI backend + Streamlit UI

---

## Model Performance

The MentalRoBERTa model was trained on the [Mental Health sentiment dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (51,013 samples after deduplication).

| Task | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Depression Detection | 85.8% | 0.820 | 0.944 |
| Anxiety Detection | 96.3% | ~0.860 | — |
| Suicidal Ideation | 87.3% | 0.738 | — |

**Architecture:**
```
mental/mental-roberta-base backbone
→ [CLS] token → Dropout(0.3)
→ Three independent heads (768 → 256 → 1 each)
    ├── depression_head → Sigmoid
    ├── anxiety_head    → Sigmoid
    └── suicidal_head   → Sigmoid
```

**Known limitation:** Negation failure — "not depressed" may score high for depression. This is mitigated by the 60% weight given to clinical PHQ-9/GAD-7 scores in the fused risk calculation.

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | PyTorch → ONNX (MentalRoBERTa) |
| Agent Framework | LangGraph |
| LLM (empathetic replies) | Groq (LLaMA 3.1 8B Instant) |
| Text Features | HuggingFace Transformers (AutoTokenizer) |
| Memory / Sessions | SQLite + SQLModel |
| Safety Layer | Hard-coded Python rules |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Deployment | HuggingFace Spaces |

---

## Project Structure

```
mental_health_agent/
│
├── models/
│   └── mental_health_model.py      # ONNX model wrapper + LangChain tool
│
├── agent/
│   ├── screener.py                 # PHQ-9 + GAD-7 questions, scoring, state
│   ├── escalation.py               # Deterministic safety rules (no LLM)
│   └── langgraph_agent.py          # Full LangGraph state machine (6 nodes)
│
├── database/
│   └── db.py                       # SQLite session storage + trend detection
│
├── ui/
│   └── streamlit_app.py            # Streamlit chat UI
│
├── utils/
│   └── report.py                   # PDF report generator
│
├── app.py                          # FastAPI backend (deployment)
├── Dockerfile                      # Docker container config
├── requirements.txt                # Python dependencies
│
├── mental_health_model.onnx        # ONNX model (structure)
├── mental_health_model.onnx.data   # ONNX model weights (499MB, on HuggingFace)
├── mental_health_tokenizer/        # RoBERTa tokenizer files
└── thresholds.json                 # Calibrated classification thresholds
```

---

## Installation

### Prerequisites
- Python 3.10
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/nishaatt/mental-health-screening-agent
cd mental-health-screening-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your-groq-key-here
```

Get a free Groq API key at https://console.groq.com

### Model Files

Download the large model file from HuggingFace and place in project root:

```
mental_health_model.onnx        # Download from HuggingFace Space files
mental_health_model.onnx.data   # Download from HuggingFace Space files
mental_health_tokenizer/        # Download from HuggingFace Space files
thresholds.json                 # Download from HuggingFace Space files
```

HuggingFace Space files: https://huggingface.co/spaces/nishaatt/mental-health-screening-agent/tree/main

---

## Usage

### Run Streamlit UI (Local)

```bash
streamlit run ui/streamlit_app.py
```

Open browser at: http://localhost:8501

### Run FastAPI Backend (Local)

```bash
uvicorn app:api --host 0.0.0.0 --port 7860 --reload
```

Open Swagger UI at: http://localhost:7860/docs

### Test the Model

```bash
python models/mental_health_model.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/start` | Initialize session, get opening greeting |
| POST | `/chat` | Send message, get agent response |
| GET | `/history/{user_id}` | Get all past sessions |
| GET | `/report/{user_id}` | Download PDF report |

### Example: Start a Session

```bash
curl -X POST "https://nishaatt-mental-health-screening-agent.hf.space/start" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001"}'
```

### Example: Send a Message

```bash
curl -X POST "https://nishaatt-mental-health-screening-agent.hf.space/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "message": "1",
    "state": { ...state from /start response... }
  }'
```

---

## Screening Instruments

### PHQ-9 (Patient Health Questionnaire)

| Score | Severity |
|---|---|
| 0 – 4 | Minimal |
| 5 – 9 | Mild |
| 10 – 14 | Moderate |
| 15 – 19 | Moderately Severe |
| 20 – 27 | Severe |

### GAD-7 (Generalized Anxiety Disorder)

| Score | Severity |
|---|---|
| 0 – 4 | Minimal |
| 5 – 9 | Mild |
| 10 – 14 | Moderate |
| 15 – 21 | Severe |

### Score Fusion Formula

```python
depression_risk = 0.4 * (phq9_total / 27) + 0.6 * ml_depression_prob
anxiety_risk    = 0.4 * (gad7_total / 21) + 0.6 * ml_anxiety_prob
```

---

## Escalation Rules

All escalation decisions are made by deterministic Python code — never by the LLM.

| Level | Trigger | Action |
|---|---|---|
| **Immediate** | PHQ-9 Q9 ≥ 1 (suicidal ideation) | Surface crisis hotline immediately |
| **Urgent** | PHQ-9 ≥ 20 OR GAD-7 ≥ 15 | Strongly recommend professional help this week |
| **Recommend** | PHQ-9 ≥ 10 OR GAD-7 ≥ 10 | Suggest counselor, follow up in 2 weeks |
| **Low** | All scores below thresholds | Psychoeducation, check in next session |

### Crisis Resource (Bangladesh)
**Kaan Pete Roi — 01779-554391**

---

## Model Files

The large model weight file (`mental_health_model.onnx.data`, 499 MB) exceeds GitHub's file size limit. Download it directly from HuggingFace:

```
https://huggingface.co/spaces/nishaatt/mental-health-screening-agent/tree/main
```

Files needed:
- `mental_health_model.onnx`
- `mental_health_model.onnx.data`
- `mental_health_tokenizer/` (folder)
- `thresholds.json`

---


## Author

**Md. Rakibul Hasan Nishat**
Mechatronics Engineering, RUET (CGPA: 3.29/4.00)

- 🌐 Portfolio: [rakibulnishat.github.io](https://rakibulnishat.github.io)
- 💼 LinkedIn: [linkedin.com/in/nishaatt](https://linkedin.com/in/nishaatt)
- 🤗 HuggingFace: [huggingface.co/nishaatt](https://huggingface.co/nishaatt)
- 📧 Research interests: CNN-based transfer learning, Explainable AI, Edge AI, Embedded Systems

---

## License

This project is licensed under the MIT License.

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

Crisis Support (Bangladesh): **Kaan Pete Roi — 01779-554391**

</div>
