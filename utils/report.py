# utils/report.py
# ─────────────────────────────────────────────────────────────────────────────
# Generates a printable PDF report the user can share with their doctor.
# Uses ReportLab. No raw conversation text is included — only scores and dates.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)


def generate_pdf_report(user_id: str, records: list, output_dir: str = ".") -> str:
    """
    Generates a PDF report for the given user and session records.

    Args:
        user_id    : User identifier
        records    : List of SessionRecord objects from database
        output_dir : Directory to save the PDF

    Returns:
        Path to the generated PDF file.
    """
    filename = f"{output_dir}/report_{user_id}_{datetime.today().strftime('%Y-%m-%d')}.pdf"
    doc      = SimpleDocTemplate(filename, pagesize=A4,
                                  leftMargin=20*mm, rightMargin=20*mm,
                                  topMargin=20*mm,  bottomMargin=20*mm)

    styles   = getSampleStyleSheet()
    elements = []

    # ── Title ────────────────────────────────────────────────────────────────
    elements.append(Paragraph("Mental Health Screening Report", styles["Title"]))
    elements.append(Spacer(1, 6*mm))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    disclaimer = (
        "<i>This report is produced by a screening and support tool — "
        "not a diagnostic system. All information is intended as supportive "
        "context for discussion with a qualified mental health professional. "
        "It does not constitute a clinical diagnosis.</i>"
    )
    elements.append(Paragraph(disclaimer, styles["Italic"]))
    elements.append(Spacer(1, 8*mm))

    # ── User info ────────────────────────────────────────────────────────────
    info_data = [
        ["User ID",         user_id],
        ["Report Generated", datetime.today().strftime("%Y-%m-%d %H:%M")],
        ["Total Sessions",  str(len(records))],
    ]
    if records:
        info_data.append(["First Session", str(records[0].session_date)])
        info_data.append(["Latest Session", str(records[-1].session_date)])

    info_table = Table(info_data, colWidths=[55*mm, 110*mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F4FD")),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#F9F9F9")]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10*mm))

    # ── Session history table ────────────────────────────────────────────────
    elements.append(Paragraph("Session History", styles["Heading2"]))
    elements.append(Spacer(1, 4*mm))

    if records:
        headers = ["Date", "PHQ-9", "PHQ-9 Level", "GAD-7", "GAD-7 Level", "Escalation"]
        rows    = [headers]

        for r in records:
            # Compute severity labels from scores
            phq9_sev = _phq9_severity(r.phq9_score)
            gad7_sev = _gad7_severity(r.gad7_score)
            rows.append([
                str(r.session_date),
                str(r.phq9_score),
                phq9_sev,
                str(r.gad7_score),
                gad7_sev,
                r.escalation_level.capitalize(),
            ])

        col_widths = [30*mm, 18*mm, 32*mm, 18*mm, 32*mm, 30*mm]
        hist_table = Table(rows, colWidths=col_widths)
        hist_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#2C5F8A")),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F2F8FC")]),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
            ("ALIGN",         (1,0), (3,-1), "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))

        # Highlight immediate escalation rows in red
        for i, r in enumerate(records, start=1):
            if r.escalation_level == "immediate":
                hist_table.setStyle(TableStyle([
                    ("BACKGROUND", (0,i), (-1,i), colors.HexColor("#FDECEA")),
                ]))

        elements.append(hist_table)
    else:
        elements.append(Paragraph("No sessions recorded yet.", styles["Normal"]))

    elements.append(Spacer(1, 10*mm))

    # ── Score key ────────────────────────────────────────────────────────────
    elements.append(Paragraph("Score Reference", styles["Heading2"]))
    elements.append(Spacer(1, 4*mm))

    key_data = [
        ["PHQ-9 Score", "Severity",      "GAD-7 Score", "Severity"],
        ["0 – 4",       "Minimal",        "0 – 4",       "Minimal"],
        ["5 – 9",       "Mild",           "5 – 9",       "Mild"],
        ["10 – 14",     "Moderate",       "10 – 14",     "Moderate"],
        ["15 – 19",     "Mod. Severe",    "15 – 21",     "Severe"],
        ["20 – 27",     "Severe",         "",             ""],
    ]
    key_table = Table(key_data, colWidths=[30*mm, 40*mm, 30*mm, 40*mm])
    key_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#2C5F8A")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F2F8FC")]),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    elements.append(key_table)
    elements.append(Spacer(1, 10*mm))

    # ── Footer note ──────────────────────────────────────────────────────────
    elements.append(Paragraph(
        "<i>Crisis support (Bangladesh): Kaan Pete Roi — 01779-554391</i>",
        styles["Normal"]
    ))

    doc.build(elements)
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# Severity label helpers
# ─────────────────────────────────────────────────────────────────────────────
def _phq9_severity(score: int) -> str:
    if score <= 4:   return "Minimal"
    if score <= 9:   return "Mild"
    if score <= 14:  return "Moderate"
    if score <= 19:  return "Mod. Severe"
    return "Severe"

def _gad7_severity(score: int) -> str:
    if score <= 4:   return "Minimal"
    if score <= 9:   return "Mild"
    if score <= 14:  return "Moderate"
    return "Severe"
