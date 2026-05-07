import os
import json
import math
import io
import re
import psycopg2
import psycopg2.extras
import streamlit as st
from openai import OpenAI
from datetime import datetime, date
from fpdf import FPDF

# ── OpenAI client ──────────────────────────────────────────────────────────────
_base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
_ai_key   = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
_is_replit = bool(_base_url and _ai_key)

openai_client = OpenAI(
    api_key=_ai_key if _is_replit else os.environ.get("OPENAI_API_KEY"),
    base_url=_base_url if _is_replit else None,
)
AI_MODEL = "gpt-5.4" if _is_replit else "gpt-4o"

CYARA_PRODUCTS = ["Velocity", "Pulse", "Pulse 360", "Cruncher", "Botium", "ResolveAX", "Voice Assure"]

# ── Database ───────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor)

def fetch_all_analyses():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, company_name, industry, arr, health_score, health_category,
                       churn_risk_level, renewal_date, created_at
                FROM analyses ORDER BY created_at DESC
            """)
            return cur.fetchall()

def fetch_analysis(analysis_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM analyses WHERE id = %s", (analysis_id,))
            return cur.fetchone()

def delete_analysis(analysis_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM analyses WHERE id = %s", (analysis_id,))
        conn.commit()

def insert_analysis(data: dict) -> dict:
    cols = list(data.keys())
    vals = list(data.values())
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT INTO analyses ({col_names}) VALUES ({placeholders}) RETURNING *"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, vals)
            row = cur.fetchone()
        conn.commit()
    return dict(row)

# ── Health score logic (mirrors TypeScript exactly) ────────────────────────────
RISK_LEVELS = ["low", "medium", "high", "critical"]

def escalate_risk(current: str, minimum: str) -> str:
    return current if RISK_LEVELS.index(current) >= RISK_LEVELS.index(minimum) else minimum

def calculate_health_score(d: dict) -> int:
    score = 0
    score += min(20, (d["featureAdoptionPct"] / 100) * 20)
    score += min(15, (d["loginFrequencyPerMonth"] / 10) * 15)
    days = d["daysSinceLastActive"]
    if days <= 1:   score += 15
    elif days <= 7:  score += 12
    elif days <= 14: score += 8
    elif days <= 30: score += 4
    score += ((d["npsScore"] + 100) / 200) * 15
    tickets = d["openSupportTickets"]
    if tickets == 0:   score += 10
    elif tickets <= 2: score += 7
    elif tickets <= 5: score += 4
    score += (d["csatScore"] / 5) * 10
    eng_map = {"none": 0, "low": 2, "medium": 5, "high": 8}
    score += eng_map.get(d["executiveEngagementLevel"], 0)
    qbr = d["qbrLastHeldDaysAgo"]
    if qbr == -1:       score += 0
    elif qbr <= 90:     score += 4
    elif qbr <= 180:    score += 2
    if d["executiveSponsorPresent"]: score += 3
    product_count = len(d.get("products", []))
    if product_count >= 5:   score += 5
    elif product_count >= 3: score += 3
    elif product_count >= 1: score += 1
    product_usage = d.get("productUsage", {})
    if product_usage and product_count > 0:
        avg = sum(product_usage.values()) / len(product_usage)
        score += min(5, (avg / 100) * 5)
    return round(min(100, max(0, score)))

def get_health_category(score: int) -> str:
    if score >= 90: return "champion"
    if score >= 75: return "healthy"
    if score >= 30: return "at_risk"
    return "critical"

def get_churn_risk(score: int, category: str, d: dict) -> str:
    risk_map = {"critical": "critical", "at_risk": "high", "healthy": "low", "champion": "low"}
    return risk_map.get(category, "low")

def get_expansion_score(d: dict) -> int:
    score = 0
    score += (d["healthScore"] / 100) * 40
    score += ((100 - d["featureAdoptionPct"]) / 100) * 20
    score += min(15, (d["powerUsers"] / max(1, d["mau"])) * 150)
    eng_map = {"none": 0, "low": 5, "medium": 10, "high": 15}
    score += eng_map.get(d["executiveEngagementLevel"], 0)
    product_count = len(d.get("products", []))
    score += round(((7 - product_count) / 7) * 10)
    return round(min(100, max(0, score)))

# ── AI analysis ────────────────────────────────────────────────────────────────
def generate_ai_analysis(d: dict) -> dict:
    products      = d.get("products", [])
    product_usage = d.get("productUsage", {})
    unused = [p for p in CYARA_PRODUCTS if p not in products]
    product_summary = ", ".join(
        f"{p}: {product_usage[p]}% usage" if p in product_usage else p
        for p in products
    ) or "No Cyara products recorded."
    qbr_text = "Never" if d["qbrLastHeldDaysAgo"] == -1 else f"{d['qbrLastHeldDaysAgo']} days ago"

    prompt = f"""You are a Senior Customer Success Manager at Cyara (a CX testing and monitoring platform) providing an executive-ready analysis of a customer account. Based on the data below, provide a comprehensive, insightful analysis specific to Cyara's product portfolio.

CUSTOMER DATA:
- Company: {d['companyName']}
- Industry: {d['industry']}
- ARR: ${d['arr']:,.0f}
- Plan: {d['planTier']}
- Contract Start: {d['contractStartDate']}
- Renewal Date: {d['renewalDate']}
- Health Score: {d['healthScore']}/100 ({d['healthCategory'].replace('_', ' ')})
- Churn Risk: {d['churnRiskLevel']}
- Expansion Score: {d['expansionScore']}/100

CYARA PRODUCT USAGE:
- Active Products: {', '.join(products) if products else 'None recorded'}
- Per-Product Usage: {product_summary}
- Unused Cyara Products (expansion opportunities): {', '.join(unused) if unused else 'None — full platform adoption!'}

USAGE METRICS:
- Monthly Active Users: {d['mau']}
- Feature Adoption: {d['featureAdoptionPct']}%
- Login Frequency: {d['loginFrequencyPerMonth']} logins/user/month
- Days Since Last Active: {d['daysSinceLastActive']}

RELATIONSHIP & ENGAGEMENT:
- NPS Score: {d['npsScore']}
- CSAT Score: {d['csatScore']}/5
- Executive Engagement: {d['executiveEngagementLevel']}
- Executive Sponsor Present: {'Yes' if d['executiveSponsorPresent'] else 'No'}
- Last QBR: {qbr_text}
- Open Support Tickets: {d['openSupportTickets']}
- Power Users: {d['powerUsers']}
{f"- CSM Notes: {d['notes']}" if d.get('notes') else ''}

Respond ONLY with a valid JSON object (no markdown fences, no extra text) with these exact fields:
{{
  "executiveSummary": "2-3 sentence executive summary of account health and strategic position, mentioning specific Cyara products in use",
  "usageAnalysis": "2-3 sentence analysis of usage patterns across Cyara products, highlighting strong and weak adoption",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "risks": ["risk 1", "risk 2", "risk 3"],
  "nextSteps": ["specific action 1", "specific action 2", "specific action 3", "specific action 4"],
  "bestPractices": ["best practice 1 specific to Cyara products in use", "best practice 2", "best practice 3"],
  "recommendations": ["strategic recommendation 1 referencing specific Cyara products", "strategic recommendation 2", "strategic recommendation 3"],
  "expansionPotential": "2-3 sentence analysis of expansion opportunity, specifically calling out which unused Cyara products would be a natural fit and why",
  "fullReport": "A full, executive-ready markdown report covering all aspects: health overview, Cyara product adoption breakdown, usage analysis, relationship health, risks, opportunities, recommended next steps, and expansion potential. Reference specific Cyara product names throughout. Use ## headers. Be specific and data-driven. Minimum 400 words."
}}"""

    resp = openai_client.chat.completions.create(
        model=AI_MODEL,
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        p = json.loads(content)
        return {
            "executiveSummary":   str(p.get("executiveSummary", "")),
            "usageAnalysis":      str(p.get("usageAnalysis", "")),
            "strengths":          [str(x) for x in p.get("strengths", [])],
            "risks":              [str(x) for x in p.get("risks", [])],
            "nextSteps":          [str(x) for x in p.get("nextSteps", [])],
            "bestPractices":      [str(x) for x in p.get("bestPractices", [])],
            "recommendations":    [str(x) for x in p.get("recommendations", [])],
            "expansionPotential": str(p.get("expansionPotential", "")),
            "fullReport":         str(p.get("fullReport", "")),
        }
    except Exception:
        return {
            "executiveSummary": "Analysis generated.", "usageAnalysis": "See full report.",
            "strengths": [], "risks": [], "nextSteps": [], "bestPractices": [],
            "recommendations": [], "expansionPotential": "", "fullReport": content,
        }

# ── UI helpers ─────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "champion": "🟢", "healthy": "🟩",
    "at_risk": "🟠", "critical": "🔴",
}
RISK_COLORS = {
    "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴",
}

def health_badge(score: int, category: str) -> str:
    icon = CATEGORY_COLORS.get(category, "⚪")
    label = category.replace("_", " ").title()
    return f"{icon} **{score}/100** — {label}"

def risk_badge(level: str) -> str:
    icon = RISK_COLORS.get(level, "⚪")
    return f"{icon} {level.title()} Risk"

def score_bar(label: str, value: float, max_val: float = 100, suffix: str = "%"):
    pct = min(100, max(0, (value / max_val) * 100))
    if pct >= 70:   color = "#22c55e"
    elif pct >= 40: color = "#eab308"
    else:           color = "#ef4444"
    st.markdown(f"**{label}** — {value}{suffix}")
    st.markdown(
        f'<div style="background:#e5e7eb;border-radius:4px;height:10px">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:4px"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

# ── PDF export ─────────────────────────────────────────────────────────────────
def _plain(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

class SignalPDF(FPDF):
    def __init__(self, company: str):
        super().__init__()
        self.company = company
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Signal  |  Cyara Customer Success Health Analyzer  |  {self.company}", align="L")
        self.ln(2)
        self.set_draw_color(220, 220, 220)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()} — Confidential — Generated by Signal", align="C")

    def h1(self, text: str):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.ln(4)
        self.multi_cell(0, 8, _plain(text))
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def h2(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(241, 245, 249)
        self.set_text_color(15, 23, 42)
        self.ln(3)
        self.cell(0, 7, f"  {_plain(text)}", fill=True, ln=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, _plain(text))
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def bullet(self, text: str, prefix: str = "-"):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        indent = 6
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, f"{prefix} {_plain(text)}")
        self.set_text_color(0, 0, 0)

    def kv_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(55, 6, label + ":", ln=False)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(15, 15, 15)
        self.multi_cell(0, 6, _plain(value))

    def score_box(self, label: str, value: str, color: tuple):
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        box_w = (self.w - self.l_margin - self.r_margin - 9) / 4
        self.cell(box_w, 14, f"{label}", ln=False, fill=True, align="C")

    def progress_bar(self, label: str, pct: float, suffix: str = "%"):
        bar_w = self.w - self.l_margin - self.r_margin
        filled = bar_w * min(1.0, max(0.0, pct / 100))
        if pct >= 70:   fill_color = (34, 197, 94)
        elif pct >= 40: fill_color = (234, 179, 8)
        else:           fill_color = (239, 68, 68)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.cell(70, 5, _plain(label), ln=False)
        self.cell(20, 5, f"{pct:.0f}{suffix}", ln=False, align="R")
        self.ln(5)
        y = self.get_y()
        self.set_fill_color(229, 231, 235)
        self.rect(self.l_margin, y, bar_w, 4, "F")
        self.set_fill_color(*fill_color)
        if filled > 0:
            self.rect(self.l_margin, y, filled, 4, "F")
        self.ln(6)
        self.set_text_color(0, 0, 0)


def generate_pdf(row: dict, products: list, product_usage: dict,
                 strengths: list, risks: list, next_steps: list,
                 best_practices: list, recommendations: list) -> bytes:
    company = row["company_name"]
    pdf = SignalPDF(company)
    pdf.add_page()

    # Title
    pdf.h1(f"{company} — Executive Health Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%B %d, %Y')}  |  "
                   f"{row['industry']}  |  ${row['arr']:,.0f} ARR  |  {row['plan_tier'].title()} Plan", ln=True)
    pdf.ln(4)

    # Score summary boxes
    score = round(row["health_score"])
    cat   = row["health_category"].replace("_", " ").title()
    risk  = row["churn_risk_level"].title()
    exp   = round(row["expansion_score"])
    renew = str(row["renewal_date"])

    def _risk_color(r):
        return {"Low": (34,197,94), "Medium": (234,179,8), "High": (249,115,22), "Critical": (239,68,68)}.get(r, (100,100,100))
    def _score_color(s):
        return (34,197,94) if s >= 70 else (234,179,8) if s >= 40 else (239,68,68)

    box_w = (pdf.w - pdf.l_margin - pdf.r_margin - 6) / 4
    boxes = [
        ("Health Score", f"{score}/100\n{cat}", _score_color(score)),
        ("Churn Risk",   risk,                  _risk_color(risk)),
        ("Expansion",    f"{exp}/100",           _score_color(exp)),
        ("Renewal Date", renew,                  (71, 85, 105)),
    ]
    for i, (lbl, val, col) in enumerate(boxes):
        x = pdf.l_margin + i * (box_w + 2)
        y = pdf.get_y()
        pdf.set_fill_color(*col)
        pdf.rect(x, y, box_w, 16, "F")
        pdf.set_xy(x, y + 1)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(box_w, 5, lbl, align="C", ln=False)
        pdf.set_xy(x, y + 6)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(box_w, 5, val.split("\n")[0], align="C", ln=False)
        if "\n" in val:
            pdf.set_xy(x, y + 11)
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(box_w, 4, val.split("\n")[1], align="C", ln=False)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)

    # Executive summary
    pdf.h2("Executive Summary")
    pdf.body(row["executive_summary"])

    # Key metrics
    pdf.h2("Key Metrics")
    metrics = [
        ("Feature Adoption", row["feature_adoption_pct"], "%"),
        ("Health Score",     row["health_score"],         ""),
        ("NPS (normalized)", ((row["nps_score"] + 100) / 200) * 100, "%"),
        ("CSAT",             (row["csat_score"] / 5) * 100, "%"),
        ("Expansion Score",  row["expansion_score"],      ""),
        ("Login Frequency",  min(100, (row["login_frequency_per_month"] / 10) * 100), "%"),
    ]
    for lbl, val, sfx in metrics:
        pdf.progress_bar(lbl, val, sfx)
    pdf.ln(2)

    # Product adoption
    if products:
        pdf.h2("Cyara Product Adoption")
        for prod in products:
            usage = product_usage.get(prod, 0)
            pdf.progress_bar(prod, usage)
        unused = [p for p in CYARA_PRODUCTS if p not in products]
        if unused:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, f"Not yet adopted: {', '.join(unused)}")
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # Strengths & Risks
    pdf.h2("Strengths")
    for s in strengths:
        pdf.bullet(s, prefix="+ ")
    pdf.ln(2)

    pdf.h2("Risks")
    for r in risks:
        pdf.bullet(r, prefix="! ")
    pdf.ln(2)

    # Next steps
    pdf.h2("Next Steps")
    for i, step in enumerate(next_steps, 1):
        pdf.bullet(step, prefix=f"{i}.")
    pdf.ln(2)

    # Recommendations
    pdf.h2("Recommendations")
    for rec in recommendations:
        pdf.bullet(rec)
    pdf.ln(2)

    # Best practices
    pdf.h2("Best Practices")
    for bp in best_practices:
        pdf.bullet(bp)
    pdf.ln(2)

    # Usage analysis
    pdf.h2("Usage Analysis")
    pdf.body(row["usage_analysis"])
    pdf.ln(2)

    # Relationship
    pdf.h2("Relationship & Engagement")
    qbr_val = row["qbr_last_held_days_ago"]
    pdf.kv_row("Executive Engagement",  row["executive_engagement_level"].title())
    pdf.kv_row("Executive Sponsor",     "Yes" if row["executive_sponsor_present"] else "No")
    pdf.kv_row("Last QBR",              "Never" if qbr_val == -1 else f"{qbr_val} days ago")
    pdf.kv_row("Open Support Tickets",  str(row["open_support_tickets"]))
    pdf.kv_row("Power Users",           str(row["power_users"]))
    pdf.kv_row("Monthly Active Users",  str(row["mau"]))
    pdf.ln(2)

    # Expansion potential
    pdf.h2("Expansion Potential")
    pdf.body(row["expansion_potential"])
    pdf.ln(2)

    # Full report
    pdf.add_page()
    pdf.h2("Full Executive Report")
    for line in row["full_report"].splitlines():
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(15, 23, 42)
            pdf.ln(2)
            pdf.multi_cell(0, 6, _plain(stripped))
            pdf.set_text_color(0, 0, 0)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(15, 23, 42)
            pdf.ln(2)
            pdf.multi_cell(0, 7, _plain(stripped))
            pdf.set_text_color(0, 0, 0)
        elif stripped.startswith(("- ", "* ", "+ ")):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(0, 5.5, "- " + _plain(stripped[2:]))
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 5.5, _plain(stripped))
            pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())


# ── Pages ──────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.title("Signal — Customer Health Dashboard")
    st.caption("Cyara Customer Success Health Analyzer")

    rows = fetch_all_analyses()

    if not rows:
        st.info("No analyses yet. Use **New Analysis** in the sidebar to get started.")
        return

    # Stats row
    total = len(rows)
    avg_score = round(sum(r["health_score"] for r in rows) / total, 1)
    avg_arr   = round(sum(r["arr"] for r in rows) / total)
    high_risk = sum(1 for r in rows if r["churn_risk_level"] in ("high", "critical"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accounts", total)
    c2.metric("Avg Health Score", f"{avg_score}/100")
    c3.metric("Avg ARR", f"${avg_arr:,.0f}")
    c4.metric("High / Critical Risk", high_risk)

    st.divider()

    # Category summary
    st.subheader("Portfolio Overview")
    categories = ["champion", "healthy", "at_risk", "critical"]
    counts = {c: sum(1 for r in rows if r["health_category"] == c) for c in categories}
    cc = st.columns(4)
    for i, cat in enumerate(categories):
        icon = CATEGORY_COLORS[cat]
        cc[i].metric(f"{icon} {cat.replace('_',' ').title()}", counts[cat])

    st.divider()

    # Account list
    st.subheader("All Accounts")
    for row in rows:
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            col1.markdown(f"**{row['company_name']}**  \n{row['industry']}")
            col2.markdown(health_badge(round(row["health_score"]), row["health_category"]))
            col3.markdown(risk_badge(row["churn_risk_level"]))
            col4.markdown(f"${row['arr']:,.0f}")
            col5.markdown(f"Renews {row['renewal_date']}")
            if col5.button("View →", key=f"view_{row['id']}"):
                st.session_state["page"]         = "detail"
                st.session_state["selected_id"]  = row["id"]
                st.rerun()


def page_new_analysis():
    st.title("New Analysis")
    st.caption("Enter customer data to generate a health score and AI-powered executive briefing.")

    with st.form("analysis_form"):
        st.subheader("Company Details")
        c1, c2 = st.columns(2)
        company_name = c1.text_input("Company Name *")
        industry     = c2.text_input("Industry *")
        c3, c4 = st.columns(2)
        arr      = c3.number_input("ARR ($) *", min_value=0.0, step=10000.0)
        plan_tier = c4.selectbox("Plan Tier *", ["starter", "growth", "enterprise", "strategic"])
        c5, c6 = st.columns(2)
        contract_start = c5.date_input("Contract Start Date *", value=date.today())
        renewal_date   = c6.date_input("Renewal Date *", value=date.today())

        st.subheader("Cyara Products in Use")
        st.caption("Select which products this customer has, then enter usage %")
        selected_products = []
        product_usage = {}
        for prod in CYARA_PRODUCTS:
            row = st.columns([2, 3])
            has = row[0].checkbox(prod, key=f"prod_{prod}")
            if has:
                selected_products.append(prod)
                usage = row[1].slider(f"{prod} usage %", 0, 100, 50, key=f"usage_{prod}")
                product_usage[prod] = usage

        st.subheader("Usage Metrics")
        c7, c8, c9 = st.columns(3)
        mau                    = c7.number_input("Monthly Active Users *", min_value=0, step=1)
        feature_adoption_pct   = c8.slider("Feature Adoption %", 0, 100, 50)
        login_frequency        = c9.number_input("Login Frequency / Month", min_value=0.0, step=0.5, value=5.0)
        c10, c11 = st.columns(2)
        days_since_active = c10.number_input("Days Since Last Active", min_value=0, step=1)
        power_users       = c11.number_input("Power Users", min_value=0, step=1)

        st.subheader("Satisfaction & Support")
        c12, c13, c14 = st.columns(3)
        nps_score            = c12.slider("NPS Score", -100, 100, 0)
        csat_score           = c13.slider("CSAT Score (0–5)", 0.0, 5.0, 3.5, step=0.1)
        open_support_tickets = c14.number_input("Open Support Tickets", min_value=0, step=1)

        st.subheader("Executive Relationship")
        c15, c16 = st.columns(2)
        exec_engagement     = c15.selectbox("Executive Engagement Level", ["none", "low", "medium", "high"])
        exec_sponsor        = c16.checkbox("Executive Sponsor Present")
        qbr_never = st.checkbox("QBR Never Held")
        qbr_days  = st.number_input("Days Since Last QBR", min_value=0, step=1, disabled=qbr_never)

        notes = st.text_area("CSM Notes (optional)")

        submitted = st.form_submit_button("Generate Analysis", type="primary")

    if submitted:
        if not company_name or not industry or arr <= 0 or mau <= 0:
            st.error("Please fill in all required fields (Company Name, Industry, ARR, MAU).")
            return

        with st.spinner("Calculating health score and generating AI analysis…"):
            qbr_days_val = -1 if qbr_never else int(qbr_days)
            input_data = {
                "companyName":              company_name,
                "industry":                 industry,
                "arr":                      float(arr),
                "planTier":                 plan_tier,
                "contractStartDate":        str(contract_start),
                "renewalDate":              str(renewal_date),
                "mau":                      int(mau),
                "featureAdoptionPct":       float(feature_adoption_pct),
                "loginFrequencyPerMonth":   float(login_frequency),
                "daysSinceLastActive":      int(days_since_active),
                "npsScore":                 int(nps_score),
                "openSupportTickets":       int(open_support_tickets),
                "csatScore":                float(csat_score),
                "powerUsers":               int(power_users),
                "executiveEngagementLevel": exec_engagement,
                "qbrLastHeldDaysAgo":       qbr_days_val,
                "executiveSponsorPresent":  bool(exec_sponsor),
                "notes":                    notes or None,
                "products":                 selected_products,
                "productUsage":             product_usage,
            }

            health_score    = calculate_health_score(input_data)
            health_category = get_health_category(health_score)
            churn_risk      = get_churn_risk(health_score, health_category, input_data)
            input_data["healthScore"]    = health_score
            input_data["healthCategory"] = health_category
            input_data["churnRiskLevel"] = churn_risk
            expansion_score = get_expansion_score(input_data)
            input_data["expansionScore"] = expansion_score

            ai = generate_ai_analysis(input_data)

            row = insert_analysis({
                "company_name":              company_name,
                "industry":                  industry,
                "arr":                       float(arr),
                "plan_tier":                 plan_tier,
                "contract_start_date":       str(contract_start),
                "renewal_date":              str(renewal_date),
                "mau":                       int(mau),
                "feature_adoption_pct":      float(feature_adoption_pct),
                "login_frequency_per_month": float(login_frequency),
                "days_since_last_active":    int(days_since_active),
                "nps_score":                 int(nps_score),
                "open_support_tickets":      int(open_support_tickets),
                "csat_score":                float(csat_score),
                "power_users":               int(power_users),
                "executive_engagement_level": exec_engagement,
                "qbr_last_held_days_ago":    qbr_days_val,
                "executive_sponsor_present": bool(exec_sponsor),
                "notes":                     notes or None,
                "products":                  json.dumps(selected_products),
                "product_usage":             json.dumps(product_usage),
                "health_score":              float(health_score),
                "health_category":           health_category,
                "churn_risk_level":          churn_risk,
                "expansion_score":           float(expansion_score),
                "executive_summary":         ai["executiveSummary"],
                "usage_analysis":            ai["usageAnalysis"],
                "strengths":                 json.dumps(ai["strengths"]),
                "risks":                     json.dumps(ai["risks"]),
                "next_steps":                json.dumps(ai["nextSteps"]),
                "best_practices":            json.dumps(ai["bestPractices"]),
                "recommendations":           json.dumps(ai["recommendations"]),
                "expansion_potential":       ai["expansionPotential"],
                "full_report":               ai["fullReport"],
            })

        st.success("Analysis complete!")
        st.session_state["page"]        = "detail"
        st.session_state["selected_id"] = row["id"]
        st.rerun()


def page_detail():
    analysis_id = st.session_state.get("selected_id")
    if not analysis_id:
        st.error("No analysis selected.")
        return

    row = fetch_analysis(analysis_id)
    if not row:
        st.error("Analysis not found.")
        return

    row = dict(row)
    products      = json.loads(row.get("products") or "[]")
    product_usage = json.loads(row.get("product_usage") or "{}")
    strengths     = json.loads(row.get("strengths") or "[]")
    risks         = json.loads(row.get("risks") or "[]")
    next_steps    = json.loads(row.get("next_steps") or "[]")
    best_practices = json.loads(row.get("best_practices") or "[]")
    recommendations = json.loads(row.get("recommendations") or "[]")

    # Header
    col_back, col_title, col_pdf = st.columns([1, 7, 2])
    if col_back.button("← Back"):
        st.session_state["page"] = "dashboard"
        st.rerun()
    col_title.title(row["company_name"])
    st.caption(f"{row['industry']} · ${row['arr']:,.0f} ARR · {row['plan_tier'].title()} Plan")

    # PDF export button (top of page for easy access)
    with col_pdf:
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf(
            row, products, product_usage,
            strengths, risks, next_steps, best_practices, recommendations
        )
        safe_name = row["company_name"].replace(" ", "_").replace("/", "-")
        st.download_button(
            label="Export PDF",
            data=pdf_bytes,
            file_name=f"{safe_name}_health_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

    # Score cards
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Health Score", f"{round(row['health_score'])}/100",
              help=row["health_category"].replace("_", " ").title())
    s2.metric("Churn Risk", row["churn_risk_level"].title())
    s3.metric("Expansion Score", f"{round(row['expansion_score'])}/100")
    s4.metric("Renewal", row["renewal_date"])

    st.divider()

    # Executive summary
    st.subheader("Executive Summary")
    st.info(row["executive_summary"])

    # Scores visual
    st.subheader("Key Metrics")
    m1, m2 = st.columns(2)
    with m1:
        score_bar("Feature Adoption", row["feature_adoption_pct"])
        score_bar("Health Score", row["health_score"])
        score_bar("NPS (normalized)", ((row["nps_score"] + 100) / 200) * 100)
        score_bar("CSAT", (row["csat_score"] / 5) * 100)
    with m2:
        score_bar("Expansion Score", row["expansion_score"])
        score_bar("Login Frequency", min(100, (row["login_frequency_per_month"] / 10) * 100))
        score_bar("MAU", row["mau"], max_val=max(row["mau"], 1), suffix=" users")

    # Product usage
    if products:
        st.subheader("Cyara Product Adoption")
        unused = [p for p in CYARA_PRODUCTS if p not in products]
        for prod in products:
            usage = product_usage.get(prod, 0)
            score_bar(prod, usage)
        if unused:
            st.caption(f"**Not yet adopted:** {', '.join(unused)}")

    st.divider()

    # AI insights in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Insights", "Action Plan", "Usage Analysis", "Full Report"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Strengths")
            for s in strengths:
                st.success(f"✓ {s}")
            st.subheader("Recommendations")
            for r in recommendations:
                st.info(f"→ {r}")
        with c2:
            st.subheader("Risks")
            for r in risks:
                st.error(f"⚠ {r}")
            st.subheader("Expansion Potential")
            st.markdown(row["expansion_potential"])

    with tab2:
        st.subheader("Next Steps")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"**{i}.** {step}")
        st.subheader("Best Practices")
        for bp in best_practices:
            st.markdown(f"- {bp}")

    with tab3:
        st.subheader("Usage Analysis")
        st.markdown(row["usage_analysis"])
        st.subheader("Relationship & Engagement")
        r1, r2, r3 = st.columns(3)
        r1.metric("Executive Engagement", row["executive_engagement_level"].title())
        r2.metric("Executive Sponsor", "Yes" if row["executive_sponsor_present"] else "No")
        qbr_val = row["qbr_last_held_days_ago"]
        r3.metric("Last QBR", "Never" if qbr_val == -1 else f"{qbr_val} days ago")
        r4, r5 = st.columns(2)
        r4.metric("Open Support Tickets", row["open_support_tickets"])
        r5.metric("Power Users", row["power_users"])

    with tab4:
        st.subheader("Full Executive Report")
        st.markdown(row["full_report"])
        st.divider()
        report_text = f"# {row['company_name']} — Executive Report\n\n{row['full_report']}"
        st.download_button("Download Report (.md)", report_text,
                           file_name=f"{row['company_name'].replace(' ', '_')}_report.md",
                           mime="text/markdown")

    # Delete
    st.divider()
    with st.expander("Danger Zone"):
        if st.button("Delete this analysis", type="secondary"):
            delete_analysis(analysis_id)
            st.session_state["page"] = "dashboard"
            st.session_state.pop("selected_id", None)
            st.rerun()


# ── App shell ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Signal — Cyara CSM",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

    with st.sidebar:
        st.markdown("## Signal")
        st.caption("Cyara Customer Success Health Analyzer")
        st.divider()
        if st.button("Dashboard", use_container_width=True,
                     type="primary" if st.session_state["page"] == "dashboard" else "secondary"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        if st.button("New Analysis", use_container_width=True,
                     type="primary" if st.session_state["page"] == "new" else "secondary"):
            st.session_state["page"] = "new"
            st.rerun()
        st.divider()
        st.caption("Built for Cyara CSMs")

    page = st.session_state["page"]
    if page == "dashboard":
        page_dashboard()
    elif page == "new":
        page_new_analysis()
    elif page == "detail":
        page_detail()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
