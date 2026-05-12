"""PDF report generator for single-stock analysis.

Uses reportlab to produce a structured one-page summary PDF:
- Company header (name, ticker, sector, price)
- Recommendation verdict box
- Key metrics table (valuation, profitability, growth)
- DCF summary (if available)
- Quality scores (Piotroski, Altman, Beneish)
- Reasoning chain table (7 pillars)
- Investment thesis, bull case, bear case
"""
from __future__ import annotations

import io
from datetime import date
from typing import Optional


def generate_report(
    company_info: dict,
    fund_data: dict,
    recommendation,          # HolisticRecommendation
    verdict=None,            # InvestmentVerdict (may be None)
    dcf_result=None,         # DCFResult
    quality_scores=None,     # QualityScores
) -> bytes:
    """Return PDF as bytes. Raises ImportError if reportlab is not installed."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ---- Color map ----
    rec_colors = {
        "STRONG BUY": colors.HexColor("#00C853"),
        "BUY":        colors.HexColor("#69F0AE"),
        "HOLD":       colors.HexColor("#FFD54F"),
        "SELL":       colors.HexColor("#FF8A65"),
        "STRONG SELL":colors.HexColor("#FF1744"),
    }
    rec = recommendation.recommendation
    rec_color = rec_colors.get(rec, colors.grey)

    # ---- Header ----
    name = company_info.get("name", "Unknown")
    ticker = company_info.get("sector", "")
    sector = company_info.get("sector", "")
    industry = company_info.get("industry", "")
    price = company_info.get("price", 0)

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=10)
    normal = ParagraphStyle("normal", parent=styles["Normal"], fontSize=9, leading=13)
    small  = ParagraphStyle("small",  parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    bold9  = ParagraphStyle("bold9",  parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold")

    story.append(Paragraph(f"{name}", h1))
    story.append(Paragraph(f"{sector}  ·  {industry}  ·  ${price:.2f}  ·  {date.today().isoformat()}", small))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=8))

    # ---- Verdict banner ----
    verdict_tbl = Table(
        [[Paragraph(f"<b>{rec}</b>", ParagraphStyle("rv", parent=styles["Normal"],
                    fontSize=14, textColor=colors.white, alignment=TA_CENTER)),
          Paragraph(
              f"Score: {recommendation.overall_score:+.2f}  ·  "
              f"Confidence: {recommendation.confidence}  ·  "
              f"Fundamental: {recommendation.fundamental.overall_score:+.2f}  ·  "
              f"Technical: {recommendation.technical.overall_score:+.2f}",
              ParagraphStyle("rs", parent=styles["Normal"], fontSize=9,
                             textColor=colors.white, alignment=TA_CENTER))]],
        colWidths=["30%", "70%"],
    )
    verdict_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), rec_color),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#1c1f30")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [rec_color, colors.HexColor("#1c1f30")]),
        ("BOX", (0, 0), (-1, -1), 1, rec_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(verdict_tbl)
    story.append(Spacer(1, 10))

    # ---- Key Metrics ----
    story.append(Paragraph("Key Metrics", h2))

    def fmt(v, pct=False, dollar=False, dec=2):
        if v is None:
            return "N/A"
        if pct:
            return f"{v*100:.1f}%"
        if dollar:
            return f"${v:.{dec}f}"
        return f"{v:.{dec}f}"

    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["P/E (TTM)",      fmt(fund_data.get("pe_trailing"), dec=1),
         "ROE",            fmt(fund_data.get("roe"), pct=True)],
        ["Forward P/E",    fmt(fund_data.get("pe_forward"), dec=1),
         "ROA",            fmt(fund_data.get("roa"), pct=True)],
        ["PEG Ratio",      fmt(fund_data.get("peg_ratio")),
         "Net Margin",     fmt(fund_data.get("net_margin"), pct=True)],
        ["EV/EBITDA",      fmt(fund_data.get("ev_ebitda"), dec=1),
         "Gross Margin",   fmt(fund_data.get("gross_margin"), pct=True)],
        ["P/B Ratio",      fmt(fund_data.get("pb_ratio")),
         "Op. Margin",     fmt(fund_data.get("operating_margin"), pct=True)],
        ["P/S Ratio",      fmt(fund_data.get("ps_ratio")),
         "Revenue Growth", fmt(fund_data.get("revenue_growth"), pct=True)],
        ["Debt/Equity",    fmt(fund_data.get("debt_to_equity"), dec=1),
         "Earn. Growth",   fmt(fund_data.get("earnings_growth"), pct=True)],
        ["Current Ratio",  fmt(fund_data.get("current_ratio")),
         "Beta",           fmt(fund_data.get("beta"))],
        ["Div. Yield",     fmt(fund_data.get("dividend_yield"), pct=True),
         "Analyst Target", fmt(fund_data.get("target_mean_price"), dollar=True)],
    ]

    m_tbl = Table(metrics_data, colWidths=["30%", "20%", "30%", "20%"])
    m_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c1f30")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
    ]))
    story.append(m_tbl)
    story.append(Spacer(1, 8))

    # ---- DCF Summary ----
    if dcf_result and dcf_result.intrinsic_value and not dcf_result.error:
        story.append(Paragraph("DCF Intrinsic Value", h2))
        dcf_data = [
            ["Intrinsic Value", f"${dcf_result.intrinsic_value:.2f}",
             "Margin of Safety", f"{dcf_result.margin_of_safety:.1%}" if dcf_result.margin_of_safety else "N/A"],
            ["WACC", f"{dcf_result.wacc*100:.1f}%" if dcf_result.wacc else "N/A",
             "Upside / Downside", f"{dcf_result.upside_pct:+.1%}" if dcf_result.upside_pct else "N/A"],
            ["Stage 1 Growth", f"{dcf_result.growth_stage1*100:.1f}%" if dcf_result.growth_stage1 else "N/A",
             "Valuation",      dcf_result.valuation_label],
        ]
        dcf_tbl = Table(dcf_data, colWidths=["25%", "25%", "25%", "25%"])
        dcf_tbl.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ]))
        story.append(dcf_tbl)
        story.append(Spacer(1, 8))

    # ---- Quality Scores ----
    if quality_scores:
        story.append(Paragraph("Quality & Distress Scores", h2))
        qs_data = [
            ["Piotroski F-Score", f"{quality_scores.piotroski_f}/9" if quality_scores.piotroski_f is not None else "N/A",
             quality_scores.piotroski_label or ""],
            ["Altman Z-Score",   f"{quality_scores.altman_z:.2f}" if quality_scores.altman_z else "N/A",
             quality_scores.altman_zone or "N/A"],
            ["Beneish M-Score",  f"{quality_scores.beneish_m:.2f}" if quality_scores.beneish_m else "N/A",
             "MANIPULATION RISK" if quality_scores.beneish_flag else "Low Risk"],
        ]
        qs_tbl = Table(qs_data, colWidths=["25%", "20%", "55%"])
        qs_tbl.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(qs_tbl)
        story.append(Spacer(1, 8))

    # ---- Reasoning Chain ----
    if verdict and verdict.reasoning_chain:
        story.append(Paragraph("7-Pillar Reasoning Chain", h2))
        chain_data = [["Pillar", "Signal", "Score", "Headline"]]
        for step in verdict.reasoning_chain:
            chain_data.append([
                step.pillar,
                step.signal,
                f"{step.score:+.2f}",
                Paragraph(step.headline, normal),
            ])
        c_tbl = Table(chain_data, colWidths=["18%", "15%", "10%", "57%"])
        c_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c1f30")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(c_tbl)
        story.append(Spacer(1, 8))

    # ---- Thesis / Bull / Bear ----
    if verdict:
        story.append(Paragraph("Investment Thesis", h2))
        story.append(Paragraph(verdict.thesis or "N/A", normal))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Bull Case", h2))
        story.append(Paragraph(verdict.bull_case or "N/A", normal))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Bear Case / Key Risks", h2))
        story.append(Paragraph(verdict.bear_case or "N/A", normal))
        story.append(Spacer(1, 6))

    # ---- Footer ----
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=8))
    story.append(Paragraph(
        "Generated by Financial Analyst · Data from Yahoo Finance · Not financial advice",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
