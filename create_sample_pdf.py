"""
create_sample_pdf.py
Generates a realistic sample PDF agreement with Schedule A for testing.
The PDF intentionally contains:
  - Some ISRCs in non-standard formats (to test normalization)
  - Some dates in non-standard formats (to test date parsing)
  - Missing writer data for some rows (filled from emails)
  - One extra song not present in any email (PDF-only)
  - One conflicting ISRC vs email (PDF wins)

Run: python create_sample_pdf.py
Output: sample_schedule_a.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT = "sample_schedule_a.pdf"

# Schedule A data — intentional discrepancies for conflict-resolution testing
SONGS = [
    # (Title, Artist, Writers, ISRC [possibly bad format], Release Date [possibly bad format])
    # Songs 1-5 (overlap with Email A — PDF is source of truth)
    ("Tokyo Midnight",  "The Neon Lights", "Alex Park, Jane Miller",  "USWB1-2400432",  "15/11/2024"),
    ("Shatter",         "Glass Atlas",     "Sarah Stone, Kevin Webb", "US-WB1-24-00433","2024-11-15"),
    ("Desert Rain",     "Mirage",          "Leyla Ademi",             "USWB12400434",   "Nov 22 2024"),
    ("Neon Dreams",     "The Neon Lights", "Alex Park, Jane Miller",  "US-WB1-24-00435","November 15, 2024"),
    ("Uptown",          "Funk Theory",     "",                        "US-WB1-24-00436","01/12/2024"),
    # Songs 6-10 (overlap with Email B)
    ("Golden Gate",     "Bridges",         "Rachel Davis",            "US-WB1-24-00437","Dec 05 2024"),
    ("Velocity",        "Turbo",           "Paul Walker",             "USWB1-2400438",  "05/12/2024"),
    ("Silent Echo",     "The Void",        "Elena Black",             "US-WB1-24-00439","12/12/2024"),
    ("Blue Monday",     "New Wave",        "",                        "US-WB1-24-00440","December 12, 2024"),
    ("Last Stop",       "Transit",         "John Doe",                "US-WB1-24-00441","20/12/2024"),
    # Song 11 — PDF-only (not in any email)
    ("Phantom City",    "The Neon Lights", "Alex Park",               "US-WB1-24-00442","2024-12-27"),
]


def build_pdf(path: str):
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title2",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a1a"),
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body2",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#1a1a1a"),
    )

    story = []

    # Header
    story.append(Paragraph("MUSIC PUBLISHING AGREEMENT", title_style))
    story.append(Paragraph("Warner Chappell Music, Inc.", sub_style))
    story.append(Paragraph("Agreement Reference: WCM-2024-NS-001 | Effective Date: October 1, 2024", sub_style))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9a010")))
    story.append(Spacer(1, 0.3 * cm))

    # Recitals
    story.append(Paragraph("RECITALS", section_style))
    story.append(Paragraph(
        "This Agreement is entered into by and between Warner Chappell Music, Inc. "
        "('Publisher') and the respective songwriters listed in Schedule A hereto "
        "('Writer(s)'), collectively referred to as the 'Parties'. The Publisher agrees "
        "to administer, promote, and collect royalties for the Compositions set forth in "
        "Schedule A, in accordance with the terms and conditions of this Agreement.",
        body_style,
    ))
    story.append(Paragraph(
        "NOW, THEREFORE, in consideration of the mutual covenants contained herein and for "
        "other good and valuable consideration, the receipt and sufficiency of which are "
        "hereby acknowledged, the Parties agree as follows:",
        body_style,
    ))

    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    # Schedule A
    story.append(Paragraph("SCHEDULE A — COMPOSITIONS", section_style))
    story.append(Paragraph(
        "The following compositions are covered under this Agreement. "
        "The ISRC and Release Date listed herein constitute the official record.",
        body_style,
    ))
    story.append(Spacer(1, 0.25 * cm))

    # Table header + data
    col_widths = [3.8 * cm, 3.0 * cm, 3.6 * cm, 3.6 * cm, 2.4 * cm]
    header = ["Song Title", "Recording Artist", "Writer(s)", "ISRC", "Release Date"]

    table_data = [header]
    for song in SONGS:
        table_data.append(list(song))

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.HexColor("#F5C518")),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 7),
        ("TOPPADDING",   (0, 0), (-1, 0), 7),
        # Body
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
        ("TOPPADDING",   (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        # Grid
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("LINEABOVE",    (0, 1), (-1, 1), 1, colors.HexColor("#c9a010")),
        # Alignment
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)

    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    # Footer note
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<i>Note: This Schedule A supersedes all prior lists or email communications "
        "regarding ISRCs and release dates. In case of conflict, this document shall "
        "be considered the authoritative source of record. Missing writer information "
        "should be sourced from accompanying correspondence.</i>",
        ParagraphStyle("Note", parent=body_style, textColor=colors.grey, fontSize=8),
    ))

    doc.build(story)
    print(f"✓ Sample PDF created: {path}")
    print(f"  Songs included: {len(SONGS)}")
    print(f"  Intentional issues:")
    print(f"    - Non-standard ISRC formats in rows 1, 3, 7")
    print(f"    - Non-standard date formats in rows 1, 3, 4, 5, 7, 8, 10")
    print(f"    - Missing writer info in rows 5, 10 (filled from emails)")
    print(f"    - Row 11 ('Phantom City') exists only in PDF")


if __name__ == "__main__":
    build_pdf(OUTPUT)
