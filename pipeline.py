"""
pipeline.py  —  Metadata Cleaning Pipeline
Warner Chappell Music Intelligence · Task 1

Key design decisions:
  - PDF is the source of truth. For any field where PDF has a real value
    (not blank / TBD / N/A), the PDF value wins over email values.
  - Emails fill gaps only (blank / TBD / missing fields in PDF).
  - Matching across sources is done by normalised ISRC first, then by
    normalised song title as a fallback.
  - PDF text is extracted by pdfplumber and parsed by LLM so column
    wrapping artefacts (e.g. "Shattered Glass" spanning two cells) are
    handled correctly via raw text parsing.
"""

from __future__ import annotations

import re
import json
import io
import csv
from typing import Optional


# ── Default email content (editable in the UI) ────────────────────────────────

EMAIL_A = """From: creative@neonpublishing.com
To: metadata@neonpublishing.com
Subject: Schedule A — Tracks 1-5 Metadata Confirmation

Hi team,

Here are the confirmed details for the first batch of releases.

1. Song Title: Tokyo Midnight
   Recording Artist: The Neon Lights
   Writers: Alex Park, Jane Miller
   ISRC: US-WB1-24-00432
   Release Date: November 15, 2024

2. Song Title: Shatter Glass
   Recording Artist: Atlas
   Writers: Sarah Stone, Kevin Webb
   ISRC: US-WB1-24-00433
   Release Date: November 15, 2024

3. Song Title: Desert Rain
   Recording Artist: Mirage
   Writers: Leyla Ademi
   ISRC: US-WB1-24-00434
   Release Date: November 22, 2024

4. Song Title: Neon Dreams
   Recording Artist: The Neon Lights
   Writers: Alex Park, Jane Miller
   ISRC: US-WB1-24-00435
   Release Date: November 15, 2024

5. Song Title: Uptown Funk
   Recording Artist: Funk Theory
   Writers: Marcus Davis
   ISRC: US-WB1-24-00436
   Release Date: December 1, 2024

Best,
Creative Department"""

EMAIL_B = """From: management@neonpublishing.com
To: metadata@neonpublishing.com
Subject: Schedule A — Tracks 6-11 Metadata Update

Team,

Sending over the remaining tracks for Schedule A.

6. Song Title: Golden Gate
   Recording Artist: Bridges
   Writers: TBD
   ISRC: US-WB1-24-00437
   Release Date: December 5, 2024

7. Song Title: Velocity
   Recording Artist: Turbo
   Writers: Paul Walker
   ISRC: US-WB1-24-00438
   Release Date: December 5, 2024

8. Song Title: Silent Echo
   Recording Artist: The Void
   Writers: Elena Black
   ISRC: US-WB1-24-00439
   Release Date: December 12, 2024

9. Song Title: Blue Monday
   Recording Artist: New Wave
   Writers: George Sumner
   ISRC: US-WB1-24-00440
   Release Date: December 12, 2024

10. Song Title: Last Stop
    Recording Artist: Transit
    Writers: John Doe
    ISRC: US-WB1-24-00441
    Release Date: December 20, 2024

11. Song Title: Phantom City
    Recording Artist: The Neon Lights
    Writers: Alex Park
    ISRC: US-WB1-24-00442
    Release Date: December 27, 2024

Thanks,
Artist Management"""


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalize_isrc(raw: str) -> str:
    """Normalise ISRC to CC-RRR-YY-NNNNN format. Returns '' if unrecognisable."""
    if not raw:
        return ""
    s = raw.strip()
    # Try structured regex first
    m = re.match(r"([A-Za-z]{2})[.\-]?([A-Za-z0-9]{3})[.\-]?(\d{2})[.\-]?(\d{5})", s)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}-{m.group(3)}-{m.group(4)}"
    # Fall back to collapsing all non-alphanum and reformatting if 12 chars
    digits = re.sub(r"[^A-Z0-9]", "", s.upper())
    if len(digits) == 12:
        return f"{digits[0:2]}-{digits[2:5]}-{digits[5:7]}-{digits[7:12]}"
    return s


def normalize_date(raw: str) -> str:
    """Normalise various date formats to YYYY-MM-DD. Returns '' if unrecognisable."""
    if not raw:
        return ""
    raw = raw.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}$", raw):
        return raw
    months = {
        "january":"01","february":"02","march":"03","april":"04",
        "may":"05","june":"06","july":"07","august":"08",
        "september":"09","october":"10","november":"11","december":"12",
        "jan":"01","feb":"02","mar":"03","apr":"04",
        "jun":"06","jul":"07","aug":"08",
        "sep":"09","oct":"10","nov":"11","dec":"12",
    }
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", raw, re.I)
    if m and m.group(1).lower() in months:
        return f"{m.group(3)}-{months[m.group(1).lower()]}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return raw


def _is_empty(val: str) -> bool:
    """True when a field is meaningfully absent."""
    return not val or val.strip().upper() in {
        "", "TBD", "N/A", "MISSING", "NONE", "-", "?", "NA", "NULL"
    }


def _norm_isrc(isrc: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", isrc.upper())


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()


# ── Stage 1 — PDF extraction ──────────────────────────────────────────────────

def extract_pdf_text(pdf_file) -> tuple[str, str]:
    """
    Extract raw text from the PDF using pdfplumber.
    Returns (all_pages_text, schedule_a_page_text).
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    if hasattr(pdf_file, "read"):
        data = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        source = io.BytesIO(data)
    else:
        source = pdf_file

    all_texts = []
    schedule_a_text = ""

    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_texts.append(text)
            # Identify the Schedule A page: has song table headers
            if ("schedule a" in text.lower() or "song title" in text.lower()) and \
               ("isrc" in text.lower()):
                schedule_a_text = text

    return "\n\n".join(all_texts), schedule_a_text


# ── Stage 2 — LLM: parse PDF ─────────────────────────────────────────────────

_PDF_PARSE_PROMPT = """You are a music publishing metadata specialist.

The raw text below was extracted from a Schedule A table in a publishing agreement.
The table columns are (in order):
  Track | Song Title | Primary Artist | Writers (Full Names) | Release Date | ISRC

CRITICAL parsing rules:
1. Song titles and artist names sometimes wrap across lines (e.g. "Tokyo\nMidnight" = "Tokyo Midnight").
2. When a line reads like "Word1 Word2 ArtistName", figure out which words are the title vs artist.
   EXAMPLE: "Shattered Glass Atlas" → title="Shattered Glass", artist="Atlas"
   EXAMPLE: "Uptown Funk Theory" → title="Uptown", artist="Funk Theory"  (because "Funk Theory" is the known band)
   EXAMPLE: "Golden Gates Bridges" or "Golden\nGates Bridges" → title="Golden Gates", artist="Bridges"
3. "TBD", "N/A", blank → return as empty string "".
4. Writers who appear in the Writers column should be listed there, not as artists.
5. Normalize ISRC to CC-RRR-YY-NNNNN format.
6. Normalize dates to YYYY-MM-DD.
7. Return EVERY track found, even those with missing fields.

Return ONLY valid JSON (no markdown fences):
{
  "songs": [
    {
      "song_title": "...",
      "primary_artist": "...",
      "writers": "...",
      "release_date": "YYYY-MM-DD or empty string",
      "isrc": "CC-RRR-YY-NNNNN or empty string"
    }
  ]
}"""


def llm_parse_pdf(raw_text: str, client, model: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PDF_PARSE_PROMPT},
            {"role": "user", "content": f"Schedule A raw extracted text:\n\n{raw_text}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw).get("songs", [])
    except Exception as e:
        raise RuntimeError(f"PDF LLM JSON parse error: {e}\nRaw: {raw[:400]}")


# ── Stage 3 — LLM: parse emails ──────────────────────────────────────────────

_EMAIL_PARSE_PROMPT = """You are a music publishing metadata specialist.

Extract every song's metadata from the email below.
For any missing/unknown field return an empty string "".
Writers listed as "TBD", "N/A", or similar → return "".

Return ONLY valid JSON (no markdown fences):
{
  "songs": [
    {
      "song_title": "...",
      "primary_artist": "...",
      "writers": "...",
      "release_date": "YYYY-MM-DD or empty string",
      "isrc": "CC-RRR-YY-NNNNN or empty string"
    }
  ]
}"""


def llm_parse_email(email_text: str, client, model: str, label: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EMAIL_PARSE_PROMPT},
            {"role": "user", "content": f"Email:\n\n{email_text}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw).get("songs", [])
    except Exception as e:
        raise RuntimeError(f"{label} LLM JSON parse error: {e}\nRaw: {raw[:400]}")


# ── Stage 4 — Merge (PDF wins conflicts) ─────────────────────────────────────

def _normalise_song(raw: dict) -> dict:
    return {
        "song_title":     (raw.get("song_title") or "").strip(),
        "primary_artist": (raw.get("primary_artist") or "").strip(),
        "writers":        (raw.get("writers") or "").strip(),
        "release_date":   normalize_date(raw.get("release_date") or ""),
        "isrc":           normalize_isrc(raw.get("isrc") or ""),
    }


def _find_match(target: dict, pool: list[dict]) -> Optional[int]:
    """
    Return index of the best matching record in pool, or None.
    Matching priority: normalised ISRC → exact normalised title → partial title.
    """
    t_isrc  = _norm_isrc(target["isrc"])
    t_title = _norm_title(target["song_title"])

    # 1. ISRC match (most reliable)
    if t_isrc:
        for i, s in enumerate(pool):
            if _norm_isrc(s["isrc"]) == t_isrc:
                return i

    # 2. Exact normalised title
    if t_title:
        for i, s in enumerate(pool):
            if _norm_title(s["song_title"]) == t_title:
                return i

    # 3. One title is wholly contained in the other (≥ 2 words to avoid false positives)
    if len(t_title.split()) >= 2:
        for i, s in enumerate(pool):
            st = _norm_title(s["song_title"])
            if st and (t_title in st or st in t_title):
                return i

    return None


def merge_sources(
    pdf_songs: list[dict],
    email_a_songs: list[dict],
    email_b_songs: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Merge PDF + Email A + Email B.

    For each PDF record:
      - Find a matching email record (ISRC-first, then title).
      - PDF field wins whenever PDF has a real (non-empty, non-TBD) value.
      - Email fills gaps: if PDF field is empty/TBD and email has a value, use email.

    Email-only records (no PDF match) are appended with source label "Email X only".
    """
    conflict_log: list[str] = []
    fill_log:     list[str] = []
    clean_records: list[dict] = []

    # Combined email pool with labels
    email_pool: list[tuple[dict, str]] = (
        [(s, "Email A") for s in email_a_songs] +
        [(s, "Email B") for s in email_b_songs]
    )
    matched_email_idx: set[int] = set()

    # Fields to compare: (pdf_key, output_display_key, email_key)
    FIELDS = [
        ("song_title",     "Song Title",       "song_title"),
        ("primary_artist", "Recording Artist", "primary_artist"),
        ("writers",        "Writers",          "writers"),
        ("isrc",           "ISRC",             "isrc"),
        ("release_date",   "Release Date",     "release_date"),
    ]

    for pdf in pdf_songs:
        display_title = pdf["song_title"] or "Unknown"

        # Build initial record seeded entirely from PDF
        record: dict = {
            "Song Title":       pdf["song_title"],
            "Writers":          pdf["writers"],
            "Recording Artist": pdf["primary_artist"],
            "ISRC":             pdf["isrc"],
            "Release Date":     pdf["release_date"],
            "_source_note":     "PDF only",
        }

        # Find best email match
        email_match: Optional[dict] = None
        email_label = ""
        email_sources_used: list[str] = []

        for idx, (es, elabel) in enumerate(email_pool):
            if idx in matched_email_idx:
                continue
            pool_copy = [es]
            if _find_match(pdf, pool_copy) is not None:
                email_match = es
                email_label = elabel
                matched_email_idx.add(idx)
                break

        if email_match:
            used_email = False
            has_conflict = False

            for pdf_key, out_key, e_key in FIELDS:
                pdf_val   = pdf[pdf_key]
                email_val = email_match[e_key]

                if _is_empty(pdf_val) and not _is_empty(email_val):
                    # Gap fill — email provides value PDF lacks
                    record[out_key] = email_val
                    fill_log.append(
                        f"[{display_title}] {out_key}: PDF was empty/TBD → "
                        f"filled from {email_label}: '{email_val}'"
                    )
                    used_email = True

                elif not _is_empty(pdf_val) and not _is_empty(email_val):
                    norm_p = _norm_title(pdf_val)
                    norm_e = _norm_title(email_val)
                    if norm_p != norm_e:
                        # Conflict — PDF wins (record[out_key] already set to pdf value)
                        conflict_log.append(
                            f"[{display_title}] {out_key}: "
                            f"PDF='{pdf_val}' vs {email_label}='{email_val}' → PDF wins"
                        )
                        used_email = True
                        has_conflict = True

            # Build source note
            if used_email:
                note = f"PDF + {email_label}"
                if has_conflict:
                    # Find which fields had conflicts for this title
                    conflict_fields = []
                    for line in conflict_log:
                        if f"[{display_title}]" in line:
                            m = re.search(r"\] (.+?):", line)
                            if m:
                                conflict_fields.append(m.group(1))
                    if conflict_fields:
                        note += f" (conflict resolved: {', '.join(conflict_fields)})"
                record["_source_note"] = note
            else:
                record["_source_note"] = "PDF only"

        # Mark missing writers
        if _is_empty(record.get("Writers", "")):
            record["Writers"] = "MISSING"
            fill_log.append(
                f"[{display_title}] Writers: no value found in any source"
            )

        clean_records.append(record)

    # Append email-only records (unmatched to any PDF record)
    for idx, (es, elabel) in enumerate(email_pool):
        if idx not in matched_email_idx:
            title = es.get("song_title") or "Unknown"
            clean_records.append({
                "Song Title":       es["song_title"],
                "Writers":          es["writers"] if not _is_empty(es.get("writers", "")) else "MISSING",
                "Recording Artist": es["primary_artist"],
                "ISRC":             es["isrc"],
                "Release Date":     es["release_date"],
                "_source_note":     f"{elabel} only",
            })
            fill_log.append(
                f"[{title}] Source: {elabel} only — no matching PDF record found"
            )

    return clean_records, conflict_log, fill_log


# ── Stage 5 — Validation ─────────────────────────────────────────────────────

_ISRC_RE  = re.compile(r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$")
_DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_records(clean_records: list[dict]) -> list[dict]:
    log = []
    for i, rec in enumerate(clean_records, 1):
        issues = []
        if not rec.get("Song Title"):
            issues.append("Missing song title")
        if rec.get("Writers") == "MISSING":
            issues.append("Writers missing")
        if not rec.get("Recording Artist"):
            issues.append("Missing recording artist")
        isrc = rec.get("ISRC", "")
        if isrc and not _ISRC_RE.match(isrc):
            issues.append(f"Malformed ISRC: {isrc}")
        date = rec.get("Release Date", "")
        if date and not _DATE_RE.match(date):
            issues.append(f"Non-ISO date: {date}")
        log.append({
            "row":        i,
            "song_title": rec.get("Song Title", ""),
            "source":     rec.get("_source_note", ""),
            "issues":     issues,
        })
    return log


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_file,
    email_a_text: str,
    email_b_text: str,
    client,
    model: str,
    progress_callback=None,
) -> dict:
    TOTAL = 5
    errors: list[str] = []

    def step(n, msg):
        if progress_callback:
            progress_callback(n, TOTAL, msg)

    result = dict(
        pdf_songs=[], email_a_songs=[], email_b_songs=[],
        clean_records=[], validation_log=[],
        conflict_log=[], fill_log=[],
        pdf_raw_text="", errors=errors,
    )

    # Step 1: Extract PDF
    step(1, "Extracting PDF text with pdfplumber…")
    try:
        _, sched_text = extract_pdf_text(pdf_file)
        result["pdf_raw_text"] = sched_text
        if not sched_text:
            errors.append("Could not locate Schedule A page in PDF.")
            return result
    except Exception as e:
        errors.append(f"PDF extraction error: {e}")
        return result

    # Step 2: LLM parse PDF
    step(2, "AI parsing Schedule A table from PDF…")
    try:
        raw_pdf = llm_parse_pdf(sched_text, client, model)
        pdf_songs = [_normalise_song(s) for s in raw_pdf]
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(str(e))
        return result

    # Step 3: LLM parse emails
    step(3, "AI parsing Email A and Email B…")
    email_a_songs, email_b_songs = [], []
    try:
        email_a_songs = [_normalise_song(s) for s in llm_parse_email(email_a_text, client, model, "Email A")]
        result["email_a_songs"] = email_a_songs
    except Exception as e:
        errors.append(f"Email A: {e}")

    try:
        email_b_songs = [_normalise_song(s) for s in llm_parse_email(email_b_text, client, model, "Email B")]
        result["email_b_songs"] = email_b_songs
    except Exception as e:
        errors.append(f"Email B: {e}")

    # Step 4: Merge (PDF wins)
    step(4, "Merging — PDF wins all conflicts, emails fill gaps only…")
    try:
        clean, conflicts, fills = merge_sources(pdf_songs, email_a_songs, email_b_songs)
        result["clean_records"]  = clean
        result["conflict_log"]   = conflicts
        result["fill_log"]       = fills
    except Exception as e:
        errors.append(f"Merge error: {e}")
        return result

    # Step 5: Validate
    step(5, "Validating output records…")
    result["validation_log"] = validate_records(result["clean_records"])

    return result
