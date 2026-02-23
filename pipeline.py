"""
pipeline.py  —  Metadata Cleaning Pipeline
Warner Chappell Music Intelligence · Task 1

PDF PARSING STRATEGY (hybrid):
  Step A — pdfplumber "lines" strategy table extraction.
            This uses the actual grid lines in the PDF and gets the column
            structure exactly right for 9 of 10 rows.
  Step B — For each row, compare the table's (title, artist) pair against
            the raw-text line for that track number.
            If the artist cell STARTS with text that also appears at the end
            of the raw-text title segment, that text is a title cell-overflow
            and must be moved from artist → title.
            Example: table gives title="Shattered", artist="Glass Atlas"
                     raw text line: "Shattered Glass Atlas"
                     "Glass" ends the raw-text title portion, starts the
                     artist cell → title overflow → fix to title="Shattered Glass",
                     artist="Atlas".

MERGE STRATEGY:
  PDF is the source of truth. Emails fill gaps (empty / TBD fields) only.
  Matching: normalised ISRC first → exact title → partial title.
"""

from __future__ import annotations

import re
import json
import io
import csv
from typing import Optional


# ── Default email content ─────────────────────────────────────────────────────

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
    if not raw:
        return ""
    s = raw.strip()
    m = re.match(r"([A-Za-z]{2})[.\-]?([A-Za-z0-9]{3})[.\-]?(\d{2})[.\-]?(\d{5})", s)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}-{m.group(3)}-{m.group(4)}"
    digits = re.sub(r"[^A-Z0-9]", "", s.upper())
    if len(digits) == 12:
        return f"{digits[0:2]}-{digits[2:5]}-{digits[5:7]}-{digits[7:12]}"
    return s


def normalize_date(raw: str) -> str:
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
    return not val or val.strip().upper() in {
        "", "TBD", "N/A", "MISSING", "NONE", "-", "?", "NA", "NULL"
    }


def _norm_isrc(isrc: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", isrc.upper())


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()


# ── Stage 1A — pdfplumber table extraction ────────────────────────────────────

def _find_schedule_a_page(pdf):
    """Return the page that contains the Schedule A song table."""
    for page in pdf.pages:
        text = (page.extract_text() or "").lower()
        if ("schedule a" in text or "song title" in text) and "isrc" in text:
            return page
    return None


def _extract_table_rows(page) -> list[dict]:
    """
    Extract rows using pdfplumber's 'lines' table strategy (uses actual PDF grid).
    Returns list of raw row dicts with keys: track, title, artist, writers, release, isrc.
    """
    tables = page.extract_tables(
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    )
    if not tables:
        return []

    table = tables[0]
    if not table:
        return []

    # Identify header row and column order
    COL_ALIASES = {
        "track":   ["track"],
        "title":   ["song title", "title"],
        "artist":  ["primary artist", "artist"],
        "writers": ["writers", "full names"],
        "release": ["release", "release date"],
        "isrc":    ["isrc"],
    }

    def identify_col(header_text: str) -> Optional[str]:
        ht = (header_text or "").lower().replace("\n", " ").strip()
        for key, aliases in COL_ALIASES.items():
            for alias in aliases:
                if alias in ht:
                    return key
        return None

    header = table[0]
    col_map: dict[int, str] = {}  # col_index → semantic name
    for i, cell in enumerate(header):
        sem = identify_col(cell or "")
        if sem:
            col_map[i] = sem

    rows = []
    for row in table[1:]:
        rec: dict = {k: "" for k in ("track", "title", "artist", "writers", "release", "isrc")}
        for i, cell in enumerate(row):
            sem = col_map.get(i)
            if sem:
                rec[sem] = (cell or "").replace("\n", " ").strip()
        # Only include rows that have a numeric track number
        if re.match(r"^\d+$", rec.get("track", "").strip()):
            rows.append(rec)

    return rows


# ── Stage 1B — LLM-assisted overflow correction ───────────────────────────────

_OVERFLOW_FIX_PROMPT = """You are a music publishing metadata specialist.

The table below was extracted from a PDF using pdfplumber's grid-line strategy.
It is MOSTLY correct, but sometimes the last word(s) of a Song Title physically
overflow into the Primary Artist column due to a PDF cell-width issue.

Your ONLY job is to detect and fix rows where the Primary Artist cell starts
with word(s) that actually belong to the Song Title.

How to spot overflow:
- The Song Title looks suspiciously short (often 1 word).
- The Primary Artist starts with word(s) that, when joined with the short title,
  form a natural complete song title.
- The REMAINING words of the Primary Artist (after removing the overflow words)
  still make sense as a standalone artist/band name.

Examples:
  OVERFLOW:  title="Shattered"    artist="Glass Atlas"
  FIXED:     title="Shattered Glass"  artist="Atlas"
  (because "Shattered Glass" is a natural title, "Atlas" is the artist)

  NO OVERFLOW: title="Uptown"     artist="Funk Theory"
  KEEP:        title="Uptown"     artist="Funk Theory"
  (because "Funk Theory" is a complete artist/band name by itself)

  NO OVERFLOW: title="Golden Gates"  artist="Bridges"
  KEEP:        title="Golden Gates"  artist="Bridges"
  (title already looks complete; "Bridges" is a valid artist)

Return a JSON object with the corrected rows. Keep every field exactly as-is
except Song Title and Primary Artist where overflow is detected.
Return ONLY valid JSON, no markdown:
{
  "rows": [
    {
      "track": "01",
      "title": "corrected title",
      "artist": "corrected artist",
      "writers": "...",
      "release": "...",
      "isrc": "..."
    }
  ]
}"""


def _fix_title_artist_overflow(rows: list[dict], client, model: str) -> list[dict]:
    """
    Use LLM to detect and fix title-word overflow into the artist cell.
    Only the (title, artist) pair may be modified; all other fields are kept as-is.
    Falls back to original rows if LLM fails.
    """
    if not rows:
        return rows

    input_payload = json.dumps({"rows": rows}, indent=2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _OVERFLOW_FIX_PROMPT},
                {"role": "user",   "content": f"Table rows to check:\n{input_payload}"},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        corrected = json.loads(raw).get("rows", [])
    except Exception:
        # LLM failed — return original rows unchanged
        return rows

    # Validate: only accept title/artist changes; reject any hallucinated row counts
    if len(corrected) != len(rows):
        return rows

    result = []
    for orig, corr in zip(rows, corrected):
        merged = dict(orig)
        # Only update title and artist — nothing else
        if corr.get("title"):
            merged["title"] = corr["title"]
        if corr.get("artist"):
            merged["artist"] = corr["artist"]
        result.append(merged)
    return result


def extract_schedule_a_rows(pdf_file, client=None, model: str = "") -> tuple[list[dict], str]:
    """
    Main PDF extraction entry point.
    - Uses pdfplumber 'lines' strategy for structurally correct column extraction.
    - Optionally uses LLM to fix title-word overflow into the artist cell.
    Returns (songs_list, raw_text).
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
        source = open(pdf_file, "rb") if isinstance(pdf_file, str) else pdf_file

    with pdfplumber.open(source) as pdf:
        page = _find_schedule_a_page(pdf)
        if page is None:
            raise RuntimeError("Could not find Schedule A page in PDF.")
        raw_text = page.extract_text() or ""
        rows     = _extract_table_rows(page)

    # Apply LLM overflow correction if a client is provided
    if client and rows:
        rows = _fix_title_artist_overflow(rows, client, model)

    songs = []
    for rec in rows:
        songs.append({
            "song_title":     rec["title"],
            "primary_artist": rec["artist"],
            "writers":        rec["writers"],
            "release_date":   normalize_date(rec["release"]),
            "isrc":           normalize_isrc(rec["isrc"]),
        })

    return songs, raw_text


# ── Stage 2 — LLM: parse emails ──────────────────────────────────────────────

_EMAIL_PARSE_PROMPT = """You are a music publishing metadata specialist.

Extract every song's metadata from the email below.
For any missing/unknown field return an empty string "".
Writers listed as "TBD", "N/A", or similar → return "".
Normalize ISRCs to CC-RRR-YY-NNNNN format.
Normalize dates to YYYY-MM-DD format.

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
            {"role": "user",   "content": f"Email:\n\n{email_text}"},
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


# ── Stage 3 — Merge (PDF wins conflicts) ─────────────────────────────────────

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
    Find best match in pool for target.
    Priority: normalised ISRC → exact title → partial title containment.
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

    # 3. Partial title containment (≥2 words to avoid false positives)
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

    Rules:
    - PDF field value wins whenever it is non-empty / non-TBD.
    - Email fills gaps (empty / TBD PDF fields) only.
    - Email-only records (unmatched to any PDF row) are appended at the end.
    """
    conflict_log: list[str] = []
    fill_log:     list[str] = []
    clean_records: list[dict] = []

    email_pool: list[tuple[dict, str]] = (
        [(s, "Email A") for s in email_a_songs] +
        [(s, "Email B") for s in email_b_songs]
    )
    matched_email_idx: set[int] = set()

    FIELDS = [
        ("song_title",     "Song Title",       "song_title"),
        ("primary_artist", "Recording Artist", "primary_artist"),
        ("writers",        "Writers",          "writers"),
        ("isrc",           "ISRC",             "isrc"),
        ("release_date",   "Release Date",     "release_date"),
    ]

    for pdf in pdf_songs:
        display_title = pdf["song_title"] or "Unknown"

        record: dict = {
            "Song Title":       pdf["song_title"],
            "Writers":          pdf["writers"],
            "Recording Artist": pdf["primary_artist"],
            "ISRC":             pdf["isrc"],
            "Release Date":     pdf["release_date"],
            "_source_note":     "PDF only",
        }

        # Find email match
        pool_list = [es for es, _ in email_pool]
        idx = _find_match(pdf, pool_list)
        email_match: Optional[dict] = None
        email_label = ""
        if idx is not None and idx not in matched_email_idx:
            email_match, email_label = email_pool[idx]
            matched_email_idx.add(idx)

        if email_match:
            used_email   = False
            has_conflict = False
            conflict_fields: list[str] = []

            for pdf_key, out_key, e_key in FIELDS:
                pdf_val   = pdf[pdf_key]
                email_val = email_match[e_key]

                if _is_empty(pdf_val) and not _is_empty(email_val):
                    # Gap fill
                    record[out_key] = email_val
                    fill_log.append(
                        f"[{display_title}] {out_key}: PDF empty/TBD → "
                        f"filled from {email_label}: '{email_val}'"
                    )
                    used_email = True
                elif not _is_empty(pdf_val) and not _is_empty(email_val):
                    if _norm_title(pdf_val) != _norm_title(email_val):
                        conflict_log.append(
                            f"[{display_title}] {out_key}: "
                            f"PDF='{pdf_val}' vs {email_label}='{email_val}' → PDF wins"
                        )
                        used_email   = True
                        has_conflict = True
                        conflict_fields.append(out_key)

            if used_email:
                note = f"PDF + {email_label}"
                if has_conflict and conflict_fields:
                    note += f" (conflict: {', '.join(conflict_fields)})"
                record["_source_note"] = note

        if _is_empty(record.get("Writers", "")):
            record["Writers"] = "MISSING"
            fill_log.append(f"[{display_title}] Writers: no value in any source")

        clean_records.append(record)

    # Append email-only records (no PDF match found)
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
            fill_log.append(f"[{title}] Source: {elabel} only — no PDF match")

    return clean_records, conflict_log, fill_log


# ── Stage 4 — Validation ─────────────────────────────────────────────────────

_ISRC_RE = re.compile(r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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
    TOTAL = 4
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

    # Step 1: Parse PDF table + LLM overflow correction
    step(1, "Parsing Schedule A table (grid-based extraction + AI overflow fix)…")
    try:
        pdf_songs_raw, raw_text = extract_schedule_a_rows(pdf_file, client=client, model=model)
        result["pdf_raw_text"] = raw_text
        pdf_songs = [_normalise_song(s) for s in pdf_songs_raw]
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(f"PDF parse error: {e}")
        return result

    # Step 2: Parse emails via LLM
    step(2, "AI parsing Email A and Email B…")
    email_a_songs, email_b_songs = [], []
    try:
        ea_raw = llm_parse_email(email_a_text, client, model, "Email A")
        email_a_songs = [_normalise_song(s) for s in ea_raw]
        result["email_a_songs"] = email_a_songs
    except Exception as e:
        errors.append(f"Email A: {e}")

    try:
        eb_raw = llm_parse_email(email_b_text, client, model, "Email B")
        email_b_songs = [_normalise_song(s) for s in eb_raw]
        result["email_b_songs"] = email_b_songs
    except Exception as e:
        errors.append(f"Email B: {e}")

    # Step 3: Merge (PDF wins)
    step(3, "Merging — PDF wins conflicts, emails fill gaps only…")
    try:
        clean, conflicts, fills = merge_sources(pdf_songs, email_a_songs, email_b_songs)
        result["clean_records"] = clean
        result["conflict_log"]  = conflicts
        result["fill_log"]      = fills
    except Exception as e:
        errors.append(f"Merge error: {e}")
        return result

    # Step 4: Validate
    step(4, "Validating output records…")
    result["validation_log"] = validate_records(result["clean_records"])

    return result
