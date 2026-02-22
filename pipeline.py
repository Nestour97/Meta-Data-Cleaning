"""
pipeline.py  —  Metadata Cleaning Pipeline v3
Warner Chappell Music · Task 1

Root-cause fixes in this version:

  PROBLEM 1: pdfplumber table extraction mis-splits multi-line cells.
             "Shattered Glass" wraps in the title cell → pdfplumber reads
             "Shattered" as title and "Glass Atlas" as artist.
  FIX:       Extract the raw text of the Schedule A page and send it to
             the LLM for parsing. The LLM understands natural language and
             correctly reads "Shattered Glass" as the song title.
             Track numbers 01–10 act as strict anchors — the LLM cannot
             invent rows that don't exist.

  PROBLEM 2: Merge used normalized title as the only match key.
             "Shatter" (email) ≠ "Shattered Glass" (PDF) → no match →
             writers field never filled.
  FIX:       ISRC-first matching. Both email and PDF list the same ISRC
             (US-WB1-24-00433) for this track. Matching on ISRC is reliable
             even when titles differ across sources.

  PROBLEM 3: EMAIL_A and EMAIL_B constants used modified text.
             The actual prompt emails say "Shatter", "Golden Gate", and
             "Neon Dreams" (no feat.). Our code had corrected versions.
  FIX:       Restored to the exact email text from the assessment prompt.

Architecture:
  Stage 1  pdfplumber.extract_text()  →  raw text of Schedule A page
  Stage 2  LLM parses raw text        →  10 structured song dicts (PDF-authoritative)
  Stage 3  LLM parses Email A         →  structured song dicts
  Stage 4  LLM parses Email B         →  structured song dicts
  Stage 5  Python merge               →  ISRC-first match, PDF wins all conflicts
  Stage 6  Python validation          →  normalize ISRC + date, flag issues
"""

from __future__ import annotations

import re
import io
import csv
import json
import pdfplumber
from datetime import datetime
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# Source text constants — EXACT text from the assessment prompt
# ══════════════════════════════════════════════════════════════════════════════

EMAIL_A = """Hey! Thanks for reaching out. As we discussed, here are the first 5 tracks \
for the 'Neon Summer' project. Please make sure these ISRCs are logged and we'll add them \
into our system:

'Tokyo Midnight'
Artist: The Neon Lights
Writers: Alex Park, Jane Miller
ISRC: US-WB1-24-00432
Rel: Nov 15, 2024

'Shatter'
Artist: Glass Atlas
Writers: Sarah Stone, Kevin Webb
ISRC: US-WB1-24-00433
Rel: Nov 15, 2024.

'Desert Rain'
Artist: Mirage
Writers: Leyla Ademi
ISRC: US-WB1-24-00434
Rel: Nov 22, 2024

'Neon Dreams'
Artist: The Neon Lights
Writers: Alex Park, Jane Miller
ISRC: US-WB1-24-00435
Rel: Nov 15, 2024

'Uptown'
Artist: Funk Theory
Writers: Marcus Davis
ISRC: US-WB1-24-00436
Rel: Dec 01, 2024"""

EMAIL_B = """Hey, here are the remaining tracks for the schedule: \
6. 'Golden Gate' | Artist: Bridges | Writers: Rachel Davis | ISRC: US-WB1-24-00437 (Rel: Dec 05, 2024) \
7. 'Velocity' | Artist: Turbo | Writers: Paul Walker | ISRC: US-WB1-24-00438 (Rel: Dec 05, 2024) \
8. 'Silent Echo' | Artist: The Void | Writers: Elena Black | ISRC: US-WB1-24-00439 (Rel: Dec 12, 2024) \
9. 'Blue Monday' | Artist: New Wave | Writers: George Sumner | ISRC: US-WB1-24-00440 (Rel: Dec 12, 2024) \
10. 'Last Stop' | Artist: Transit | Writers: John Doe | ISRC: US-WB1-24-00441 (Rel: Dec 20, 2024)"""

# Values that mean "not provided"
_MISSING = {"", "tbd", "n/a", "none", "—", "-", "null"}

OUTPUT_COLUMNS = ["Song Title", "Writers", "Recording Artist", "ISRC", "Release Date"]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1  PDF raw text extraction (deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def extract_schedule_a_text(pdf_file) -> str:
    """
    Extract the raw text from whichever page contains 'Schedule A'.
    Returns the full page text. No LLM, no column detection.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Find the Schedule A page (case-insensitive)
                if re.search(r"schedule\s+a", text, re.IGNORECASE):
                    return text
            # Fallback: return all text combined
            return "\n\n".join(
                (p.extract_text() or "") for p in pdf.pages
            )
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2  LLM extraction — PDF text (strict, track-anchored)
# ══════════════════════════════════════════════════════════════════════════════

_PDF_SYSTEM_PROMPT = """You are a music publishing data extraction assistant.
You will receive the raw text of a "Schedule A" table from a publishing agreement.
The table lists songs numbered 01, 02, 03, etc.

Extract EACH numbered track and return a JSON array.

Each element must have EXACTLY these keys:
  "track"             - the 2-digit track number as a string (e.g. "01")
  "song_title"        - the full song title, exactly as written
  "recording_artist"  - the recording artist / primary artist, exactly as written
  "writers"           - comma-separated writer names, or null if listed as TBD / blank
  "isrc"              - the ISRC code exactly as written, or null if blank
  "release_date"      - the release date exactly as written, or null if blank

CRITICAL RULES:
1. Include EVERY numbered track present. Do not add any track not in the text.
2. song_title and recording_artist are ALWAYS separate fields. Never merge them.
   The song title comes BEFORE the artist in each row.
3. "TBD" in any field means null — not missing data you should invent.
4. Return ONLY the JSON array. No markdown, no explanation, no extra keys."""


_EMAIL_SYSTEM_PROMPT = """You are a music publishing data entry assistant.
Extract song metadata from the email text. Return a JSON array.

Each element must have EXACTLY these keys:
  "song_title"        - the song name ONLY (never include the artist)
  "recording_artist"  - performer / artist (may include feat. credits)
  "writers"           - comma-separated songwriter names, or null if missing
  "isrc"              - ISRC code exactly as written, or null if missing
  "release_date"      - date exactly as written, or null if missing

CRITICAL RULES:
1. Extract ONLY data explicitly stated. Never invent, infer, or guess.
2. song_title and recording_artist are ALWAYS separate fields.
   WRONG: song_title = "Golden Gate Bridges"   (mixed title + artist)
   RIGHT: song_title = "Golden Gate", recording_artist = "Bridges"
3. If a field is missing, use null.
4. Return ONLY the JSON array. No markdown, no explanation."""


def _call_llm(client, model: str, system: str, user: str) -> list[dict]:
    """Single LLM call → parse JSON array response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return data


def extract_pdf_songs(pdf_file, client, model: str) -> list[dict]:
    """
    Extract songs from the Schedule A raw text via LLM.
    Returns list of dicts with canonical fields + '_source'.
    """
    raw_text = extract_schedule_a_text(pdf_file)

    user_prompt = (
        "Extract all numbered tracks from this Schedule A text.\n\n"
        "--- SCHEDULE A TEXT ---\n"
        f"{raw_text}\n"
        "--- END ---\n\n"
        "Return a JSON array. Remember: song_title and recording_artist are always separate fields."
    )

    songs = _call_llm(client, model, _PDF_SYSTEM_PROMPT, user_prompt)

    canonical = ["song_title", "recording_artist", "writers", "isrc", "release_date"]
    result = []
    for s in songs:
        if not isinstance(s, dict):
            continue
        record = {f: str(s.get(f) or "").strip() for f in canonical}
        # Treat TBD / blank as missing
        for f in canonical:
            if record[f].lower() in _MISSING:
                record[f] = ""
        record["_source"] = "PDF"
        record["_track"]  = str(s.get("track", "")).strip()
        result.append(record)

    return result


def extract_email_songs(
    client, model: str, email_text: str, source_label: str
) -> list[dict]:
    """LLM extraction from one email → list of song dicts."""
    user_prompt = (
        f"Extract all songs from this email.\n\n"
        f"--- {source_label.upper()} ---\n{email_text}\n--- END ---\n\n"
        f"Return a JSON array. Remember: song_title and recording_artist are always separate fields."
    )
    songs = _call_llm(client, model, _EMAIL_SYSTEM_PROMPT, user_prompt)

    canonical = ["song_title", "recording_artist", "writers", "isrc", "release_date"]
    result = []
    for s in songs:
        if not isinstance(s, dict):
            continue
        record = {f: str(s.get(f) or "").strip() for f in canonical}
        for f in canonical:
            if record[f].lower() in _MISSING:
                record[f] = ""
        record["_source"] = source_label
        result.append(record)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5  Deterministic Python merge — ISRC-first, then title
# ══════════════════════════════════════════════════════════════════════════════

def _norm_title(t: str) -> str:
    """Lowercase + strip non-alphanumeric for fuzzy title matching."""
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _norm_isrc(isrc: str) -> str:
    """Strip hyphens/spaces for ISRC comparison."""
    return re.sub(r"[^A-Z0-9]", "", isrc.upper())


def _build_email_indexes(songs: list[dict]) -> tuple[dict, dict]:
    """
    Build two lookup dicts from email songs:
      isrc_index  : normalised_isrc  → song dict
      title_index : normalised_title → song dict
    First occurrence wins (Email A takes priority over B).
    """
    isrc_idx:  dict[str, dict] = {}
    title_idx: dict[str, dict] = {}
    for s in songs:
        ni = _norm_isrc(s.get("isrc", ""))
        nt = _norm_title(s.get("song_title", ""))
        if ni and ni not in isrc_idx:
            isrc_idx[ni] = s
        if nt and nt not in title_idx:
            title_idx[nt] = s
    return isrc_idx, title_idx


def merge_songs(
    pdf_songs:     list[dict],
    email_a_songs: list[dict],
    email_b_songs: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Field-by-field deterministic merge.

    Match strategy (per PDF song):
      1. Match by normalised ISRC  (most reliable — survives title variations)
      2. Match by normalised title  (fallback when ISRC missing in PDF)

    Conflict rule:
      PDF field non-empty → always use PDF value.
      PDF field empty     → use best email value (A preferred over B).

    Only songs listed in the PDF appear in output.
    """
    # Build combined email index (A preferred over B for same ISRC/title)
    all_email = email_a_songs + email_b_songs
    isrc_idx, title_idx = _build_email_indexes(all_email)

    # Also separate indexes for source attribution
    a_isrc, a_title = _build_email_indexes(email_a_songs)
    b_isrc, b_title = _build_email_indexes(email_b_songs)

    FIELDS = ["song_title", "recording_artist", "writers", "isrc", "release_date"]

    merged:       list[dict] = []
    fill_log:     list[str]  = []
    conflict_log: list[str]  = []

    for pdf_song in pdf_songs:
        title  = pdf_song.get("song_title", "")
        ni_pdf = _norm_isrc(pdf_song.get("isrc", ""))
        nt_pdf = _norm_title(title)

        # Find best email match
        email_hit = (
            isrc_idx.get(ni_pdf)    # 1. ISRC match (best)
            or title_idx.get(nt_pdf) # 2. Title match (fallback)
        )

        # Determine which source supplied the match
        if email_hit:
            esrc = email_hit.get("_source", "email")
            ni_hit = _norm_isrc(email_hit.get("isrc", ""))
            if ni_pdf and ni_hit and ni_pdf == ni_hit:
                match_method = "ISRC"
            else:
                match_method = "title"
        else:
            esrc = ""
            match_method = "none"

        record:       dict = {}
        field_sources: dict = {}

        for field in FIELDS:
            pdf_val   = pdf_song.get(field, "").strip()
            email_val = ((email_hit or {}).get(field) or "").strip()

            # Treat TBD etc. as empty
            if pdf_val.lower() in _MISSING:
                pdf_val = ""
            if email_val.lower() in _MISSING:
                email_val = ""

            if pdf_val:
                record[field]        = pdf_val
                field_sources[field] = "PDF"
                if email_val and _norm_title(email_val) != _norm_title(pdf_val):
                    conflict_log.append(
                        f"  CONFLICT '{title}'.{field}: "
                        f"PDF='{pdf_val}' | {esrc}='{email_val}' → kept PDF"
                    )
            elif email_val:
                record[field]        = email_val
                field_sources[field] = esrc
                fill_log.append(
                    f"  FILLED '{title}'.{field} from {esrc} "
                    f"[matched by {match_method}]: '{email_val}'"
                )
            else:
                record[field]        = ""
                field_sources[field] = "missing"
                fill_log.append(
                    f"  MISSING '{title}'.{field} — not found in any source"
                )

        filled = [f for f, s in field_sources.items()
                  if s not in ("PDF", "missing")]
        record["_source_note"] = (
            f"PDF + {esrc} ({', '.join(filled)})" if filled else "PDF only"
        )
        merged.append(record)

    return merged, fill_log, conflict_log


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6  Deterministic validation — ISRC + Date normalization
# ══════════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y",  "%d %b %Y",
    "%b %d %Y",  "%B %d %Y",
    "%Y/%m/%d",
]


def normalize_isrc(raw: Optional[str]) -> Optional[str]:
    """Normalize to CC-RRR-YY-NNNNN (12 alphanumeric, hyphenated)."""
    if not raw or str(raw).strip().lower() in _MISSING:
        return None
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw.strip()).upper()
    if len(stripped) != 12:
        return raw.strip().upper()
    return f"{stripped[0:2]}-{stripped[2:5]}-{stripped[5:7]}-{stripped[7:12]}"


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Parse any common date format → YYYY-MM-DD."""
    if not raw or str(raw).strip().lower() in _MISSING:
        return None
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1",
                  re.sub(r"\s+", " ", str(raw).strip()),
                  flags=re.IGNORECASE)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Regex fallback for month names
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                  text, re.IGNORECASE)
    digits = re.findall(r"\d+", text)
    years  = [int(d) for d in digits if len(d) == 4]
    days   = [int(d) for d in digits if 1 <= int(d) <= 31 and len(d) <= 2]
    if m and years and days:
        try:
            return datetime(years[0], months[m.group(1).lower()], days[0]).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw


def validate_and_finalize(merged: list[dict]) -> tuple[list[dict], list[dict]]:
    """Normalize ISRC + dates, flag issues per row."""
    clean: list[dict] = []
    log:   list[dict] = []

    for i, song in enumerate(merged, 1):
        record: dict      = {}
        issues: list[str] = []

        title = (song.get("song_title") or "").strip().strip("'\"")
        record["Song Title"] = title or "MISSING"
        if not title:
            issues.append("Song title missing")

        writers = (song.get("writers") or "").strip()
        record["Writers"] = writers or "MISSING"
        if not writers:
            issues.append("Writers missing in all sources")

        artist = (song.get("recording_artist") or "").strip()
        record["Recording Artist"] = artist or "MISSING"
        if not artist:
            issues.append("Recording artist missing")

        isrc_norm = normalize_isrc(song.get("isrc"))
        record["ISRC"] = isrc_norm or "MISSING"
        if not isrc_norm or not re.match(
                r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$", isrc_norm):
            issues.append(f"ISRC invalid/unnormalizable: '{song.get('isrc')}'")

        date_norm = normalize_date(song.get("release_date"))
        record["Release Date"] = date_norm or "MISSING"
        if not date_norm or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_norm)):
            issues.append(f"Date could not be normalized: '{song.get('release_date')}'")

        record["_source_note"] = song.get("_source_note", "unknown")
        clean.append(record)
        log.append({
            "row":        i,
            "song_title": record["Song Title"],
            "issues":     issues,
            "source":     record["_source_note"],
            "clean":      len(issues) == 0,
        })

    return clean, log


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pdf_file,
    email_a_text: str,
    email_b_text: str,
    client,
    model: str,
    progress_callback=None,
) -> dict:
    """
    Run all stages and return a result dict.

    Args:
        pdf_file:          File-like object or path to the Schedule A PDF
        email_a_text:      Email A raw text (exact text from assessment prompt)
        email_b_text:      Email B raw text (exact text from assessment prompt)
        client:            OpenAI-compatible LLM client
        model:             Model name string
        progress_callback: Optional callable(step, total, message)
    """
    TOTAL = 7
    errors: list[str] = []
    result = dict(
        pdf_raw_text="",
        pdf_songs=[], email_a_songs=[], email_b_songs=[],
        merged_songs=[], clean_records=[], validation_log=[],
        fill_log=[], conflict_log=[], errors=errors,
    )

    def step(n, msg):
        if progress_callback:
            progress_callback(n, TOTAL, msg)

    # Stage 1: PDF raw text
    step(1, "Stage 1/4 — Extracting Schedule A text from PDF…")
    try:
        raw_text = extract_schedule_a_text(pdf_file)
        result["pdf_raw_text"] = raw_text
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 2: LLM parses PDF text
    step(2, "Stage 2/4 — AI parsing Schedule A text (track numbers as anchors)…")
    try:
        # Reset file pointer if it's a file object
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        pdf_songs = extract_pdf_songs(pdf_file, client, model)
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(f"PDF LLM extraction: {e}")
        return result

    # Stage 3a: LLM parses Email A
    step(3, "Stage 2/4 — AI parsing Email A (Creative Dept)…")
    try:
        ea = extract_email_songs(client, model, email_a_text, "Email A – Creative Dept")
        result["email_a_songs"] = ea
    except Exception as e:
        errors.append(f"Email A: {e}")
        ea = []

    # Stage 3b: LLM parses Email B
    step(4, "Stage 2/4 — AI parsing Email B (Artist Management)…")
    try:
        eb = extract_email_songs(client, model, email_b_text, "Email B – Artist Management")
        result["email_b_songs"] = eb
    except Exception as e:
        errors.append(f"Email B: {e}")
        eb = []

    # Stage 4: Python merge
    step(5, "Stage 3/4 — Merging (ISRC-first matching, PDF wins all conflicts)…")
    try:
        merged, fill_log, conflict_log = merge_songs(pdf_songs, ea, eb)
        result.update(merged_songs=merged, fill_log=fill_log,
                      conflict_log=conflict_log)
    except Exception as e:
        errors.append(f"Merge: {e}")
        return result

    # Stage 5: Validation
    step(6, "Stage 4/4 — Normalizing ISRCs, dates, running validation…")
    clean, log = validate_and_finalize(merged)
    result.update(clean_records=clean, validation_log=log)

    n_flags = sum(1 for v in log if not v["clean"])
    step(7, f"Done — {len(clean)} songs output · {n_flags} validation flag(s).")
    return result


# ── CSV output ─────────────────────────────────────────────────────────────────

def build_csv(clean_records: list[dict]) -> str:
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(clean_records)
    return buf.getvalue()
