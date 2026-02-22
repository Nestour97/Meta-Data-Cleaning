"""
pipeline.py  —  Metadata Cleaning Pipeline v2
Warner Chappell Music · Task 1

Architecture (no LLM for PDF — eliminates hallucination and field-merging):

  Stage 1  PDF structural extraction
           pdfplumber.extract_tables() reads the Schedule A table directly.
           Zero LLM calls. Each row maps to exact column headers.
           Fields are never merged — they come from separate table cells.

  Stage 2  Email LLM extraction (schema-locked, anti-hallucination prompt)
           One LLM call per email with a tight JSON schema and explicit
           rules against field-merging and data invention.

  Stage 3  Deterministic Python merge
           For each PDF song, field-by-field:
             PDF field non-empty / non-TBD  ->  keep PDF value (always)
             PDF field empty / TBD          ->  fill from email match
           Only songs from the PDF appear in output — impossible to hallucinate extras.

  Stage 4  Deterministic validation
           Normalize ISRC  -> CC-RRR-YY-NNNNN
           Normalize dates -> YYYY-MM-DD
           Flag per-row issues in a validation log.
"""

from __future__ import annotations

import re
import io
import csv
import json
import pdfplumber
from datetime import datetime
from typing import Optional

# ── Email text constants ───────────────────────────────────────────────────────

EMAIL_A = """Hey! Thanks for reaching out. As we discussed, here are the first 5 tracks \
for the 'Neon Summer' project. Please make sure these ISRCs are logged and we'll add them \
into our system:

'Tokyo Midnight'
Artist: The Neon Lights
Writers: Alex Park, Jane Miller
ISRC: US-WB1-24-00432
Rel: Nov 15, 2024

'Shattered Glass'
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
Artist: The Neon Lights (feat. DJ Flux)
Writers: Alex Park, Jane Miller, Thomas Flux
ISRC: US-WB1-24-00435
Rel: Nov 15, 2024

'Uptown'
Artist: Funk Theory
Writers: Marcus Davis
ISRC: US-WB1-24-00436
Rel: Dec 01, 2024"""

EMAIL_B = """Hey, here are the remaining tracks for the schedule: \
6. 'Golden Gates' | Artist: Bridges | Writers: Rachel Davis | ISRC: US-WB1-24-00437 (Rel: Dec 05, 2024) \
7. 'Velocity' | Artist: Turbo | Writers: Paul Walker | ISRC: US-WB1-24-00438 (Rel: Dec 05, 2024) \
8. 'Silent Echo' | Artist: The Void | Writers: Elena Black | ISRC: US-WB1-24-00439 (Rel: Dec 12, 2024) \
9. 'Blue Monday' | Artist: New Wave | Writers: George Sumner | ISRC: US-WB1-24-00440 (Rel: Dec 12, 2024) \
10. 'Last Stop' | Artist: Transit | Writers: John Doe | ISRC: US-WB1-24-00441 (Rel: Dec 20, 2024)"""

# Values that mean "not provided" in the PDF table
_MISSING_SENTINELS = {"", "tbd", "n/a", "none", "—", "-", "null"}

OUTPUT_COLUMNS = ["Song Title", "Writers", "Recording Artist", "ISRC", "Release Date"]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1  PDF structural extraction (zero LLM)
# ══════════════════════════════════════════════════════════════════════════════

_PDF_HEADER_MAP = {
    "song title":        "song_title",
    "title":             "song_title",
    "recording artist":  "recording_artist",
    "artist":            "recording_artist",
    "writer(s)":         "writers",
    "writers":           "writers",
    "writer":            "writers",
    "isrc":              "isrc",
    "release date":      "release_date",
    "rel date":          "release_date",
    "date":              "release_date",
}


def _clean_cell(value) -> str:
    """Strip newlines, tabs, and extra whitespace from a table cell."""
    if value is None:
        return ""
    return re.sub(r"  +", " ", re.sub(r"[\n\r\t]+", " ", str(value))).strip()


def _is_missing(value: str) -> bool:
    return value.lower().strip() in _MISSING_SENTINELS


def extract_pdf_songs(pdf_file) -> list[dict]:
    """
    Read the Schedule A table directly from the PDF.
    Returns a list of song dicts — no LLM, no ambiguity.
    """
    songs: list[dict] = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    if not table:
                        continue

                    col_map: dict[int, str] = {}

                    for row_idx, row in enumerate(table):
                        clean = [_clean_cell(c) for c in row]

                        # First row: build column index map
                        if row_idx == 0:
                            for ci, cell in enumerate(clean):
                                key = cell.lower()
                                if key in _PDF_HEADER_MAP:
                                    col_map[ci] = _PDF_HEADER_MAP[key]
                            continue

                        # Skip entirely empty rows
                        if all(not v for v in clean):
                            continue

                        song: dict = {f: "" for f in
                                      ["song_title", "recording_artist", "writers",
                                       "isrc", "release_date"]}

                        for ci, field in col_map.items():
                            if ci < len(clean):
                                v = clean[ci]
                                if not _is_missing(v):
                                    song[field] = v

                        if song["song_title"]:
                            song["_source"] = "PDF"
                            songs.append(song)

    except Exception as e:
        raise RuntimeError(f"PDF table extraction failed: {e}")

    if not songs:
        raise RuntimeError(
            "No songs found in the PDF table. Ensure the PDF contains a "
            "Schedule A table with standard column headers."
        )

    return songs


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2  Email LLM extraction (schema-locked, anti-hallucination)
# ══════════════════════════════════════════════════════════════════════════════

_EMAIL_SYSTEM_PROMPT = """You are a music publishing data entry assistant.
Extract song metadata from the email text. Return a JSON array.

Each element MUST have EXACTLY these five keys:
  "song_title"        - the song name ONLY (never mix in the artist name)
  "recording_artist"  - performer / recording artist (may include feat. credits)
  "writers"           - comma-separated songwriter names
  "isrc"              - ISRC code exactly as written in the email
  "release_date"      - date exactly as written in the email

MANDATORY RULES:
1. Extract ONLY data explicitly present in the email. Never invent or infer.
2. song_title and recording_artist are ALWAYS separate fields.
   WRONG: song_title = "Golden Gate Bridges"   (title + artist merged)
   RIGHT: song_title = "Golden Gates",  recording_artist = "Bridges"
3. writers = only the SONGWRITERS listed as writing the song.
   Do NOT put the recording artist in the writers field unless the email
   explicitly says they wrote the song.
4. If a field is absent from the email, use null — never fill it in.
5. Return ONLY the JSON array. No markdown. No extra text. No extra keys."""


def extract_email_songs(
    client, model: str, email_text: str, source_label: str
) -> list[dict]:
    """LLM extraction from one email — returns list of song dicts."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EMAIL_SYSTEM_PROMPT},
                {"role": "user",   "content":
                 f"Extract all songs from this email.\n\n"
                 f"--- {source_label.upper()} ---\n{email_text}\n--- END ---\n\n"
                 f"Remember: song_title and recording_artist are always separate fields."},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed ({source_label}): {e}")

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

    try:
        songs = json.loads(cleaned)
        if not isinstance(songs, list):
            raise ValueError("Expected a JSON array")
    except Exception as e:
        raise RuntimeError(
            f"JSON parse error ({source_label}): {e}\nRaw: {raw[:400]}"
        )

    fields = ["song_title", "recording_artist", "writers", "isrc", "release_date"]
    result = []
    for s in songs:
        if not isinstance(s, dict):
            continue
        record = {f: str(s.get(f) or "").strip() for f in fields}
        record["_source"] = source_label
        result.append(record)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3  Deterministic Python merge (zero LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _title_key(title: str) -> str:
    """Lowercase + strip non-alphanumerics for fuzzy title matching."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _build_index(songs: list[dict]) -> dict[str, dict]:
    """Build normalized_title -> song dict; first occurrence wins."""
    idx: dict[str, dict] = {}
    for s in songs:
        k = _title_key(s.get("song_title", ""))
        if k and k not in idx:
            idx[k] = s
    return idx


def merge_songs(
    pdf_songs:     list[dict],
    email_a_songs: list[dict],
    email_b_songs: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Field-by-field deterministic merge.

    Rules:
    - Only songs listed in the PDF appear in the output (no extras possible).
    - PDF field non-empty  ->  use PDF value unconditionally (source of truth).
    - PDF field empty      ->  use Email A value if available, else Email B.
    - When PDF and email differ -> PDF always wins (logged in conflict_log).

    Returns: (merged_songs, fill_log, conflict_log)
    """
    a_idx = _build_index(email_a_songs)
    b_idx = _build_index(email_b_songs)

    merged: list[dict] = []
    fill_log: list[str] = []
    conflict_log: list[str] = []

    FIELDS = ["song_title", "recording_artist", "writers", "isrc", "release_date"]

    for pdf_song in pdf_songs:
        title = pdf_song.get("song_title", "")
        key   = _title_key(title)

        # Find email match: prefer A, fall back to B
        email_hit  = a_idx.get(key) or b_idx.get(key)
        email_src  = (email_hit or {}).get("_source", "email") if email_hit else ""

        record: dict         = {}
        field_sources: dict  = {}

        for field in FIELDS:
            pdf_val   = pdf_song.get(field, "").strip()
            email_val = ((email_hit or {}).get(field) or "").strip()

            if pdf_val:                        # PDF has data → always wins
                record[field]        = pdf_val
                field_sources[field] = "PDF"
                if email_val and email_val != pdf_val:
                    conflict_log.append(
                        f"  CONFLICT '{title}'.{field}: "
                        f"PDF='{pdf_val}' | {email_src}='{email_val}' → kept PDF"
                    )
            elif email_val:                    # PDF missing → fill from email
                record[field]        = email_val
                field_sources[field] = email_src
                fill_log.append(
                    f"  FILLED '{title}'.{field} from {email_src}: '{email_val}'"
                )
            else:                              # Missing everywhere
                record[field]        = ""
                field_sources[field] = "missing"
                fill_log.append(
                    f"  MISSING '{title}'.{field} — not found in any source"
                )

        # Source annotation
        filled = [f for f, s in field_sources.items()
                  if s not in ("PDF", "missing")]
        record["_source_note"] = (
            f"PDF + {email_src} ({', '.join(filled)})" if filled else "PDF only"
        )
        merged.append(record)

    return merged, fill_log, conflict_log


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4  Deterministic validation (ISRC + Date)
# ══════════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y",  "%d %b %Y",
    "%b %d %Y",  "%B %d %Y",
    "%Y/%m/%d",  "%B, %d %Y",
]


def normalize_isrc(raw: Optional[str]) -> Optional[str]:
    """
    Normalize to CC-RRR-YY-NNNNN.
    Strips all non-alphanumeric chars, then re-hyphenates by fixed positions.
    """
    if not raw or str(raw).strip().lower() in _MISSING_SENTINELS:
        return None
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw.strip()).upper()
    if len(stripped) != 12:
        return raw.strip().upper()
    return f"{stripped[0:2]}-{stripped[2:5]}-{stripped[5:7]}-{stripped[7:12]}"


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Parse any common date string and return YYYY-MM-DD."""
    if not raw or str(raw).strip().lower() in _MISSING_SENTINELS:
        return None
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1",
                  re.sub(r"\s+", " ", str(raw).strip()),
                  flags=re.IGNORECASE)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Regex fallback
    months = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
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
    """Normalize ISRC + dates, flag issues, return (clean_records, validation_log)."""
    clean:   list[dict] = []
    log:     list[dict] = []

    for i, song in enumerate(merged, 1):
        record: dict     = {}
        issues: list[str] = []

        # Title
        title = (song.get("song_title") or "").strip().strip("'\"")
        record["Song Title"] = title or "MISSING"
        if not title:
            issues.append("Song title missing")

        # Writers
        writers = (song.get("writers") or "").strip()
        record["Writers"] = writers or "MISSING"
        if not writers:
            issues.append("Writers missing — not in any source")

        # Artist
        artist = (song.get("recording_artist") or "").strip()
        record["Recording Artist"] = artist or "MISSING"
        if not artist:
            issues.append("Recording artist missing")

        # ISRC
        isrc_norm = normalize_isrc(song.get("isrc"))
        record["ISRC"] = isrc_norm or "MISSING"
        if not isrc_norm or not re.match(
                r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$", isrc_norm):
            issues.append(f"ISRC invalid: '{song.get('isrc')}'")

        # Date
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
    Run all four stages and return a result dict with all intermediate data.
    """
    TOTAL = 6
    errors: list[str] = []
    result = dict(
        pdf_songs=[], email_a_songs=[], email_b_songs=[],
        merged_songs=[], clean_records=[], validation_log=[],
        fill_log=[], conflict_log=[], errors=errors,
    )

    def step(n, msg):
        if progress_callback:
            progress_callback(n, TOTAL, msg)

    # Stage 1
    step(1, "Stage 1/4 — Extracting Schedule A table from PDF (no LLM)…")
    try:
        pdf_songs = extract_pdf_songs(pdf_file)
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 2a
    step(2, "Stage 2/4 — AI extracting songs from Email A…")
    try:
        ea = extract_email_songs(client, model, email_a_text, "Email A – Creative Dept")
        result["email_a_songs"] = ea
    except Exception as e:
        errors.append(f"Email A: {e}")
        ea = []

    # Stage 2b
    step(3, "Stage 2/4 — AI extracting songs from Email B…")
    try:
        eb = extract_email_songs(client, model, email_b_text, "Email B – Artist Management")
        result["email_b_songs"] = eb
    except Exception as e:
        errors.append(f"Email B: {e}")
        eb = []

    # Stage 3
    step(4, "Stage 3/4 — Merging (PDF wins all conflicts, Python logic only)…")
    try:
        merged, fill_log, conflict_log = merge_songs(pdf_songs, ea, eb)
        result.update(merged_songs=merged, fill_log=fill_log,
                      conflict_log=conflict_log)
    except Exception as e:
        errors.append(f"Merge: {e}")
        return result

    # Stage 4
    step(5, "Stage 4/4 — Normalizing ISRCs and dates, validating…")
    clean, log = validate_and_finalize(merged)
    result.update(clean_records=clean, validation_log=log)

    n_flags = sum(1 for v in log if not v["clean"])
    step(6, f"Done — {len(clean)} songs · {n_flags} validation flag(s).")
    return result


# ── CSV output helper ──────────────────────────────────────────────────────────

def build_csv(clean_records: list[dict]) -> str:
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(clean_records)
    return buf.getvalue()
