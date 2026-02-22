"""
pipeline.py
Core processing logic for the Metadata Cleaning pipeline.

Stages:
  1. PDF text extraction (pdfplumber)
  2. LLM-based song extraction from each source
  3. Conflict-aware merge (PDF = source of truth)
  4. Deterministic validation (ISRC format, date format)
  5. CSV output
"""

import re
import json
import pdfplumber
from datetime import datetime
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

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


SONG_FIELDS = ["song_title", "recording_artist", "writers", "isrc", "release_date"]

EXTRACTION_SYSTEM_PROMPT = """You are a precise music metadata extraction assistant working for a major music publisher.
Your job is to extract structured song metadata from text documents.

You must return a valid JSON array where each element is a song object with these exact keys:
  - song_title      (string)
  - recording_artist (string)
  - writers         (string — comma-separated list of full names)
  - isrc            (string — as written in the source, do NOT normalize)
  - release_date    (string — as written in the source, do NOT normalize)

Rules:
- Extract EVERY song mentioned.
- If a field is genuinely missing or unclear, use null.
- Do NOT invent or infer data not present in the text.
- Do NOT normalize ISRCs or dates — return them exactly as they appear.
- Return ONLY the JSON array. No markdown, no explanations, no extra text.

Example output:
[
  {
    "song_title": "Tokyo Midnight",
    "recording_artist": "The Neon Lights",
    "writers": "Alex Park, Jane Miller",
    "isrc": "US-WB1-24-00432",
    "release_date": "Nov 15, 2024"
  }
]"""

MERGE_SYSTEM_PROMPT = """You are a music metadata specialist responsible for deduplicating and merging song records from multiple sources.

You will receive song records from three sources: "PDF" (most authoritative), "Email A", and "Email B".
Apply this strict hierarchy when merging:

HIERARCHY RULES:
1. PDF is the source of truth. For any field that exists in the PDF record, use the PDF value — even if emails differ.
2. If the PDF record has null/empty for a field, fill it in from Email A or Email B (prefer Email A if both have it).
3. Deduplicate by song title (case-insensitive). Songs appearing in multiple sources are the same song.
4. Include songs that appear ONLY in the PDF (not in any email) — these are valid entries.
5. Include songs that appear ONLY in emails — PDF may have missed logging them.
6. Do NOT merge distinct songs just because they have similar names. Use judgment.

Return a JSON array of merged song objects with these exact keys:
  song_title, recording_artist, writers, isrc, release_date, source_note

Where source_note explains: "PDF + Email A", "PDF only", "Email B only", etc.

Return ONLY the JSON array. No markdown, no explanations."""


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_file) -> str:
    """Extract all text from a PDF file-like object or path."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


# ── ISRC Validation ────────────────────────────────────────────────────────────

def normalize_isrc(raw: Optional[str]) -> Optional[str]:
    """
    Normalize ISRC to standard format: LLRRRYYNNNNN (no hyphens, 12 chars).
    Display format: CC-XXX-YY-NNNNN (with hyphens).

    Accepts variants like:
      US-WB1-24-00432   → USWB1240432  → US-WB1-24-00432  ✓
      USWB1-2400432     → strip non-alpha-num, reformat
      USWB12400434      → split by known positions
    """
    if not raw or str(raw).strip().lower() in ("", "null", "none"):
        return None

    # Strip all non-alphanumeric characters
    cleaned = re.sub(r"[^A-Z0-9a-z]", "", raw.strip()).upper()

    if len(cleaned) != 12:
        # Return raw but uppercased if we can't parse it
        return raw.strip().upper()

    # Standard 12-char ISRC: CC(2) + Registrant(3) + Year(2) + Designation(5)
    cc = cleaned[0:2]
    reg = cleaned[2:5]
    yr = cleaned[5:7]
    des = cleaned[7:12]

    return f"{cc}-{reg}-{yr}-{des}"


def validate_isrc(raw: Optional[str]) -> dict:
    """Return normalized ISRC and a validation flag."""
    normalized = normalize_isrc(raw)
    pattern = r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$"
    is_valid = bool(normalized and re.match(pattern, normalized))
    return {"value": normalized, "valid": is_valid}


# ── Date Validation ────────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%Y/%m/%d",
]


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Parse any reasonable date string and return YYYY-MM-DD."""
    if not raw or str(raw).strip().lower() in ("", "null", "none"):
        return None

    raw = re.sub(r"\s+", " ", str(raw).strip())
    # Remove ordinal suffixes: 1st, 2nd, 3rd, 4th
    raw = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Last-resort: extract digits and try to make sense
    digits = re.findall(r"\d+", raw)
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    month_match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                            raw, re.IGNORECASE)
    if month_match and digits:
        month_num = months[month_match.group(1).lower()]
        year_candidates = [int(d) for d in digits if len(d) == 4]
        day_candidates = [int(d) for d in digits if 1 <= int(d) <= 31]
        if year_candidates and day_candidates:
            try:
                return datetime(year_candidates[0], month_num, day_candidates[0]).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return raw  # Return as-is if all else fails


def validate_date(raw: Optional[str]) -> dict:
    normalized = normalize_date(raw)
    is_valid = bool(normalized and re.match(r"^\d{4}-\d{2}-\d{2}$", str(normalized)))
    return {"value": normalized, "valid": is_valid}


# ── LLM Interaction ────────────────────────────────────────────────────────────

def _call_llm(client, model: str, system: str, user: str, label: str) -> str:
    """Single LLM call with basic error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed ({label}): {e}")


def _parse_json_response(raw: str, label: str) -> list:
    """Robustly parse JSON from LLM response, stripping markdown fences."""
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "songs" in result:
            return result["songs"]
        raise ValueError(f"Unexpected JSON structure in {label}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse error in {label}: {e}\nRaw: {raw[:500]}")


def extract_songs_from_source(client, model: str, text: str, source_label: str) -> list:
    """Use LLM to extract songs from a text source."""
    user_prompt = f"""Extract all song records from the following {source_label} text.
Return a JSON array as specified.

--- {source_label.upper()} TEXT ---
{text}
--- END ---"""

    raw = _call_llm(client, model, EXTRACTION_SYSTEM_PROMPT, user_prompt, source_label)
    songs = _parse_json_response(raw, source_label)

    # Tag each record with its source
    for s in songs:
        s["_source"] = source_label
        for field in SONG_FIELDS:
            if field not in s:
                s[field] = None

    return songs


def merge_all_sources(client, model: str, pdf_songs: list, email_a_songs: list, email_b_songs: list) -> list:
    """Use LLM to intelligently merge records from all three sources."""

    combined_input = {
        "pdf_songs": pdf_songs,
        "email_a_songs": email_a_songs,
        "email_b_songs": email_b_songs,
    }

    user_prompt = f"""Merge the following song records from three sources using the hierarchy rules in your instructions.

{json.dumps(combined_input, indent=2)}

Return a single deduplicated JSON array of merged songs."""

    raw = _call_llm(client, model, MERGE_SYSTEM_PROMPT, user_prompt, "merge")
    return _parse_json_response(raw, "merge")


# ── Deterministic Post-Processing ─────────────────────────────────────────────

def clean_and_validate(merged_songs: list) -> tuple[list, list]:
    """
    Apply deterministic validation rules to merged songs:
      - Normalize ISRC to standard hyphenated format
      - Normalize Release Date to YYYY-MM-DD
      - Track validation issues

    Returns: (clean_records, validation_log)
    """
    clean_records = []
    validation_log = []

    for i, song in enumerate(merged_songs, 1):
        record = {}
        issues = []

        # Song title — strip extra whitespace/quotes
        title_raw = song.get("song_title") or ""
        record["Song Title"] = title_raw.strip().strip("'\"") or "MISSING"
        if not record["Song Title"] or record["Song Title"] == "MISSING":
            issues.append("Missing song title")

        # Writers
        writers_raw = song.get("writers") or ""
        record["Writers"] = str(writers_raw).strip() if writers_raw else "MISSING"

        # Recording artist
        artist_raw = song.get("recording_artist") or ""
        record["Recording Artist"] = str(artist_raw).strip() if artist_raw else "MISSING"

        # ISRC
        isrc_result = validate_isrc(song.get("isrc"))
        record["ISRC"] = isrc_result["value"] or "MISSING"
        if not isrc_result["valid"]:
            issues.append(f"ISRC could not be fully validated: '{song.get('isrc')}'")

        # Release date
        date_result = validate_date(song.get("release_date"))
        record["Release Date"] = date_result["value"] or "MISSING"
        if not date_result["valid"]:
            issues.append(f"Date could not be normalized: '{song.get('release_date')}'")

        # Source note (pass-through)
        record["_source_note"] = song.get("source_note", song.get("_source", "unknown"))

        clean_records.append(record)
        validation_log.append({
            "row": i,
            "song_title": record["Song Title"],
            "issues": issues,
            "source": record["_source_note"],
        })

    return clean_records, validation_log


# ── Main Pipeline Orchestrator ─────────────────────────────────────────────────

def run_pipeline(
    pdf_file,
    email_a_text: str,
    email_b_text: str,
    client,
    model: str,
    progress_callback=None,
) -> dict:
    """
    Full pipeline: ingest → extract → merge → validate → return results.

    Args:
        pdf_file:         File-like object or path to PDF
        email_a_text:     Email A text content
        email_b_text:     Email B text content
        client:           OpenAI-compatible client (OpenAI or Groq)
        model:            Model name string
        progress_callback: Optional callable(step: int, total: int, message: str)

    Returns:
        dict with keys: pdf_text, pdf_songs, email_a_songs, email_b_songs,
                        merged_songs, clean_records, validation_log, errors
    """

    def progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    errors = []
    result = {
        "pdf_text": None,
        "pdf_songs": [],
        "email_a_songs": [],
        "email_b_songs": [],
        "merged_songs": [],
        "clean_records": [],
        "validation_log": [],
        "errors": errors,
    }

    # ── Step 1: Extract PDF text ───────────────────────────────────────────────
    progress(1, 6, "Extracting text from PDF…")
    try:
        pdf_text = extract_pdf_text(pdf_file)
        result["pdf_text"] = pdf_text
    except Exception as e:
        errors.append(f"PDF extraction error: {e}")
        return result

    # ── Step 2: LLM extraction — PDF ──────────────────────────────────────────
    progress(2, 6, "AI extracting songs from PDF (Schedule A)…")
    try:
        pdf_songs = extract_songs_from_source(client, model, pdf_text, "PDF Agreement (Schedule A)")
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(f"PDF LLM extraction error: {e}")
        pdf_songs = []

    # ── Step 3: LLM extraction — Email A ──────────────────────────────────────
    progress(3, 6, "AI extracting songs from Email A (Creative Dept)…")
    try:
        email_a_songs = extract_songs_from_source(client, model, email_a_text, "Email A – Creative Dept")
        result["email_a_songs"] = email_a_songs
    except Exception as e:
        errors.append(f"Email A LLM extraction error: {e}")
        email_a_songs = []

    # ── Step 4: LLM extraction — Email B ──────────────────────────────────────
    progress(4, 6, "AI extracting songs from Email B (Artist Management)…")
    try:
        email_b_songs = extract_songs_from_source(client, model, email_b_text, "Email B – Artist Management")
        result["email_b_songs"] = email_b_songs
    except Exception as e:
        errors.append(f"Email B LLM extraction error: {e}")
        email_b_songs = []

    # ── Step 5: Merge with conflict resolution ─────────────────────────────────
    progress(5, 6, "Merging sources with conflict resolution (PDF = source of truth)…")
    try:
        merged = merge_all_sources(client, model, pdf_songs, email_a_songs, email_b_songs)
        result["merged_songs"] = merged
    except Exception as e:
        errors.append(f"Merge error: {e}")
        # Fallback: concatenate all sources
        merged = pdf_songs + email_a_songs + email_b_songs
        result["merged_songs"] = merged

    # ── Step 6: Deterministic validation ──────────────────────────────────────
    progress(6, 6, "Validating and normalizing ISRCs and dates…")
    clean_records, validation_log = clean_and_validate(merged)
    result["clean_records"] = clean_records
    result["validation_log"] = validation_log

    return result
