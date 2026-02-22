"""
pipeline.py  —  Metadata Cleaning Pipeline v4 (LLM-centric)
Warner Chappell Music · Task 1

This version keeps the PDF side LLM-driven (as you prefer), but makes it
MUCH more robust by:

  • Pre-segmenting the Schedule A text into one block per numbered track
    (01–10) before it ever hits the model, so it never has to infer row
    boundaries from the whole page at once.
  • Giving the model an explicit example of how to separate
    Song Title vs Recording Artist vs Writers using the actual
    column order.
  • Adding light, deterministic post-processing to prevent obvious
    leaks like "feat." living in the title instead of the artist.

Emails are still parsed by the LLM as before.
"""

from __future__ import annotations

import re
import io
import csv
import json
import pdfplumber
from datetime import datetime
from typing import Optional, List, Dict

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
# Stage 1  PDF raw text extraction (for debug display)
# ══════════════════════════════════════════════════════════════════════════════

def extract_schedule_a_text(pdf_file) -> str:
    """
    Extract the raw text from whichever page contains 'Schedule A'.
    Used for UI inspection and as the input for LLM segmentation.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if re.search(r"schedule\s+a", text, re.IGNORECASE):
                    return text
            # Fallback: return all text combined if we somehow don't see the label
            return "\n\n".join(
                (p.extract_text() or "") for p in pdf.pages
            )
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2  LLM extraction — PDF text (track-segmented)
# ══════════════════════════════════════════════════════════════════════════════

def _segment_schedule_rows(raw_text: str) -> List[Dict[str, str]]:
    """
    Split the Schedule A text into one block per numbered track.

    We treat any line that begins with a 2-digit number at the start of the line
    (01–10) as the start of a new track block and keep all following lines
    until the next track number.
    """
    blocks: List[Dict[str, str]] = []

    lines = [ln for ln in raw_text.splitlines()]
    current_track: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        # Match lines like "01 ", "02   ", "10  "
        m = re.match(r"^\s*(0[1-9]|10)\b", line)
        if m:
            # Flush previous block
            if current_track is not None:
                blocks.append(
                    {"track": current_track, "text": "\n".join(current_lines).strip()}
                )
            current_track = m.group(1)
            current_lines = [line]
        else:
            if current_track is not None:
                current_lines.append(line)

    if current_track is not None:
        blocks.append(
            {"track": current_track, "text": "\n".join(current_lines).strip()}
        )

    # In this case study we expect exactly 10 rows; if fewer, we still continue,
    # but the LLM prompt will mention any missing numbers.
    return blocks


_PDF_SYSTEM_PROMPT = """You are a meticulous music publishing data extraction assistant.

You will receive pre-segmented text blocks from a "Schedule A" table in a
publishing agreement. The original table has COLUMNS in this exact order:

  Track | Song Title | Primary Artist | Writers | Release Date | ISRC

The text you receive is already grouped by TRACK, so each block corresponds
to exactly one row, including any wrapped lines.

Your job is to read ALL of the blocks and return ONE JSON array.

Each element of the array must have EXACTLY these keys:

  "track"             - the 2-digit track number as a string (e.g. "01")
  "song_title"        - the full song title ONLY, never including the artist
  "recording_artist"  - the primary recording / performing artist
                        (may include feat. credits)
  "writers"           - comma-separated writer names, or null if listed as
                        TBD / blank
  "isrc"              - the ISRC code exactly as written, or null if blank
  "release_date"      - the release date exactly as written (e.g. 2024-12-10),
                        or null if blank

CRITICAL FIELD RULES:

1. The columns are in the order:
   Song Title → Primary Artist → Writers → Release Date → ISRC.
   Respect this order when splitting the row text.

2. song_title NEVER contains the artist or writers.
   Example: row text "... Uptown  Funk Theory  Marcus Davis ...":
     • song_title        = "Uptown"
     • recording_artist  = "Funk Theory"
     • writers           = "Marcus Davis"

3. recording_artist is the performer, and may include a "feat." section.
   If the row text contains "(feat. DJ Flux)" or similar inside the title
   portion, MOVE that to the recording_artist instead of leaving it in the
   title. For example:
     Row text: "Neon Dreams (feat. DJ Flux)  The Neon Lights ..."
       → song_title        = "Neon Dreams"
       → recording_artist  = "The Neon Lights (feat. DJ Flux)"

4. writers must only contain songwriter names, not artists.
   If a name appears in the Writers column (e.g. "Rachel Davis") and also
   appears in another column, it is a WRITER, not an artist.

5. "TBD", "N/A", "-", "—", "None" etc. mean that a field is not provided.
   Represent that field as null in the JSON.

6. Include EVERY track block provided. Do not invent any extra tracks.
   Do not drop any.

7. Return ONLY a valid JSON array (no markdown, no comments)."""


def _call_llm(client, model: str, system: str, user: str) -> List[dict]:
    """Single LLM call → parse JSON array response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if any
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array from the model.")
    return data


def extract_pdf_songs(pdf_file, client, model: str) -> List[dict]:
    """
    Extract songs from the Schedule A text via LLM, with per-track segmentation.

    Returns list of dicts with canonical fields + '_source' + '_track'.
    """
    raw_text = extract_schedule_a_text(pdf_file)
    blocks = _segment_schedule_rows(raw_text)
    if not blocks:
        raise RuntimeError("Could not find any numbered Schedule A rows (01–10).")

    # Build a compact, strongly-labeled prompt for the model
    block_texts = []
    for b in blocks:
        block_texts.append(
            f"TRACK {b['track']} BLOCK:\n{b['text']}\nEND TRACK {b['track']}\n"
        )
    user_prompt = (
        "Below are the text blocks for each numbered track from the Schedule A "
        "table. Use them to extract the fields for every track.\n\n"
        + "\n".join(block_texts)
        + "\nRemember: output a single JSON array for ALL tracks."
    )

    songs = _call_llm(client, model, _PDF_SYSTEM_PROMPT, user_prompt)

    canonical = ["song_title", "recording_artist", "writers", "isrc", "release_date"]
    result: List[dict] = []

    for s in songs:
        if not isinstance(s, dict):
            continue
        # Track number
        track = str(s.get("track") or "").strip()
        record = {f: str(s.get(f) or "").strip() for f in canonical}

        # Treat missing-style values as empty strings
        for f in canonical:
            if record[f].lower() in _MISSING:
                record[f] = ""

        # Light clean-ups:
        #   • strip any quotes from song titles
        #   • move "(feat. ...)" from title to artist if needed
        title = record["song_title"].strip().strip("'\"")
        artist = record["recording_artist"].strip()

        lower_title = title.lower()
        if "feat." in lower_title or "featuring" in lower_title or "ft." in lower_title:
            # If the artist field does NOT already contain a feat, assume the
            # feat part belongs in the artist.
            if not re.search(r"\bfeat\.|\bft\.|\bfeaturing", artist, re.IGNORECASE):
                m = re.search(
                    r"(\s*\(?\bfeat\.|\s*\(?\bft\.|\s*\(?\bfeaturing\b).*",
                    title,
                    flags=re.IGNORECASE,
                )
                if m:
                    feat_part = title[m.start():].strip()
                    base_title = title[: m.start()].rstrip(" -(/").strip()
                    title = base_title
                    artist = (artist + " " + feat_part).strip() if artist else feat_part

        record["song_title"] = title
        record["recording_artist"] = artist

        record["_source"] = "PDF"
        record["_track"] = track
        result.append(record)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3  LLM extraction — Emails
# ══════════════════════════════════════════════════════════════════════════════

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


def extract_email_songs(
    client, model: str, email_text: str, source_label: str
) -> List[dict]:
    """LLM extraction from one email → list of song dicts."""
    user_prompt = (
        f"Extract all songs from this email.\n\n"
        f"--- {source_label.upper()} ---\n{email_text}\n--- END ---\n\n"
        f"Return a JSON array. Remember: song_title and recording_artist are always separate fields."
    )
    songs = _call_llm(client, model, _EMAIL_SYSTEM_PROMPT, user_prompt)

    canonical = ["song_title", "recording_artist", "writers", "isrc", "release_date"]
    result: List[dict] = []
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
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _norm_isrc(isrc: str) -> str:
    """Strip hyphens/spaces for ISRC comparison."""
    return re.sub(r"[^A-Z0-9]", "", (isrc or "").upper())


def _build_email_indexes(songs: List[dict]) -> tuple[dict, dict]:
    """
    Build two lookup dicts from email songs:
      isrc_index  : normalised_isrc  → song dict
      title_index : normalised_title → song dict
    First occurrence wins (Email A takes priority over B).
    """
    isrc_idx: Dict[str, dict] = {}
    title_idx: Dict[str, dict] = {}
    for s in songs:
        ni = _norm_isrc(s.get("isrc", ""))
        nt = _norm_title(s.get("song_title", ""))
        if ni and ni not in isrc_idx:
            isrc_idx[ni] = s
        if nt and nt not in title_idx:
            title_idx[nt] = s
    return isrc_idx, title_idx


def merge_songs(
    pdf_songs: List[dict],
    email_a_songs: List[dict],
    email_b_songs: List[dict],
) -> tuple[List[dict], List[str], List[str]]:
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

    FIELDS = ["song_title", "recording_artist", "writers", "isrc", "release_date"]

    merged: List[dict] = []
    fill_log: List[str] = []
    conflict_log: List[str] = []

    for pdf_song in pdf_songs:
        title = pdf_song.get("song_title", "")
        ni_pdf = _norm_isrc(pdf_song.get("isrc", ""))
        nt_pdf = _norm_title(title)

        # Find best email match
        email_hit = (
            isrc_idx.get(ni_pdf)    # 1. ISRC match (best)
            or title_idx.get(nt_pdf)  # 2. Title match (fallback)
        )

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

        record: dict = {}
        field_sources: dict = {}

        for field in FIELDS:
            pdf_val = (pdf_song.get(field, "") or "").strip()
            email_val = ((email_hit or {}).get(field) or "").strip()

            if pdf_val.lower() in _MISSING:
                pdf_val = ""
            if email_val.lower() in _MISSING:
                email_val = ""

            if pdf_val:
                record[field] = pdf_val
                field_sources[field] = "PDF"
                if email_val and _norm_title(email_val) != _norm_title(pdf_val):
                    conflict_log.append(
                        f"  CONFLICT '{title}'.{field}: "
                        f"PDF='{pdf_val}' | {esrc}='{email_val}' → kept PDF"
                    )
            elif email_val:
                record[field] = email_val
                field_sources[field] = esrc
                fill_log.append(
                    f"  FILLED '{title}'.{field} from {esrc} "
                    f"[matched by {match_method}]: '{email_val}'"
                )
            else:
                record[field] = ""
                field_sources[field] = "missing"
                fill_log.append(
                    f"  MISSING '{title}'.{field} — not found in any source"
                )

        filled = [
            f for f, s in field_sources.items() if s not in ("PDF", "missing")
        ]
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
    "%d %B %Y", "%d %b %Y",
    "%b %d %Y", "%B %d %Y",
    "%Y/%m/%d",
]


def normalize_isrc(raw: Optional[str]) -> Optional[str]:
    """Normalize to CC-RRR-YY-NNNNN (12 alphanumeric, hyphenated)."""
    if not raw or str(raw).strip().lower() in _MISSING:
        return None
    stripped = re.sub(r"[^A-Za-z0-9]", "", str(raw).strip()).upper()
    if len(stripped) != 12:
        # Keep an uppercased, cleaned value for inspection even if invalid
        return str(raw).strip().upper()
    return f"{stripped[0:2]}-{stripped[2:5]}-{stripped[5:7]}-{stripped[7:12]}"


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Parse any common date format → YYYY-MM-DD."""
    if not raw or str(raw).strip().lower() in _MISSING:
        return None
    text = re.sub(
        r"(\d+)(st|nd|rd|th)\b",
        r"\1",
        re.sub(r"\s+", " ", str(raw).strip()),
        flags=re.IGNORECASE,
    )

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try a loose month-name parse if formats failed
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                  text, re.IGNORECASE)
    digits = re.findall(r"\d+", text)
    years = [int(d) for d in digits if len(d) == 4]
    days = [int(d) for d in digits if 1 <= int(d) <= 31 and len(d) <= 2]

    if m and years and days:
        try:
            return datetime(years[0], months[m.group(1).lower()], days[0])\
                .strftime("%Y-%m-%d")
        except ValueError:
            pass

    return raw


def validate_and_finalize(merged: List[dict]) -> tuple[List[dict], List[dict]]:
    """Normalize ISRC + dates, flag issues per row."""
    clean: List[dict] = []
    log: List[dict] = []

    for i, song in enumerate(merged, 1):
        record: dict = {}
        issues: List[str] = []

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
            r"^[A-Z]{2}-[A-Z0-9]{3}-\d{2}-\d{5}$", str(isrc_norm)
        ):
            issues.append(f"ISRC invalid/unnormalizable: '{song.get('isrc')}'")

        date_norm = normalize_date(song.get("release_date"))
        record["Release Date"] = date_norm or "MISSING"
        if not date_norm or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_norm)):
            issues.append(f"Date could not be normalized: '{song.get('release_date')}'")

        record["_source_note"] = song.get("_source_note", "unknown")
        clean.append(record)
        log.append(
            {
                "row": i,
                "song_title": record["Song Title"],
                "issues": issues,
                "source": record["_source_note"],
                "clean": len(issues) == 0,
            }
        )

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
    errors: List[str] = []
    result = dict(
        pdf_raw_text="",
        pdf_songs=[],
        email_a_songs=[],
        email_b_songs=[],
        merged_songs=[],
        clean_records=[],
        validation_log=[],
        fill_log=[],
        conflict_log=[],
        errors=errors,
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

    # Stage 2: LLM parses PDF text (track-segmented)
    step(2, "Stage 2/4 — AI parsing Schedule A text (one block per track)…")
    try:
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        pdf_songs = extract_pdf_songs(pdf_file, client, model)
        result["pdf_songs"] = pdf_songs
    except Exception as e:
        errors.append(f"PDF LLM extraction: {e}")
        return result

    # Stage 3a: LLM parses Email A
    step(3, "Stage 3/4 — AI parsing Email A (Creative Dept)…")
    try:
        ea = extract_email_songs(
            client, model, email_a_text, "Email A – Creative Dept"
        )
        result["email_a_songs"] = ea
    except Exception as e:
        errors.append(f"Email A: {e}")
        ea = []

    # Stage 3b: LLM parses Email B
    step(4, "Stage 3/4 — AI parsing Email B (Artist Management)…")
    try:
        eb = extract_email_songs(
            client, model, email_b_text, "Email B – Artist Management"
        )
        result["email_b_songs"] = eb
    except Exception as e:
        errors.append(f"Email B: {e}")
        eb = []

    # Stage 4: Python merge
    step(5, "Stage 3/4 — Merging (ISRC-first matching, PDF wins all conflicts)…")
    try:
        merged, fill_log, conflict_log = merge_songs(pdf_songs, ea, eb)
        result.update(
            merged_songs=merged,
            fill_log=fill_log,
            conflict_log=conflict_log,
        )
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

def build_csv(clean_records: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(clean_records)
    return buf.getvalue()
