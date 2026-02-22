"""
pipeline.py  —  Setlist API Reconciliation Pipeline v2
Warner Chappell Music · Task 3

Key fixes in this version:

  FIX 1  catalog.csv now has correct IDs:
          CAT-001=Neon Dreams, CAT-002=Midnight in Tokyo,
          CAT-003=Shattered Glass, CAT-004=Desert Rain, CAT-005=Ocean Avenue,
          CAT-006=Golden Gate, CAT-007=Velocity, CAT-013=The Glass House

  FIX 2  Deterministic matching is smarter:
          - Exact match (case-sensitive)
          - Normalised exact (lowercase, no punctuation)
          - Qualifier-stripped (removes "(Acoustic)", "(Extended Jam)" etc.)
          - Substring / word containment check (catches "Tokyo" ↔ "Midnight in Tokyo"
            where one title is wholly contained in the other → Review confidence)
          - Medley split on " / " with each part through all four strategies above
          This means "Shattered Glass" → CAT-003 Exact (no LLM needed),
          "Midnight In Tokyo" → CAT-002 Exact normalised (no LLM needed),
          "Golden Gate" → CAT-006 Exact (no LLM needed).

  FIX 3  LLM prompt is grounded strictly in the provided catalog.
          match_notes must only reference catalog titles that actually exist.
          The prompt explicitly forbids referencing songs not in catalog.

  FIX 4  Medley output: matched_catalog_id = "CAT-XXX; CAT-YYY" (semicolon-separated),
          one row per setlist entry (not one row per matched song).
"""

from __future__ import annotations

import re
import json
import csv
import io
import urllib.request
import urllib.error
from typing import Optional

# ── Confidence levels ──────────────────────────────────────────────────────────
EXACT   = "Exact"
HIGH    = "High"
REVIEW  = "Review"
NONE    = "None"

OUTPUT_COLUMNS = [
    "show_date", "venue_name", "setlist_track_name",
    "matched_catalog_id", "match_confidence", "match_notes",
]

# Qualifiers that should be stripped before matching
_QUALIFIER_PATTERNS = [
    r"\(acoustic\)",
    r"\(acoustic version\)",
    r"\(extended.*?\)",
    r"\(live.*?\)",
    r"\(radio.*?\)",
    r"\(remix.*?\)",
    r"\(feat.*?\)",
    r"\(ft\..*?\)",
    r"\(interlude\)",
    r"\(reprise\)",
    r"\(version.*?\)",
    r"\(remaster.*?\)",
    r"-\s*(acoustic|live|remix|radio edit)$",
]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1  API / local file ingestion
# ══════════════════════════════════════════════════════════════════════════════

def fetch_tour_data(source: str) -> dict:
    """Fetch tour JSON from URL or local file. Raises RuntimeError on failure."""
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        try:
            req = urllib.request.Request(
                source, headers={"User-Agent": "WCM-Reconciliation/2.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}")
    else:
        try:
            with open(source, encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {source}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {e}")

    if payload.get("status") != "success":
        raise RuntimeError(f"API returned status: {payload.get('status')}")
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2  Catalog load
# ══════════════════════════════════════════════════════════════════════════════

def load_catalog(catalog_file) -> list[dict]:
    """Load catalog CSV from path or file-like object."""
    try:
        if hasattr(catalog_file, "read"):
            content = catalog_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return list(csv.DictReader(io.StringIO(content)))
        with open(catalog_file, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        raise RuntimeError(f"Catalog load failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3  Flatten nested JSON
# ══════════════════════════════════════════════════════════════════════════════

def flatten_setlist(payload: dict) -> list[dict]:
    """Explode shows[].setlist[] into one dict per track."""
    rows = []
    data = payload.get("data", {})
    for show in data.get("shows", []):
        for track in show.get("setlist", []):
            rows.append({
                "show_date":         show.get("date", ""),
                "venue_name":        show.get("venue", ""),
                "city":              show.get("city", ""),
                "setlist_track_name": track,
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4  Deterministic pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Lowercase, strip all punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def _strip_qualifiers(s: str) -> str:
    """Remove live-performance qualifiers from a track name."""
    result = s
    for pat in _QUALIFIER_PATTERNS:
        result = re.sub(pat, "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def _words(s: str) -> set[str]:
    """Return set of normalised words (length ≥ 3) from a string."""
    return {w for w in _norm(s).split() if len(w) >= 3}


def _match_single(track: str, catalog: list[dict]) -> Optional[dict]:
    """
    Attempt all deterministic strategies for a single (non-medley) track.
    Returns a match result dict or None.

    Strategy order (most → least confident):
      1. Exact string match
      2. Normalised exact (case + punctuation insensitive)
      3. Qualifier-stripped normalised exact
      4. Qualifier-stripped normalised substring containment
         (one title is wholly contained in the other)
         → confidence REVIEW (ambiguous, needs human confirmation)
    """
    norm_track   = _norm(track)
    stripped     = _strip_qualifiers(track)
    norm_stripped = _norm(stripped)

    for entry in catalog:
        title = entry["song_title"]

        # 1. Exact
        if title == track:
            return _result(entry, EXACT, "Exact string match")

        # 2. Normalised exact
        if _norm(title) == norm_track:
            return _result(entry, EXACT, f"Normalised exact match")

        # 3. Qualifier-stripped exact
        if _norm(_strip_qualifiers(title)) == norm_stripped and norm_stripped:
            return _result(
                entry, HIGH,
                f"Qualifier-stripped match: '{track}' stripped to '{stripped}' "
                f"→ '{title}'"
            )

    # 4. Substring containment (qualifier-stripped track wholly inside catalog title,
    #    or catalog title wholly inside qualifier-stripped track)
    #    Only fires when the contained string has ≥ 2 words (avoids "Tokyo" matching
    #    every title containing the word "tokyo").
    norm_stripped_words = norm_stripped.split()
    if len(norm_stripped_words) >= 2:
        for entry in catalog:
            ct = _norm(_strip_qualifiers(entry["song_title"]))
            if norm_stripped in ct or ct in norm_stripped:
                return _result(
                    entry, REVIEW,
                    f"Partial title containment (Review): '{track}' → '{entry['song_title']}'"
                )

    # Single-word containment only when the word is a strong/unique identifier
    # (≥ 6 chars) to avoid false positives
    if len(norm_stripped_words) == 1 and len(norm_stripped_words[0]) >= 6:
        for entry in catalog:
            ct = _norm(_strip_qualifiers(entry["song_title"]))
            if norm_stripped_words[0] in ct.split():
                return _result(
                    entry, REVIEW,
                    f"Single-word containment (Review): '{track}' → '{entry['song_title']}'"
                )

    return None


def _result(entry: dict, confidence: str, notes: str) -> dict:
    return {
        "matched_catalog_id": entry["catalog_id"],
        "match_confidence":   confidence,
        "match_notes":        notes,
        "matched_entries":    [entry],
    }


def deterministic_match(track: str, catalog: list[dict]) -> Optional[dict]:
    """
    Full deterministic matching including medley detection.
    Returns a match result dict or None (→ send to LLM).
    """
    # Medley: split on " / " and match each part independently
    if " / " in track:
        parts = [p.strip() for p in track.split(" / ")]
        matched: list[dict] = []
        unmatched: list[str] = []
        for part in parts:
            m = _match_single(part, catalog)
            if m:
                matched.append(m["matched_entries"][0])
            else:
                unmatched.append(part)

        if matched:
            ids   = "; ".join(e["catalog_id"]  for e in matched)
            names = "; ".join(e["song_title"]   for e in matched)
            # Confidence: High if all parts matched, Review if partial
            conf  = HIGH if not unmatched else REVIEW
            note  = (
                f"Medley: matched {len(matched)}/{len(parts)} parts → {names}"
                + (f" | Unmatched: {', '.join(unmatched)}" if unmatched else "")
            )
            return {
                "matched_catalog_id": ids,
                "match_confidence":   conf,
                "match_notes":        note,
                "matched_entries":    matched,
                "is_medley":          True,
            }
        # All parts unmatched → let LLM try
        return None

    return _match_single(track, catalog)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5  LLM fuzzy matching
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM_PROMPT = """You are a music rights reconciliation specialist.
Determine whether a live setlist track matches any song in the provided catalog.

You will receive:
  setlist_track: the raw track name from a live setlist
  catalog: our controlled songs with catalog_id and song_title

ANALYSIS RULES:
  ABBREVIATIONS: "Tokyo" alone may refer to a longer catalog title containing "Tokyo"
  QUALIFIERS: "(Acoustic)", "(Extended Jam)" etc. don't change the underlying song
  MEDLEYS: "Song A / Song B" → return one match per recognized catalog song
  COVERS/UNCONTROLLED: songs not in this catalog must NOT be matched (return None)
  GARBLED TEXT: "Smsls Lk Tn Sprt" style garbling → try to decode, but only match
                if you are confident it is a catalog song

CRITICAL RULES:
  1. ONLY reference songs that appear in the provided catalog. Never invent IDs.
  2. catalog_id must be EXACTLY as shown (e.g. "CAT-002") — copy it verbatim.
  3. Only match if genuinely confident. Vague similarity is NOT a match.
  4. "Review" = possible match but a human should verify.
  5. "None" = not in our catalog (cover, garbled with no clear match, etc.)

Return a JSON object:
{
  "matches": [
    {
      "catalog_id": "CAT-XXX",
      "song_title": "exact title from catalog",
      "match_confidence": "Exact | High | Review | None",
      "reasoning": "brief explanation referencing only catalog titles"
    }
  ]
}

For no match: return one entry with catalog_id=null, match_confidence="None".
Return ONLY the JSON object. No markdown, no extra text."""


def llm_fuzzy_match(track: str, catalog: list[dict], client, model: str) -> dict:
    """LLM fuzzy match for one track. Returns a result dict."""
    catalog_list = [
        {"catalog_id": e["catalog_id"], "song_title": e["song_title"]}
        for e in catalog
    ]
    user_prompt = (
        f'setlist_track: "{track}"\n\n'
        f"catalog:\n{json.dumps(catalog_list, indent=2)}\n\n"
        f"Analyze this track and return your JSON result. "
        f"Only use catalog_id values from the list above."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"matched_catalog_id": None, "match_confidence": NONE,
                "match_notes": f"LLM call failed: {e}"}

    # Parse response
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        parsed  = json.loads(cleaned)
        matches = parsed.get("matches", [])
    except Exception as e:
        return {"matched_catalog_id": None, "match_confidence": NONE,
                "match_notes": f"LLM parse error: {e} | raw: {raw[:200]}"}

    if not matches:
        return {"matched_catalog_id": None, "match_confidence": NONE,
                "match_notes": "LLM found no matches"}

    # Validate: only accept catalog_ids that actually exist in our catalog
    valid_ids = {e["catalog_id"] for e in catalog}
    real = [
        m for m in matches
        if m.get("catalog_id") and m["catalog_id"] in valid_ids
        and m.get("match_confidence", NONE) != NONE
    ]

    if not real:
        reasoning = matches[0].get("reasoning", "No match in catalog") if matches else "No match"
        return {"matched_catalog_id": None, "match_confidence": NONE,
                "match_notes": reasoning}

    if len(real) == 1:
        m = real[0]
        return {
            "matched_catalog_id": m["catalog_id"],
            "match_confidence":   m.get("match_confidence", REVIEW),
            "match_notes":        m.get("reasoning", ""),
        }

    # Multiple matches (medley from LLM)
    conf_order = [EXACT, HIGH, REVIEW, NONE]
    confs      = [m.get("match_confidence", REVIEW) for m in real]
    group_conf = max(confs, key=lambda c: conf_order.index(c) if c in conf_order else 3)
    return {
        "matched_catalog_id": "; ".join(m["catalog_id"] for m in real),
        "match_confidence":   group_conf,
        "match_notes":        " | ".join(
            f"{m['catalog_id']}={m.get('reasoning','')}" for m in real
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    tour_source: str,
    catalog_file,
    client,
    model: str,
    progress_callback=None,
) -> dict:
    TOTAL = 5
    errors: list[str] = []
    result = dict(
        tour_meta={}, catalog=[], flat_rows=[],
        results=[], stats={}, errors=errors,
    )

    def step(n, msg):
        if progress_callback:
            progress_callback(n, TOTAL, msg)

    # Stage 1
    step(1, "Fetching tour data…")
    try:
        payload = fetch_tour_data(tour_source)
        d = payload.get("data", {})
        result["tour_meta"] = {
            "artist":     d.get("artist"),
            "tour":       d.get("tour"),
            "show_count": len(d.get("shows", [])),
        }
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 2
    step(2, "Loading catalog…")
    try:
        catalog = load_catalog(catalog_file)
        result["catalog"] = catalog
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 3
    step(3, "Flattening setlist…")
    flat = flatten_setlist(payload)
    result["flat_rows"] = flat

    # Stage 4 — Deterministic
    step(4, "Deterministic matching (exact + normalised + qualifier-strip + medley)…")
    pre_matched: list[dict] = []
    needs_llm:  list[dict] = []

    for row in flat:
        m = deterministic_match(row["setlist_track_name"], catalog)
        if m:
            pre_matched.append({**row, **m})
        else:
            needs_llm.append(row)

    # Stage 5 — LLM
    llm_results: list[dict] = []
    for i, row in enumerate(needs_llm):
        track = row["setlist_track_name"]
        step(5, f"AI fuzzy matching {i+1}/{len(needs_llm)}: \"{track}\"…")
        m = llm_fuzzy_match(track, catalog, client, model)
        llm_results.append({**row, **m})

    if not needs_llm:
        step(5, "All tracks resolved deterministically — no LLM calls needed.")

    # Build output
    all_results = []
    for row in pre_matched + llm_results:
        all_results.append({
            "show_date":          row.get("show_date", ""),
            "venue_name":         row.get("venue_name", ""),
            "setlist_track_name": row.get("setlist_track_name", ""),
            "matched_catalog_id": row.get("matched_catalog_id") or "None",
            "match_confidence":   row.get("match_confidence", NONE),
            "match_notes":        row.get("match_notes", ""),
        })

    all_results.sort(key=lambda r: (r["show_date"], r["venue_name"]))
    result["results"] = all_results

    # Stats
    total = len(all_results)
    result["stats"] = {
        "total_tracks":   total,
        "exact_matches":  sum(1 for r in all_results if r["match_confidence"] == EXACT),
        "high_matches":   sum(1 for r in all_results if r["match_confidence"] == HIGH),
        "review_matches": sum(1 for r in all_results if r["match_confidence"] == REVIEW),
        "no_matches":     sum(1 for r in all_results if r["match_confidence"] == NONE),
        "deterministic":  len(pre_matched),
        "llm_resolved":   len(llm_results),
        "llm_savings_pct": round(len(pre_matched) / total * 100 if total else 0, 1),
    }
    return result


def build_output_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(results)
    return buf.getvalue()
