# Task 1 — Metadata Cleaning Pipeline
### Warner Chappell Music · AI Automation Analyst Assessment

A production-grade Streamlit application that ingests song metadata from multiple unstructured sources (a PDF publishing agreement and two emails), resolves conflicts using a defined source hierarchy, and delivers a clean, validated CSV.

---

## Quick Start

```bash
# 1. Clone / unzip and enter the directory
cd task1_metadata_cleaning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the test PDF (Schedule A with intentional discrepancies)
python create_sample_pdf.py

# 4. Launch the app
streamlit run app.py
```

Then in the sidebar:
- Paste your **Groq** or **OpenAI** API key
- Upload `sample_schedule_a.pdf` (or your own PDF)
- Click **▶ Run Pipeline**

---

## Environment

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Auto-detected if set — no sidebar input needed |
| `OPENAI_API_KEY` | Fallback if Groq key not set |

---

## Architecture

```
app.py                 ← Streamlit UI (sidebar config, tabs, progress, results)
pipeline.py            ← Core processing logic (all stages)
create_sample_pdf.py   ← Generates a realistic test PDF with intentional issues
requirements.txt
README.md
sample_schedule_a.pdf  ← Generated test file (after running create_sample_pdf.py)
schedule_a_clean.csv   ← Pipeline output (generated after running)
```

---

## Design Decisions

### Process Orchestration

The pipeline runs in **six sequential stages**, each calling back to the UI with progress updates:

1. **PDF text extraction** (deterministic, via `pdfplumber`)
2. **LLM extraction — PDF** (one focused LLM call per source)
3. **LLM extraction — Email A**
4. **LLM extraction — Email B**
5. **LLM merge** (one additional call that applies hierarchy rules)
6. **Deterministic validation** (regex-based ISRC + date normalization — no LLM)

I chose a **multi-stage chain** rather than a single mega-prompt for these reasons:
- **Separation of concerns**: each LLM call has a narrow, verifiable task. If extraction from Email A fails, the other sources are unaffected.
- **Debuggability**: the UI exposes each stage's raw JSON output (`Show raw LLM extractions` toggle), so it's easy to pinpoint where any error originated.
- **Auditability**: the source of each output row is tracked via a `source_note` field throughout the pipeline.

### Logic vs. Inference

| Task | Approach | Rationale |
|---|---|---|
| Text extraction from PDF | Deterministic (`pdfplumber`) | Structured extraction is precise and cheap |
| Parsing metadata fields from free-form text | LLM | Emails are unstructured prose with many formats |
| Deduplication + conflict resolution | LLM | Requires semantic understanding of "same song" across differently-formatted titles |
| ISRC normalization | Deterministic (regex) | Format is strictly defined; LLM would introduce variability |
| Date normalization | Deterministic (multiple `strptime` patterns) | All common formats are enumerable; determinism is essential |
| Flagging validation issues | Deterministic (regex match post-normalize) | No inference needed — pass/fail against a known pattern |

**Key principle**: LLMs handle ambiguity and natural language; code handles structured, rule-bound transformations.

### Conflict & Exception Handling

#### Source Hierarchy
The merge LLM prompt explicitly encodes the hierarchy:

```
PDF Agreement → authoritative for any field it provides
Email A       → fills gaps the PDF left empty (preferred over B)
Email B       → fills gaps neither PDF nor A filled
```

The system prompt uses unambiguous language ("use the PDF value **even if emails differ**") rather than softer phrasing, which reduces hallucination risk.

#### Edge Cases Handled
| Scenario | Handling |
|---|---|
| ISRC in non-standard format (e.g., `USWB1-2400432`) | Stripped to 12 alphanumeric chars, re-hyphenated by position |
| Date in any common format (e.g., `Nov 15, 2024`, `15/11/2024`) | Tried against 10 `strptime` patterns, then regex fallback |
| Missing writer in PDF, present in email | LLM merge fills gap from email |
| Song only in PDF (not in emails) | Included in output with `source_note: "PDF only"` |
| Song only in emails (not in PDF) | Included with `source_note: "Email A only"` etc. |
| ISRC that can't be fully normalized | Flagged in validation log, output as-is |
| LLM returns non-JSON / markdown-fenced JSON | Regex strips fences before `json.loads` |
| LLM call fails entirely | `errors` list in result dict; UI shows error box |
| PDF extraction fails | Pipeline aborts early with clear error message |

#### JSON Parsing Robustness
LLM responses sometimes wrap JSON in markdown code fences (` ```json ... ``` `). `_parse_json_response()` strips these before parsing, and also handles the case where the LLM wraps the array in a `{"songs": [...]}` envelope.

### Scalability & Reliability

**Volume**: The current approach makes 5 LLM calls per run (3 extractions + 1 merge + overhead). For batch processing of many agreements:
- Extract PDF and email texts in parallel using `asyncio` + the async OpenAI/Groq client
- Cache extraction results (hash the source text; skip re-extraction if unchanged)
- Process PDF/email extraction as a map step, merge as a reduce step

**Format shifts**: Because each extraction call uses a source-agnostic natural-language prompt rather than rigid templates, the system naturally adapts to format changes in emails. If PDF layout changes significantly, `pdfplumber`'s table extraction mode can be added as a fallback to text extraction.

**LLM cost**: Groq's Llama 3.3 70B is free-tier for reasonable volumes. If using OpenAI, `gpt-4o-mini` handles this task well at low cost (~$0.001/run for typical input sizes).

---

## Sample PDF Discrepancies (for testing conflict resolution)

`create_sample_pdf.py` deliberately generates a Schedule A with these issues:

| Row | Issue |
|---|---|
| Tokyo Midnight | ISRC: `USWB1-2400432` (non-standard) · Date: `15/11/2024` |
| Desert Rain | ISRC: `USWB12400434` (no hyphens) · Date: `Nov 22 2024` |
| Neon Dreams | Date: `November 15, 2024` (long month name) |
| Uptown | Writers: empty in PDF (filled from Email A) |
| Velocity | ISRC: `USWB1-2400438` · Date: `05/12/2024` |
| Blue Monday | Writers: empty in PDF (filled from Email B) |
| Phantom City | PDF-only entry (not in any email) |

---

## Output Format

`schedule_a_clean.csv` columns:

| Column | Format | Example |
|---|---|---|
| Song Title | String | `Tokyo Midnight` |
| Writers | Comma-separated names | `Alex Park, Jane Miller` |
| Recording Artist | String | `The Neon Lights` |
| ISRC | `CC-RRR-YY-NNNNN` | `US-WB1-24-00432` |
| Release Date | `YYYY-MM-DD` | `2024-11-15` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `openai` | OpenAI-compatible client (works with Groq via base_url) |
| `pdfplumber` | PDF text and table extraction |
| `reportlab` | Generates the sample test PDF |
| `pandas` | DataFrame display in Streamlit |
