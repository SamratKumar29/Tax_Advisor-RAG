"""
Chunking pipeline for Indian Tax Advisor RAG.

- Parses PDFs with pypdf (page by page, so each chunk keeps its page number)
  and Markdown with plain text reading.
- Normalises extracted text (rupee sign, amounts in words) — see src.text_utils.
- Splits with a recursive character splitter, then, for structured legal docs
  (Act, Rules), assigns each chunk the section / rule / schedule it belongs to
  by tracking headings across the ordered chunk stream.
- Outputs data/processed/chunks.jsonl.

Why track headings instead of splitting on them: the Act's headings look like
"202. (1) ..." but the same pattern also appears in numbered table rows and
footnotes ("1. Substituted by the Finance Act, 2026"). Section numbers in the
Act increase monotonically, so a heading is only accepted when its number is a
small step forward from the current section. This rejects nearly all the noise
without needing a PDF layout parser.
"""

import hashlib
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_JSONL_PATH,
    DATA_RAW_DIR,
)
from src.ingest.tag_metadata import tag_chunk
from src.text_utils import normalize_text

# ── Text extraction ───────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract text per page using pypdf (empty pages are kept as "")."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_markdown_text(md_path: Path) -> str:
    """Read a Markdown file as plain text."""
    return md_path.read_text(encoding="utf-8")


# ── Section / rule / schedule tracking (for Act / Rules) ──────────────────

# "202. (1) ...", "123. Deduction ...", "80A. ..." at the start of a line,
# excluding amendment footnotes that use the same numbering.
HEADING_PATTERN = re.compile(
    r"^\s*(\d{1,3})([A-Z]{0,2})\.\s+"
    r"(?!(?:Substituted|Omitted|Inserted|Ins\.|Subs\.|Words)\b)[A-Z“\"(]",
    re.MULTILINE,
)
SCHEDULE_PATTERN = re.compile(r"^\s*SCHEDULE\s+([IVXLC]+)\b", re.MULTILINE)

# Maximum forward jump accepted between consecutive headings. A larger real jump
# (e.g. a run of omitted sections) is accepted once two consecutive heading
# numbers beyond it are seen (resync), so one bad gap cannot freeze tracking.
MAX_SECTION_GAP = 8

# Structured docs and the label used for their numbered units.
SECTION_AWARE_DOCS = {
    "act_2025_full_text.pdf": "Section",
    "income_tax_rules_2026.pdf": "Rule",
}


def heading_title(before: str) -> str:
    """
    The marginal heading printed just before a section number, e.g.
    "New tax regime for individuals, Hindu undivided family and others." before
    "202. (1) ...". Titles end with a full stop and may wrap onto a second line.
    Returns "" when the preceding text does not look like a title.
    """
    lines = [ln.strip() for ln in before.rstrip().split("\n")]
    if not lines or not lines[-1].endswith("."):
        return ""
    title = lines[-1]
    prev = lines[-2] if len(lines) > 1 else ""
    if prev[:1].isupper() and not re.search(r"[.;:—]$", prev) and len(prev) + len(title) < 200:
        # wrapped title; "pro-" + "vident fund" is a hyphenated line break
        title = prev[:-1] + title if prev.endswith("-") else f"{prev} {title}"
    title = re.sub(r"^\d+\[", "", title)  # amendment marker, e.g. "3[Fee for default ..."
    if not title[:1].isupper() or len(title) > 200 or len(title) < 6:
        return ""
    return title


def schedule_title(after: str) -> str:
    """
    Title of a schedule from the text after "SCHEDULE XV": the "[See section N]"
    line plus the following upper-case title lines, e.g.
    "Deductions ... (see section 123)". Returns "" when absent.
    """
    lines = [ln.strip() for ln in after.split("\n")[1:8]]
    see = ""
    if lines and lines[0].startswith("[See"):
        see = lines.pop(0).strip("[]")
    words = []
    for ln in lines:
        if not ln or not re.search(r"[A-Z]", ln) or ln != ln.upper():
            break
        words.append(ln[:-1] if ln.endswith("-") else ln + " ")
    title = re.sub(r"\s+", " ", "".join(words)).strip().capitalize()
    if see:
        title = f"{title} ({see[0].lower()}{see[1:]})" if title else see
    return title[:200]


def assign_section_refs(chunks: list[dict], unit_label: str,
                        max_gap: int = MAX_SECTION_GAP) -> list[dict]:
    """
    Set chunk["section_ref"] (and chunk["section_title"], the heading text of
    that unit when it could be read) for an ordered list of chunks from ONE document.

    A chunk gets the unit that is active at its start; if the chunk itself
    begins with an accepted heading, it gets that heading instead. Once a
    "SCHEDULE X" heading is seen, the schedule becomes the active unit.
    """
    last_num, last_suffix = 0, ""
    pending = None  # an out-of-range heading number awaiting confirmation (resync)
    current = ""
    in_schedules = False
    titles: dict[str, str] = {}
    prev_text = ""

    for chunk in chunks:
        text = chunk["text"]
        ref_at_start = current
        first_heading_ref = None

        events = []
        for m in SCHEDULE_PATTERN.finditer(text):
            events.append((m.start(), "schedule", m.group(1)))
        if not in_schedules:
            for m in HEADING_PATTERN.finditer(text):
                events.append((m.start(), "heading", m))
        events.sort(key=lambda e: e[0])

        for pos, kind, payload in events:
            if kind == "schedule":
                in_schedules = True
                current = f"Schedule {payload}"
                title = schedule_title(text[pos:])
                if title:
                    titles.setdefault(current, title)
            elif not in_schedules:
                num, suffix = int(payload.group(1)), payload.group(2)
                in_step = last_num < num <= last_num + max_gap
                lettered = num == last_num and suffix > last_suffix  # 80 → 80A
                resync = pending is not None and num == pending + 1   # two consecutive far headings
                if in_step or lettered or resync:
                    last_num, last_suffix, pending = num, suffix, None
                    current = f"{unit_label} {num}{suffix}"
                    before = text[:pos] if pos > 0 else prev_text[-300:]
                    title = heading_title(before)
                    if title:
                        titles.setdefault(current, title)
                else:
                    if num > last_num + max_gap:
                        pending = num
                    continue
            else:
                continue
            if first_heading_ref is None and pos < 5:
                first_heading_ref = current

        chunk["section_ref"] = first_heading_ref or ref_at_start or current
        prev_text = text

    for chunk in chunks:
        chunk["section_title"] = titles.get(chunk["section_ref"], "")
    return chunks


def index_text(chunk: dict) -> str:
    """
    Text that is embedded and BM25-indexed: the section ref and title, then the
    chunk text. Rate tables and long clauses rarely repeat their section's
    subject ("New tax regime ...", "Fee for default ..."), so the header gives
    them the vocabulary users ask with. The stored/displayed text is unchanged.
    """
    header = " — ".join(p for p in (chunk.get("section_ref", ""), chunk.get("section_title", "")) if p)
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


# ── Splitting ─────────────────────────────────────────────────────────────

def split_recursive(text: str, source_doc: str,
                    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
                    page_starts: list[int] | None = None) -> list[dict]:
    """
    Split with RecursiveCharacterTextSplitter. If page_starts (character
    offsets where each page begins in `text`) is given, each chunk records the
    1-based page on which it starts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    cursor = 0
    for i, raw in enumerate(splitter.split_text(text)):
        chunk_text = raw.strip()
        if len(chunk_text) < 20:
            continue

        page = None
        if page_starts:
            pos = text.find(chunk_text[:60], max(0, cursor - chunk_overlap - 60))
            if pos >= 0:
                cursor = pos
                page = sum(1 for s in page_starts if s <= pos)

        chunk_id = hashlib.md5(
            f"{source_doc}:chunk_{i}:{chunk_text[:100]}".encode()
        ).hexdigest()[:12]

        chunk = {
            "id": chunk_id,
            "text": chunk_text,
            "source_doc": source_doc,
            "section_ref": "",
            # source_type, topic, tax_year_scope filled by tag_metadata
            "source_type": "",
            "topic": "",
            "tax_year_scope": "",
        }
        if page is not None:
            chunk["page"] = page
        chunks.append(chunk)
    return chunks


# ── Main chunking pipeline ───────────────────────────────────────────────

def chunk_document(filepath: Path) -> list[dict]:
    """Extract, normalise, split, assign section refs (Act/Rules) and tag one document."""
    suffix = filepath.suffix.lower()
    page_starts = None
    if suffix == ".pdf":
        pages = [normalize_text(p) for p in extract_pdf_pages(filepath)]
        page_starts, offset = [], 0
        for p in pages:
            page_starts.append(offset)
            offset += len(p) + 2  # "\n\n" separator
        text = "\n\n".join(pages)
    elif suffix in (".md", ".txt"):
        text = normalize_text(extract_markdown_text(filepath))
    else:
        print(f"  ⚠ Skipping unsupported file type: {filepath}")
        return []

    if not text.strip():
        print(f"  ⚠ No text extracted from: {filepath}")
        return []

    try:
        source_doc = str(filepath.relative_to(DATA_RAW_DIR))
    except ValueError:
        source_doc = filepath.name

    chunks = split_recursive(text, source_doc, page_starts=page_starts)
    unit = SECTION_AWARE_DOCS.get(filepath.name)
    if unit:
        assign_section_refs(chunks, unit)
    with_ref = sum(1 for c in chunks if c["section_ref"])
    print(f"  ✅ {len(chunks)} chunks from {source_doc} ({with_ref} with section_ref)")

    for chunk in chunks:
        tag_chunk(chunk)
    return chunks


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    """Drop chunks whose (source_doc, text) duplicates an earlier chunk."""
    seen, out = set(), []
    for c in chunks:
        key = (c["source_doc"], c["text"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def write_chunks(chunks: list[dict], path: Path = CHUNKS_JSONL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def collect_source_files(raw_dir: Path = DATA_RAW_DIR) -> list[Path]:
    files = sorted(raw_dir.glob("*.pdf"))
    circulars = raw_dir / "circulars"
    if circulars.exists():
        files += sorted(circulars.glob("*.pdf"))
    # SOURCES.md is documentation, not a source doc
    files += [md for md in sorted(raw_dir.glob("*.md")) if md.name != "SOURCES.md"]
    return files


def run_chunking_pipeline():
    """Process all documents in data/raw/ and output chunks.jsonl."""
    print("=" * 60)
    print("  Chunking Pipeline")
    print("=" * 60)

    files = collect_source_files()
    print(f"\n  Found {len(files)} files to process:\n")

    all_chunks = []
    for filepath in files:
        all_chunks.extend(chunk_document(filepath))
    all_chunks = dedupe_chunks(all_chunks)
    write_chunks(all_chunks)

    print(f"\n{'=' * 60}")
    print(f"  Total: {len(all_chunks)} chunks → {CHUNKS_JSONL_PATH}")
    print(f"{'=' * 60}")
    return all_chunks


def upgrade_existing_chunks(path: Path = CHUNKS_JSONL_PATH) -> list[dict]:
    """
    Apply the current normalisation, section tracking, tagging and dedup to an
    existing chunks.jsonl WITHOUT the raw PDFs (chunk boundaries and ids are
    kept; page numbers cannot be recovered). Used when data/raw/*.pdf is absent.
    """
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    for c in chunks:
        c["text"] = normalize_text(c["text"])
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["source_doc"], []).append(c)
    for doc, doc_chunks in by_doc.items():
        unit = SECTION_AWARE_DOCS.get(Path(doc).name)
        if unit:
            assign_section_refs(doc_chunks, unit)
    for c in chunks:
        tag_chunk(c)

    chunks = dedupe_chunks(chunks)
    write_chunks(chunks, path)
    print(f"  Upgraded {len(chunks)} chunks in {path}")
    return chunks


if __name__ == "__main__":
    import sys

    if "--upgrade-existing" in sys.argv:
        upgrade_existing_chunks()
    else:
        run_chunking_pipeline()
