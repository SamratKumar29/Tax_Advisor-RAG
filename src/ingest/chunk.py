"""
Chunking pipeline for Indian Tax Advisor RAG.

- Parses PDFs with pypdf, Markdown with plain text reading.
- Uses section/clause-aware splitting for structured legal docs (Act, Rules).
- Falls back to recursive character splitting for unstructured docs (FAQs, circulars).
- Outputs data/processed/chunks.jsonl.
"""

import json
import re
import hashlib
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, CHUNKS_JSONL_PATH
from src.ingest.tag_metadata import tag_chunk


# ── PDF text extraction ───────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file using pypdf."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_markdown_text(md_path: Path) -> str:
    """Read a Markdown file as plain text."""
    return md_path.read_text(encoding="utf-8")


# ── Section-aware splitting (for Act / Rules) ────────────────────────────

# Pattern matches lines like "Section 123." or "SECTION 123." or "Section 123 -"
SECTION_PATTERN = re.compile(
    r"^\s*(Section|SECTION)\s+\d+[A-Z]?\s*[\.\-\—]",
    re.MULTILINE
)

# Pattern for chapter boundaries
CHAPTER_PATTERN = re.compile(
    r"^\s*(CHAPTER|Chapter)\s+[IVXLCDM\d]+",
    re.MULTILINE
)


def split_by_sections(text: str, source_doc: str) -> list[dict]:
    """
    Split legal text on Section N. boundaries.

    Each chunk is one complete section (or the text between two section headers).
    This avoids splitting mid-clause, which is critical for legal documents.
    """
    # Find all section boundary positions
    boundaries = [m.start() for m in SECTION_PATTERN.finditer(text)]

    if len(boundaries) < 5:
        # Not enough section markers — not a structured legal document,
        # fall back to recursive splitting
        return []

    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 20:
            continue  # skip empty/trivial chunks

        # Extract section reference from the chunk header
        section_match = re.match(r"(Section|SECTION)\s+(\d+[A-Z]?)", chunk_text)
        section_ref = f"Section {section_match.group(2)}" if section_match else ""

        chunk_id = hashlib.md5(
            f"{source_doc}:{section_ref}:{chunk_text[:100]}".encode()
        ).hexdigest()[:12]

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source_doc": source_doc,
            "section_ref": section_ref,
            # source_type, topic, tax_year_scope filled by tag_metadata
            "source_type": "",
            "topic": "",
            "tax_year_scope": "",
        })

    return chunks


# ── Fallback recursive splitting ─────────────────────────────────────────

def split_recursive(text: str, source_doc: str,
                    chunk_size: int = 1500, chunk_overlap: int = 150) -> list[dict]:
    """
    Fall back to RecursiveCharacterTextSplitter for docs without clear
    section structure (FAQs, press releases, circulars).

    ~500 tokens ≈ ~1500 characters (rough 1:3 ratio).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 20:
            continue

        chunk_id = hashlib.md5(
            f"{source_doc}:chunk_{i}:{chunk_text[:100]}".encode()
        ).hexdigest()[:12]

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source_doc": source_doc,
            "section_ref": "",
            "source_type": "",
            "topic": "",
            "tax_year_scope": "",
        })

    return chunks


# ── Main chunking pipeline ───────────────────────────────────────────────

# Documents where we should try section-aware splitting first
SECTION_AWARE_DOCS = {
    "act_2025_full_text.pdf",
    "income_tax_rules_2026.pdf",
}


def chunk_document(filepath: Path) -> list[dict]:
    """
    Chunk a single document. Uses section-aware splitting for Act/Rules,
    falls back to recursive splitting for everything else.
    """
    filename = filepath.name

    # Extract text based on file type
    if filepath.suffix.lower() == ".pdf":
        text = extract_pdf_text(filepath)
    elif filepath.suffix.lower() in (".md", ".txt"):
        text = extract_markdown_text(filepath)
    else:
        print(f"  ⚠ Skipping unsupported file type: {filepath}")
        return []

    if not text.strip():
        print(f"  ⚠ No text extracted from: {filepath}")
        return []

    # Determine the source_doc name (relative to data/raw/)
    try:
        source_doc = str(filepath.relative_to(DATA_RAW_DIR))
    except ValueError:
        source_doc = filename

    # Try section-aware splitting for structured legal docs
    chunks = []
    if filename in SECTION_AWARE_DOCS:
        chunks = split_by_sections(text, source_doc)
        if chunks:
            print(f"  ✅ Section-aware: {len(chunks)} chunks from {source_doc}")

    # Fall back to recursive splitting
    if not chunks:
        chunks = split_recursive(text, source_doc)
        print(f"  ✅ Recursive: {len(chunks)} chunks from {source_doc}")

    # Apply metadata tagging to each chunk
    for chunk in chunks:
        tag_chunk(chunk)

    return chunks


def run_chunking_pipeline():
    """
    Process all documents in data/raw/ and output chunks.jsonl.
    """
    print("=" * 60)
    print("  Chunking Pipeline")
    print("=" * 60)

    all_chunks = []

    # Collect all files to process
    files_to_process = []

    # PDFs in data/raw/ (top-level)
    for pdf in sorted(DATA_RAW_DIR.glob("*.pdf")):
        files_to_process.append(pdf)

    # PDFs in data/raw/circulars/
    circulars_dir = DATA_RAW_DIR / "circulars"
    if circulars_dir.exists():
        for pdf in sorted(circulars_dir.glob("*.pdf")):
            files_to_process.append(pdf)

    # Markdown files (section mapping, etc.)
    for md in sorted(DATA_RAW_DIR.glob("*.md")):
        # Skip SOURCES.md — it's documentation, not a source doc
        if md.name == "SOURCES.md":
            continue
        files_to_process.append(md)

    print(f"\n  Found {len(files_to_process)} files to process:\n")

    for filepath in files_to_process:
        chunks = chunk_document(filepath)
        all_chunks.extend(chunks)

    # Write output
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Total: {len(all_chunks)} chunks → {CHUNKS_JSONL_PATH}")
    print(f"{'=' * 60}")

    return all_chunks


if __name__ == "__main__":
    run_chunking_pipeline()
