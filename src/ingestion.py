"""
src/ingestion.py
================
Task 1 – Advanced Ingestion & Metadata Enrichment

Handles:
  1.1  route_and_parse()  →  parse all 5 file types into clean text
  1.2  build_documents()  →  attach rich metadata to every chunk
       + build_vectorstore() → embed chunks and persist ChromaDB index

File-type routing
-----------------
  .pdf   →  pdfplumber  (layout-aware text extraction)
  .md    →  plain read  (text is already clean)
  .html  →  BeautifulSoup (strip tags, keep semantic text)
  .json  →  json.dumps  (pretty-print for embedding)
  .xlsx  →  pandas      (sheet-by-sheet, labeled table strings)

Metadata fields per chunk (persisted in ChromaDB)
--------------------------------------------------
  document_category  – domain label assigned by categorise()
  source             – original file name
  file_type          – extension (.pdf, .md, …)
  author             – extracted or inferred
  creation_date      – extracted or inferred
  language           – detected (simple heuristic, extend with langdetect)
  chunk_index        – position of this chunk within the document
  total_chunks       – total chunks produced from this document
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ── LangChain imports (install: pip install langchain langchain-GoogleGenerativeAIEmbeddings chromadb) ──
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent.parent / "data"
CHROMA_DIR  = Path(__file__).parent.parent / "chroma_db"
COLLECTION  = "financial_research"

CHUNK_SIZE    = 800   # characters
CHUNK_OVERLAP = 150

# Map filename keywords → document_category labels
# Edit / extend this dict to match your domain files.
CATEGORY_MAP: dict[str, str] = {
    "analyst_report":   "analyst_report",
    "earnings_summary": "earnings_summary",
    "macro_indicators": "macroeconomic_data",
    "stock_data":       "stock_and_financial_data",
    "news_article":     "financial_news",
}

# Static metadata that would normally come from a CMS / file header
STATIC_METADATA: dict[str, dict[str, str]] = {
    "analyst_report.pdf": {
        "author":        "Brian Holloway, CFA – Meridian Capital Research",
        "creation_date": "2025-02-14",
        "language":      "en",
    },
    "earnings_summary.md": {
        "author":        "Investor Relations Team – TechVision Corp",
        "creation_date": "2025-02-12",
        "language":      "en",
    },
    "macro_indicators.json": {
        "author":        "World Economic Research Institute",
        "creation_date": "2025-01-15",
        "language":      "en",
    },
    "stock_data.xlsx": {
        "author":        "Meridian Capital Research – Quantitative Team",
        "creation_date": "2025-02-14",
        "language":      "en",
    },
    "news_article.html": {
        "author":        "Sarah Mitchell – FinanceWatch Daily",
        "creation_date": "2025-02-13",
        "language":      "en",
    },
}


# ── 1.1  Parsing helpers ─────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber for layout-aware extraction.
    Falls back to pypdf if pdfplumber returns empty text (e.g. scanned pages).
    """
    text_parts: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(x_tolerance=2, y_tolerance=3)

            # Extract any tables on this page as structured text
            for table in page.extract_tables():
                if not table:
                    continue
                # Convert table rows to tab-separated strings
                table_lines = []
                for row in table:
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    table_lines.append("\t".join(clean_row))
                page_text = (page_text or "") + "\n\n[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]\n"

            if page_text and page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text.strip()}")

    if text_parts:
        return "\n\n".join(text_parts)

    # Fallback: pypdf
    log.warning("%s: pdfplumber returned no text — falling back to pypdf", path.name)
    reader = PdfReader(str(path))
    return "\n\n".join(
        f"[Page {i+1}]\n{page.extract_text() or ''}"
        for i, page in enumerate(reader.pages)
    )


def _parse_markdown(path: Path) -> str:
    """
    Read Markdown as plain text.
    The raw Markdown is kept (with headers/bold markers) because
    RecursiveCharacterTextSplitter will split on header boundaries by default.
    """
    return path.read_text(encoding="utf-8")


def _parse_html(path: Path) -> str:
    """
    Strip HTML tags with BeautifulSoup.
    Preserves visible text content while removing scripts, styles, and nav chrome.
    """
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove elements that are never readable content
    for tag in soup(["script", "style", "head", "nav", "footer", "noscript", "meta"]):
        tag.decompose()

    # Walk the tree and collect text from semantic elements in document order
    blocks: list[str] = []
    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "td", "th", "caption",
         "blockquote", "pre", "code"]
    ):
        text = element.get_text(separator=" ", strip=True)
        if text:
            tag_name = element.name
            if tag_name in ("h1", "h2", "h3"):
                blocks.append(f"\n## {text}\n")
            elif tag_name in ("h4", "h5", "h6"):
                blocks.append(f"\n### {text}\n")
            else:
                blocks.append(text)

    # Remove duplicate blank lines
    content = "\n".join(blocks)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _parse_json(path: Path) -> str:
    """
    Load JSON and convert to a pretty-printed string.
    For deeply nested structures the pretty-print makes parent/child
    relationships visible in the embedding context.
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _parse_xlsx(path: Path) -> str:
    """
    Read all sheets from an Excel workbook.
    Each sheet is converted to a string table and labelled clearly.
    """
    xl = pd.ExcelFile(str(path))
    parts: list[str] = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name, dtype=str)
        df = df.fillna("")          # replace NaN with empty string
        df = df.loc[                # drop fully-empty rows / columns
            ~(df == "").all(axis=1),
            ~(df == "").all(axis=0)
        ]
        if df.empty:
            continue
        table_str = df.to_string(index=False)
        parts.append(f"[Sheet: {sheet_name}]\n{table_str}")

    return "\n\n".join(parts)


# ── Router ───────────────────────────────────────────────────────────────────────

_PARSERS = {
    ".pdf":  _parse_pdf,
    ".md":   _parse_markdown,
    ".html": _parse_html,
    ".json": _parse_json,
    ".xlsx": _parse_xlsx,
}


def route_and_parse(path: Path) -> str:
    """
    Dispatch *path* to the appropriate parser based on file extension.
    Returns a clean text string ready for chunking.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)

    if parser is None:
        supported = ", ".join(_PARSERS)
        raise ValueError(
            f"Unsupported file type '{ext}' for {path.name}. "
            f"Supported: {supported}"
        )

    log.info("Parsing %-30s [%s]", path.name, ext)
    text = parser(path)

    if not text or not text.strip():
        log.warning("%s: parser returned empty text", path.name)

    return text


# ── 1.2  Metadata enrichment ─────────────────────────────────────────────────────

def categorise(filename: str) -> str:
    """
    Return a document_category string for *filename*.
    Matches against CATEGORY_MAP keys (substring match, case-insensitive).
    Falls back to 'uncategorised' if no key matches.
    """
    lower = filename.lower()
    for keyword, category in CATEGORY_MAP.items():
        if keyword in lower:
            return category
    return "uncategorised"


def _base_metadata(path: Path) -> dict[str, Any]:
    """
    Build the base metadata dict for a file.
    Merges STATIC_METADATA (author, date, language) with derived fields.
    """
    static = STATIC_METADATA.get(path.name, {})
    return {
        "document_category": categorise(path.name),
        "source":            path.name,
        "file_type":         path.suffix.lower(),
        "author":            static.get("author", "Unknown"),
        "creation_date":     static.get("creation_date", "Unknown"),
        "language":          static.get("language", "en"),
    }


# ── Chunking ─────────────────────────────────────────────────────────────────────

def _make_splitter(file_type: str) -> RecursiveCharacterTextSplitter:
    """
    Return a text splitter tuned to the file type.

    - Markdown / HTML  →  split on headers first, then paragraphs
    - JSON             →  split on object boundaries  `},` and `{`
    - PDF / XLSX       →  default paragraph/sentence splits
    """
    if file_type in (".md", ".html"):
        # LangChain's Markdown-aware separators
        separators = [
            "\n## ", "\n### ", "\n#### ",
            "\n\n", "\n", ". ", " ", ""
        ]
    elif file_type == ".json":
        separators = ["},\n  {", "},\n{", "\n\n", "\n", " ", ""]
    else:
        separators = ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )


def build_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Parse every supported file in *data_dir*, chunk each one, and
    return a list of LangChain Document objects with full metadata.

    Each Document has:
        page_content  – the chunk text
        metadata      – dict with document_category, source, file_type,
                        author, creation_date, language,
                        chunk_index, total_chunks
    """
    supported_exts = set(_PARSERS.keys())
    all_docs: list[Document] = []

    files = sorted(
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_exts
    )

    if not files:
        log.warning("No supported files found in %s", data_dir)
        return []

    for path in files:
        # ── Parse ──────────────────────────────────────────────────
        try:
            raw_text = route_and_parse(path)
        except Exception as exc:
            log.error("Failed to parse %s: %s", path.name, exc)
            continue

        if not raw_text.strip():
            log.warning("Skipping %s: empty after parsing", path.name)
            continue

        # ── Chunk ──────────────────────────────────────────────────
        splitter = _make_splitter(path.suffix.lower())
        chunks: list[str] = splitter.split_text(raw_text)
        total = len(chunks)

        # ── Metadata enrichment ────────────────────────────────────
        base_meta = _base_metadata(path)

        for idx, chunk_text in enumerate(chunks):
            meta = {
                **base_meta,
                "chunk_index":  idx,
                "total_chunks": total,
            }
            doc = Document(page_content=chunk_text, metadata=meta)
            all_docs.append(doc)

        log.info(
            "  %-30s → %3d chunks | category: %s",
            path.name, total, base_meta["document_category"]
        )

    log.info("Total documents (chunks) ready for embedding: %d", len(all_docs))
    return all_docs


# ── ChromaDB build / load ─────────────────────────────────────────────────────────

def build_vectorstore(
    data_dir:   Path = DATA_DIR,
    chroma_dir: Path = CHROMA_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Build (or load) a persisted ChromaDB vector store.

    Strategy
    --------
    - If *chroma_dir* already contains a built index AND force_rebuild=False,
      load and return the existing store instantly (no embedding API calls).
    - Otherwise parse all files, embed every chunk, persist to *chroma_dir*.

    Parameters
    ----------
    data_dir      : directory that contains the source documents
    chroma_dir    : directory where ChromaDB will persist its files
    force_rebuild : set True to re-ingest even if the index already exists
    """


    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if index already exists ─────────────────────────────
    index_exists = any(chroma_dir.iterdir()) if chroma_dir.exists() else False

    if index_exists and not force_rebuild:
        log.info("Loading existing ChromaDB index from %s …", chroma_dir)
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )
        count = vectorstore._collection.count()
        log.info("Loaded %d chunks from existing index.", count)
        return vectorstore

    # ── Build from scratch ─────────────────────────────────────────
    log.info("Building ChromaDB index from %s …", data_dir)
    docs = build_documents(data_dir)

    if not docs:
        raise RuntimeError("No documents were produced. Check your data/ directory.")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(chroma_dir),
    )

    log.info("ChromaDB index built and persisted → %s", chroma_dir)
    _verify_metadata(vectorstore)
    return vectorstore


def _verify_metadata(vectorstore: Chroma) -> None:
    """
    Spot-check: print the metadata of the first stored chunk.
    Confirms that document_category, source, author, etc. are present.
    """
    sample = vectorstore._collection.get(limit=1, include=["metadatas", "documents"])
    if sample and sample.get("metadatas"):
        meta = sample["metadatas"][0]
        snippet = (sample["documents"][0] or "")[:120].replace("\n", " ")
        log.info("── Metadata verification ──────────────────────────────")
        for k, v in meta.items():
            log.info("  %-20s : %s", k, v)
        log.info("  %-20s : %s…", "content_preview", snippet)
        log.info("────────────────────────────────────────────────────────")

        # Assert all required fields are present
        required = {"document_category", "source", "file_type",
                    "author", "creation_date", "language"}
        missing = required - set(meta.keys())
        if missing:
            log.error("MISSING metadata fields: %s", missing)
        else:
            log.info("All required metadata fields present ✓")


# ── CLI entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force re-ingestion even if the index already exists"
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help=f"Path to data directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--chroma-dir", default=str(CHROMA_DIR),
        help=f"Path to ChromaDB directory (default: {CHROMA_DIR})"
    )
    args = parser.parse_args()

    build_vectorstore(
        data_dir=Path(args.data_dir),
        chroma_dir=Path(args.chroma_dir),
        force_rebuild=args.rebuild,
    )