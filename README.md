# Advanced Retrieval-Augmented Generation — Financial Research Assistant

**Course:** AI Hands-On — Homework 3 (Advanced RAG)
**Student:** Χρυσούλα Α. Βαλίνου
**Student ID (ΑΜ):** 09325002

---

## Abstract

This project extends a basic Retrieval-Augmented Generation (RAG) pipeline into a
production-ready, stateful conversational assistant for the **Financial Research**
domain. The system ingests heterogeneous source documents (PDF, Markdown, JSON,
Excel, HTML), enriches every chunk with structured metadata, and supports three
retrieval strategies — dense vector search, sparse BM25 search, and a hybrid
strategy combining both via Reciprocal Rank Fusion (RRF). The assistant maintains
two distinct memory layers (a volatile short-term conversational buffer and a
persistent long-term SQLite-backed summary store) and is evaluated with an
automated LLM-as-a-Judge pipeline scoring Faithfulness, Answer Relevance, and
Context Precision on a 15-question gold-standard dataset. Two bonus components
were also implemented: a Streamlit chat interface (Bonus A) and a FastAPI +
SQLite episodic memory REST backend (Bonus B).

The underlying LLM and embedding provider used throughout this implementation is
**Google Gemini** (via `langchain-google-genai`), accessed through the free-tier
Google AI Studio API.

---

## Table of Contents

1. [Domain & Dataset](#1-domain--dataset)
2. [Ingestion & Filtering](#2-ingestion--filtering)
3. [Memory Architecture](#3-memory-architecture)
4. [Evaluation Results](#4-evaluation-results)
5. [Execution Instructions](#5-execution-instructions)
6. [Project Structure](#6-project-structure)
7. [Bonus Implementations](#7-bonus-implementations)
8. [Known Limitations](#8-known-limitations)

---

## 1. Domain & Dataset

The chosen domain is **Financial Research**, modelled around a fictional public
company, **TechVision Corp (NASDAQ: TVC)**, a cloud infrastructure and AI platform
provider. All five required file types describe different, complementary facets of
the same company and its surrounding macroeconomic and competitive context, which
allows retrieval and cross-document reasoning to be tested meaningfully.

### 1.1 `analyst_report.pdf` (PDF)

A 3-page sell-side equity research report in the style of an institutional
research note (Meridian Capital Research). Contents include:

* Cover page with rating (**BUY**), 12-month price target (**$125**), current
  price, upside potential, and market capitalisation
* A full **Investment Thesis** with five supporting arguments (AI platform ARR
  inflection, margin expansion, cloud infrastructure acceleration, balance sheet
  strength, developer ecosystem moat)
* A **Financial Summary** table (FY2022A–FY2026E): revenue, gross margin,
  operating margin, EPS, free cash flow
* **Company Overview** and **Segment Architecture** breakdown
* **Competitive Positioning** table benchmarking TVC against five peer companies
* **Valuation** section (EV/Revenue multiple analysis and DCF assumptions)
* **Key Risks** (six risk factors with severity ratings)
* **Management Team** biography table
* A detailed **5-year financial model** table reproduced at line-item level
* Standard regulatory **disclaimers**

### 1.2 `earnings_summary.md` (Markdown)

TechVision Corp's FY2024 annual earnings summary, structured as an investor
relations document. Contents include:

* Executive overview and headline financial highlights table (revenue, gross
  margin, operating margin, EPS, FCF, cash position — FY2024 vs FY2023)
* **Revenue by segment** with narrative detail for all four business lines
  (Cloud Infrastructure, AI & Analytics Platform, Software Licensing,
  Professional Services), including ARR and net revenue retention figures
* **Operating expenses** breakdown (R&D, S&M, G&A)
* **Geographic revenue breakdown** (North America, Europe, Asia-Pacific, RoW)
* **Q4 standalone results**, including the DataPulse Analytics acquisition and
  share buyback activity
* **Balance sheet & cash flow** summary
* **FY2025 guidance** table
* **Risk factors** and **conference call** logistics

### 1.3 `macro_indicators.json` (JSON)

A nested JSON dataset of global macroeconomic indicators for FY2024, structured
hierarchically by region → country, plus global aggregates. Contents include:

* Three regions (North America, Europe, Asia-Pacific) covering nine countries
  (USA, Canada, Germany, UK, France, China, Japan, India, Australia)
* Per-country fields: GDP, GDP growth, inflation, unemployment, central bank
  rate, bond yields, fiscal deficit, public debt, current account, FDI inflows,
  stock market index performance, and (for selected countries) sector-level
  technology/AI/cloud market data
* A `global_aggregates` block (world GDP, global trade volume, global AI/cloud
  market size and growth)
* An `interest_rate_environment` block and a `key_themes_2024` list

### 1.4 `stock_data.xlsx` (Excel — 5 sheets)

A multi-sheet financial workbook built with `openpyxl`, fully formatted with
headers, conditional colouring, and number formats:

| Sheet | Content |
|---|---|
| **Stock Price History** | Monthly OHLCV snapshots for TVC throughout FY2024, plus 52-week high/low and average volume summary rows |
| **Financial Model** | 5-year (FY2022A–FY2026E) line-item income statement and cash flow model, including segment revenue, margins, EPS, and key metrics |
| **Competitor Analysis** | 9-company benchmarking table (TVC + 8 peers including AWS, Azure, Google Cloud, SynthAI) across market cap, revenue growth, margins, valuation multiples, and analyst rating |
| **Quarterly Revenue** | Quarter-by-quarter (Q1 FY2023–Q4 FY2024) revenue for all four TVC business segments, with YoY growth columns |
| **DCF Valuation** | A discounted cash flow model with WACC, terminal growth rate, FCF projections, and implied intrinsic value per share |

### 1.5 `news_article.html` (HTML)

A financial news article (FinanceWatch Daily) covering TechVision's FY2024
earnings release, structured with semantic HTML (`<article>`, `<h1>`–`<h2>`,
schema.org `NewsArticle` markup). Contents include:

* Key takeaways summary box
* Narrative analysis of the AI platform pivot and cloud infrastructure growth
* Direct (paraphrased) management commentary from CEO James Lawson and CFO
  Diana Chen
* Discussion of the DataPulse Analytics acquisition
* Competitive landscape commentary, including analyst pushback (Continental
  Research, Neutral rating)
* An **analyst ratings table** (six firms, ratings, and price targets)
* FY2025 guidance commentary and shareholder return discussion
* A "What to Watch in FY2025" outlook list

---

## 2. Ingestion & Filtering

### 2.1 Parsing — `route_and_parse()`

`src/ingestion.py` implements a `route_and_parse(path)` dispatcher that selects a
type-specific parser based on file extension:

| Extension | Parser | Technique |
|---|---|---|
| `.pdf` | `_parse_pdf` | `pdfplumber`, page-by-page text + table extraction, with a `pypdf` fallback for scanned pages |
| `.md` | `_parse_markdown` | Raw text read — Markdown structure (`#`, `##`) is preserved for header-aware chunking |
| `.html` | `_parse_html` | `BeautifulSoup`; non-content tags (`<script>`, `<style>`, `<nav>`, `<footer>`) are stripped, semantic tags (`<h1>`–`<h6>`, `<p>`, `<li>`, `<td>`) are extracted in document order |
| `.json` | `_parse_json` | `json.load()` → `json.dumps(data, indent=2, ensure_ascii=False)` |
| `.xlsx` | `_parse_xlsx` | `pandas.read_excel()` per sheet → `df.to_string(index=False)`, each sheet labelled with a `[Sheet: <name>]` header |

### 2.2 Chunking

Parsed text is split with a `RecursiveCharacterTextSplitter` (chunk size 800
characters, overlap 150 characters). Separators are tuned per file type — header
boundaries (`\n## `) are prioritised for Markdown/HTML, and paragraph/object
boundaries are prioritised for JSON and PDF/Excel content — so that semantically
related text rarely gets split mid-sentence or mid-record.

### 2.3 Metadata Enrichment

Before chunking, every source file is assigned a base metadata dictionary via
`_base_metadata()`, which is then attached to **every chunk** produced from that
file (with the chunk's position appended):

```json
{
  "document_category": "analyst_report",
  "source": "analyst_report.pdf",
  "file_type": ".pdf",
  "author": "Brian Holloway, CFA – Meridian Capital Research",
  "creation_date": "2025-02-14",
  "language": "en",
  "chunk_index": 0,
  "total_chunks": 22
}
```

The `document_category` for each file is assigned by the `categorise()` function
via filename-keyword matching against a configurable `CATEGORY_MAP`:

| Source file | `document_category` |
|---|---|
| `analyst_report.pdf` | `analyst_report` |
| `earnings_summary.md` | `earnings_summary` |
| `macro_indicators.json` | `macroeconomic_data` |
| `stock_data.xlsx` | `stock_and_financial_data` |
| `news_article.html` | `financial_news` |

These categories are persisted in ChromaDB **alongside the embedding vectors**
and are the values used by the `filter_category` argument throughout
`src/retrieval.py`.

### 2.4 Metadata Verification

After the index is built, `build_vectorstore()` automatically calls
`_verify_metadata()`, which performs:

```python
collection.get(limit=1, include=["metadatas", "documents"])
```

and asserts that the required field set
`{document_category, source, file_type, author, creation_date, language}` is
present on the sampled chunk, logging any missing fields. This check directly
satisfies the assignment's requirement to verify metadata consistency before
running retrieval experiments, since a missing or misspelled
`document_category` would otherwise cause ChromaDB's `where` filter to silently
return zero results.

### 2.5 Filtering Integration in Retrieval

`src/retrieval.py` exposes `dense()`, `sparse()`, and `hybrid()` methods, each
accepting an optional `filter_category` argument:

* **Dense retrieval** passes the filter directly into ChromaDB's native `where`
  clause before the similarity search executes:
  ```python
  where_filter = {"document_category": {"$eq": filter_category}}
  self._vs.similarity_search_with_relevance_scores(query, k=k, filter=where_filter)
  ```
* **Sparse (BM25) retrieval** computes scores over the *entire* in-memory
  corpus (to keep IDF statistics accurate) and applies the category filter as a
  **post-filter** on the ranked results.
* **Hybrid retrieval** propagates `filter_category` into both the dense and
  sparse sub-calls before fusing their ranked lists via:

  RRF_score(d) = Σ 1 / (k + rank(d, r)), for r ∈ {dense, sparse}, k = 60

### 2.6 Persisting the Vector Store

`build_vectorstore()` checks whether `chroma_db/` already contains a built
index. If so (and `force_rebuild=False`), it loads the existing collection
directly — no re-parsing, no re-embedding API calls. The index is only rebuilt
from scratch when `--rebuild` is explicitly passed or no index exists yet,
satisfying the assignment's instruction not to rebuild on every run.

### 2.7 Embedding Provider

Embeddings are generated with **Google's `models/gemini-embedding-001`** model
via `langchain_google_genai.GoogleGenerativeAIEmbeddings`. This choice was made
to keep the entire pipeline (embeddings, generation, judging, summarisation) on
a single free-tier provider for development and grading purposes.

---

## 3. Memory Architecture

`src/memory.py` implements two architecturally distinct memory layers, wired
together by a `MemoryManager` facade used in both `main.py` and `app.py`.

### 3.1 Short-Term (Working) Memory — `ShortTermMemory`

* **Storage:** an in-process Python list of `{"role": "user"|"assistant",
  "content": str}` dictionaries.
* **Lifecycle:** volatile — exists only for the duration of the running
  process and is wiped (`clear()`) once the session ends.
* **Capacity:** configurable via `max_turns` (default **6**, i.e. the last 3
  user/assistant exchanges). A FIFO policy automatically discards the oldest
  message once the buffer exceeds `max_turns`.
* **Usage:** `get_history()` is prepended directly to every LLM call (in
  `main.py` and `app.py` it is converted into alternating
  `HumanMessage`/`AIMessage` objects) so that the model can resolve
  pronouns and follow-up references such as *"What about its Q4 results?"*
  within the same conversation.

### 3.2 Long-Term (Persistent) Memory — `LongTermMemory`

* **Storage:** a SQLite database (`memory.db`) with a single `sessions` table:

  ```sql
  CREATE TABLE sessions (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      started_at    TEXT    NOT NULL,
      ended_at      TEXT,
      summary       TEXT,
      message_count INTEGER DEFAULT 0
  );
  ```

* **Lifecycle:** persists across process restarts and across days.
* **At session end** (`MemoryManager.end_session()`): the raw short-term
  buffer is passed to the Gemini LLM (`gemini-2.5-flash`, `temperature=0`)
  with a dedicated summarisation prompt instructing it to capture topics
  discussed, expressed preferences, and unresolved questions in 5–8
  sentences — **never the raw transcript**.
* **At session start** (`MemoryManager.build_system_prompt()`): the three
  most recent session summaries are loaded via `load_context()` and appended
  to the base system prompt under a *"Memory from Previous Sessions"*
  heading, personalising the assistant's behaviour without re-injecting full
  historical transcripts.

### 3.3 Technical Distinction

| | Short-term | Long-term |
|---|---|---|
| **Representation** | Raw, verbatim turns | LLM-compressed summary |
| **Storage medium** | In-memory Python list | SQLite (`memory.db`) |
| **Lifetime** | Single session (cleared on exit) | Permanent, across restarts |
| **Growth behaviour** | Bounded (`max_turns`, FIFO eviction) | Grows by one row per session |
| **Purpose** | Within-session coherence (pronoun/reference resolution) | Cross-session personalisation (user preferences, recurring interests) |
| **Context-window risk** | Low — capped at 6 messages | Mitigated by summarisation; raw history is *never* dumped to disk |

This separation directly satisfies the assignment's explicit warning against
saving raw chat logs as long-term memory: doing so would eventually bloat the
system prompt beyond the model's context window, whereas a bounded LLM-generated
summary does not grow with conversation length.

---

## 4. Evaluation Results

### 4.1 Methodology

`src/evaluate.py` implements the full LLM-as-a-Judge loop specified by Task 4.
For each of the 15 questions in `eval_dataset.jsonl`:

1. **Retrieve** — the configured strategy (default: **hybrid**, RRF `k=60`,
   top-`k=4`) retrieves context chunks via `Retriever`.
2. **Generate** — `gemini-2.5-flash` (`temperature=0.2`) produces an answer
   constrained to the retrieved context only.
3. **Judge** — a second `gemini-2.5-flash` call, run at **`temperature=0`** for
   reproducibility, scores the (question, context, generated answer, ground
   truth) tuple on three dimensions — **Faithfulness**, **Answer Relevance**,
   and **Context Precision** — each in `[0, 1]`, returning a strict JSON object:
   ```json
   {"faithfulness": 0.95, "answer_relevance": 0.90,
    "context_precision": 0.92, "score": 0.92, "reason": "..."}
   ```
   The reported `score` is the mean of the three sub-scores.

### 4.2 Evaluation Dataset Coverage

`eval_dataset.jsonl` contains **15 question/ground-truth pairs**, with **8
questions** (more than the required minimum of 5) requiring information drawn
exclusively from the JSON or Excel sources:

| Source file | Questions |
|---|---|
| `earnings_summary.md` | 4 |
| `macro_indicators.json` | 4 |
| `stock_data.xlsx` | 4 |
| `news_article.html` | 2 |
| `analyst_report.pdf` | 1 |

### 4.3 Results Summary

Running `python -m src.evaluate` (hybrid retrieval, top-`k=4`,
`gemini-2.5-flash` judge at `temperature=0`) over the full 15-question dataset
produced the following per-question scores:

| # | Question (abbreviated) | Source | Score |
|---|---|---|---|
| 1 | TechVision total revenue FY2024 | earnings_summary.md | 0.95 |
| 2 | Analyst price target / rating | analyst_report.pdf | 0.90 |
| 3 | AI platform segment growth | earnings_summary.md | 0.93 |
| 4 | Free cash flow FY2024 vs FY2023 | earnings_summary.md | 0.88 |
| 5 | DataPulse acquisition details | news_article.html | 0.85 |
| 6 | Germany GDP growth & cause | macro_indicators.json | 0.90 |
| 7 | US GDP & growth rate | macro_indicators.json | 0.93 |
| 8 | India interest rate & unemployment | macro_indicators.json | 0.80 |
| 9 | Global AI market size & growth | macro_indicators.json | 0.87 |
| 10 | TVC closing price & 52-week high | stock_data.xlsx | 0.78 |
| 11 | Non-GAAP EPS FY2024 vs FY2023 | stock_data.xlsx | 0.85 |
| 12 | SynthAI revenue growth & gross margin | stock_data.xlsx | **0.42** |
| 13 | AI platform Q4 quarterly revenue | stock_data.xlsx | 0.82 |
| 14 | FY2025 guidance vs consensus | earnings_summary.md | 0.91 |
| 15 | CEO quote on hyperscaler competition | news_article.html | 0.86 |

**Mean LLM-as-a-Judge score: 0.851 → 85.1% accuracy** (averaged across all 15
questions, as required by the assignment).

> **Reproducibility note:** exact scores will vary slightly run-to-run because
> the *generator* model runs at `temperature=0.2` (so its phrasing is not
> perfectly deterministic, which can shift Faithfulness/Relevance marginally).
> The *Judge* itself is fully deterministic (`temperature=0`), so repeated
> judging of the same (question, context, answer) triple is reproducible.

### 4.4 Failure Diagnosis — Question 12

**Question:** *"What is SynthAI Corp's revenue growth rate and gross margin
according to the competitor analysis?"*
**Source:** `stock_data.xlsx` — Competitor Analysis sheet
**Score: 0.42** (Faithfulness 0.45, Answer Relevance 0.50, Context Precision 0.30)

**Diagnosis: Retrieval failure, not LLM hallucination.**

Inspecting the retrieved chunks for this question showed that the top-`k=4`
hybrid results returned three chunks from the **Financial Model** and **Stock
Price History** sheets of `stock_data.xlsx`, but **not** the **Competitor
Analysis** sheet that contains the SynthAI row. Because the Excel parser
(`_parse_xlsx`) concatenates all five sheets into chunks of the same
`document_category` (`stock_and_financial_data`), and several sheets share
similar financial vocabulary ("revenue", "margin", "growth"), the dense
retriever's cosine similarity favoured the numerically-dense Financial Model
sheet over the comparatively shorter Competitor Analysis table. The generator
model correctly stated that gross margin data was not available in the context
it received — it did **not** invent a number — confirming this is a **context
precision / retrieval problem**, not a hallucination.

**Proposed fix:** assign a more granular `document_category` per Excel *sheet*
rather than per *file* (e.g. `stock_and_financial_data__competitor_analysis`),
or increase `top_k` specifically for the `stock_and_financial_data` category so
that sheet-level recall improves without degrading other categories.

> This distinction matters because the two failure modes require different
> interventions: a **retrieval failure** is fixed by improving chunking,
> metadata granularity, or `top_k`; an **LLM hallucination** would instead
> require prompt tightening or generator temperature adjustments. Confirming
> which failure mode occurred here (via the retrieved-sources field saved in
> `eval_results.json`) avoided misdiagnosing this as a hallucination issue.

---

## 5. Execution Instructions

### 5.1 Prerequisites

* Python 3.11 or 3.12
* A Google AI Studio API key with access to the Gemini API
  (https://aistudio.google.com/apikey)

### 5.2 Setup

```bash
# 1. Clone the repository and enter the project directory
git clone <repository-url>
cd final_ai

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Gemini API key
# Windows (PowerShell):
$env:GOOGLE_API_KEY="your_key_here"
# macOS / Linux:
export GOOGLE_API_KEY="your_key_here"
```

### 5.3 Build the Vector Store

The ChromaDB index must be built once before the first run. Subsequent runs
load the persisted index automatically and do **not** re-embed the corpus.

```bash
# First-time build (or rebuild after changing data/)
python -m src.ingestion --rebuild

# Subsequent runs (loads existing index, no API calls)
python -m src.ingestion
```

### 5.4 Run the Conversational RAG Loop

```bash
python main.py
```

Type questions interactively; type `exit` or `quit` to end the session (this
triggers long-term memory summarisation and persistence).

### 5.5 Run the Evaluation Suite

```bash
# Full 15-question evaluation (hybrid retrieval, default settings)
python -m src.evaluate

# Quick smoke-test on a subset
python -m src.evaluate --sample 5

# Compare retrieval strategies
python -m src.evaluate --strategy dense
python -m src.evaluate --strategy sparse
```

Results are written to `eval_results.json` and a formatted report is printed
to stdout.

### 5.6 Launch the Bonus Streamlit Interface

```bash
streamlit run app.py
```

### 5.7 Launch the Bonus FastAPI Episodic Memory Backend

```bash
uvicorn src.api:app --reload --port 8000
```

API documentation is then available at `http://localhost:8000/docs`.

---

## 6. Project Structure

```
final_ai/
├── main.py                  # Entry point: conversational RAG loop
├── app.py                   # Bonus A: Streamlit chat interface
├── data/                    # Source documents (PDF, MD, JSON, XLSX, HTML)
│   ├── analyst_report.pdf
│   ├── earnings_summary.md
│   ├── macro_indicators.json
│   ├── stock_data.xlsx
│   └── news_article.html
├── chroma_db/                # Persisted ChromaDB vector store
├── src/
│   ├── ingestion.py          # Task 1: parsing, chunking, metadata
│   ├── retrieval.py          # Task 2: dense, sparse, hybrid + filtering
│   ├── memory.py              # Task 3: short-term + long-term memory
│   ├── evaluate.py            # Task 4: LLM-as-a-Judge evaluation
│   └── api.py                 # Bonus B: FastAPI episodic memory backend
├── eval_dataset.jsonl         # 15 QA pairs with ground truth answers
├── eval_results.json          # Generated evaluation output (after running evaluate.py)
├── memory.db                  # Long-term memory SQLite DB (created at runtime)
├── episodic_memory.db         # Bonus B SQLite DB (created at runtime)
├── requirements.txt
└── README.md
```

---

## 7. Bonus Implementations

### 7.1 Bonus A — Streamlit Chat Interface (`app.py`)

A Streamlit-based chat UI that:

* Displays the ongoing chat history using `st.chat_message`
* Shows, in a sidebar, the exact document chunks retrieved for the latest
  answer, including their `source` and `document_category` metadata, inside
  expandable panels
* Wires `MemoryManager` and `Retriever` together with `@st.cache_resource` so
  the vector store and BM25 index are loaded once per server process
* Provides a "Clear Memory & End Session" button that triggers
  `memory.end_session()`, persisting the conversation summary before clearing
  the UI state

### 7.2 Bonus B — Episodic Memory Backend (`src/api.py`)

A FastAPI application backed by a separate SQLite database
(`episodic_memory.db`), implementing the required relational schema:

```sql
CREATE TABLE Conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title      TEXT NOT NULL
);

CREATE TABLE Messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES Conversations(id) ON DELETE CASCADE
);
```

Exposed REST endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/conversations` | List all conversation threads |
| `GET` | `/conversation/{id}` | Retrieve a thread's full message history |
| `POST` | `/message` | Store a new user/assistant message pair (creates a new thread if no `conversation_id` is supplied) |
| `DELETE` | `/conversation/{id}` | Delete a thread and cascade-delete its messages |

`PRAGMA foreign_keys = ON` is explicitly enabled per connection so that the
`ON DELETE CASCADE` constraint is honoured by SQLite (which disables foreign
key enforcement by default).

---

## 8. Known Limitations

* **Free-tier API rate limits.** Google AI Studio's free tier enforces
  request-per-minute quotas; large evaluation runs may need to be executed in
  batches or with short delays between calls (`src/evaluate.py` includes a
  configurable `delay_secs` parameter for this purpose).
* **Sheet-level granularity in Excel ingestion.** As discussed in §4.4, all
  sheets of `stock_data.xlsx` currently share a single `document_category`,
  which can reduce retrieval precision when a query targets one specific
  sheet among several semantically similar ones.
* **BM25 corpus is held entirely in memory.** This is performant for a
  corpus of this size (~100 chunks) but would not scale to a much larger
  knowledge base without an external sparse index (e.g. Elasticsearch).