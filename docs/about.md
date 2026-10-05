# About the Project: RFP Intelligence Platform

## 1. Executive Summary
The **RFP Intelligence Platform** is an enterprise-grade artificial intelligence platform engineered to parse, understand, index, query, and extract structured procurement data from complex Request for Proposal (RFP) packages. 

Bidding and proposal teams regularly process massive, heterogeneous documentation bundles comprising main solicitations, technical specifications, legal affidavits, and multi-version addenda across both PDF and HTML formats. Extracting critical facts (e.g., submission deadlines, bond requirements, hardware specs, SLAs) manually is time-consuming, prone to human error, and risks catastrophic omissions when addenda supersede original clauses.

This platform solves this challenge by pairing a **Dense + Sparse Hybrid RAG Search Engine** with a **Cooperating Multi-Agent Orchestration Architecture**.

---

## 2. The Core Problem Statement

Public and private sector procurement documents exhibit unique structural challenges:
1. **Scattered Information Across Multi-Page Documents**: Essential requirements are dispersed across dozens of pages, nested within dense legal clauses, tables, and exhibits.
2. **Addendum Supersession & Version Drift**: Solicitations are frequently amended post-release. Addenda often extend due dates, revise hardware SKUs, or clarify ambiguities. Naive LLMs or basic vector databases fail to recognize when an addendum invalidates a prior term.
3. **Table & Specification Integrity**: Hardware configurations (CPU, RAM, storage, SKUs) are formatted in complex tabular grids that standard PDF text scrapers fragment or lose.
4. **Hallucination Risks in Procurement**: Sales and bid teams cannot afford hallucinated answers. Every single extracted data point must be backed by an immutable, traceable source citation `[Document, Page]`.

---

## 3. The Solution Approach

Our solution tackles the problem through an end-to-end, multi-stage engineering architecture:

### A. Document Ingestion & Table Extraction
- Employs `pdfplumber` and `BeautifulSoup4` to parse multi-page PDFs and portal HTML captures without flattening table hierarchies.
- Converts tabular sections into atomic Markdown tables to preserve row-column semantics for downstream retrieval and reasoning.
- Cleans repeating headers/footers, resolves hyphenated line breaks, and classifies document types (`rfp`, `addendum`, `specs`, `affidavit`, `bid_page`).

### B. RFP-Optimized Heading & Table-Aware Chunking
- Detects section headings (e.g., `Section 1 - General Information`, `SCOPE AND SPECIFICATIONS`) and attaches them as contextual breadcrumbs to all sub-chunks.
- Enforces table atomicity to avoid slicing hardware spec rows across chunk boundaries.
- Employs a sliding character overlap (600 char window, 120 char overlap) to preserve cross-sentence continuity.

### C. True Hybrid Retrieval & Cross-Encoder Re-Ranking
- **Dense Semantic Vector Search**: `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) running locally with ONNX Runtime for semantic conceptual similarity.
- **Sparse Lexical Search**: BM25Okapi for exact alphanumeric token matching (essential for solicitation IDs like `JA-207652`, SKU numbers like `210-BLYZ`, and dates).
- **Reciprocal Rank Fusion (RRF, $k=60$)**: Blends dense and sparse candidate rankings without requiring fragile score scaling.
- **Cross-Encoder Re-Ranking**: `BAAI/bge-reranker-base` evaluates $(query, passage)$ pairs simultaneously using cross-attention, ensuring the most definitive passage lands at rank #1.

### D. Multi-Agent Orchestration Team
- **Orchestrator**: Formulates execution plans, coordinates agents, and oversees task completion.
- **Ingestion & Retrieval Agents**: Manages document discovery, indexing, and serving traceable evidence passages.
- **3 Specialized Extraction Agents**: Parallel specialists extracting *Dates & Logistics*, *Commercial & Legal*, and *Product & Specs*.
- **Addendum Reconciliation Agent**: Specifically tracks addenda, detects amended clauses, and supersedes outdated values with an audit log.
- **Validator / Critic Agent**: Enforces citation grounding, verifies data formats, detects hallucinations, and executes an automated feedback retry loop.
- **Q&A & Report Agent**: Answers free-form questions with strict inline citations.

---

## 4. Key Capabilities & Deliverables

- **20 Structured RFP Fields Extracted**: Full Section 8.1 JSON output generated for `Bid1` and `Bid2`.
- **Addendum Reconciliation**: Automatically caught that Addendum 2 extended the `Bid1` due date to **July 9, 2024 at 2:00 PM CST** (original: June 27, 2024).
- **Comprehensive Benchmark Suite**: 18 ground-truth test pairs evaluated across 4 configurations, proving a top **Recall@1 of 77.8%** and **MRR of 0.8148**.
- **Interactive Web UI**: Streamlit application on `localhost:8501` featuring Hybrid Search, Q&A Chat, Extraction Inspector, and Go/No-Go Decision Engine.
- **REST API**: Production FastAPI service on `localhost:8000` with interactive Swagger docs.
- **CI/CD & Docker**: Dockerfile, `docker-compose.yml`, and GitHub Actions workflow.
