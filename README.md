# AI Investigator

**An evidence-based, multi-document AI investigation engine** built with FastAPI, Streamlit, Qdrant, and Google Gemini.

AI Investigator retrieves evidence across multiple documents, combines semantic and keyword search, supports multi-hop investigation, and generates answers with source references. It is designed to acknowledge missing evidence and report potential contradictions rather than invent unsupported conclusions.

## Live Demo

- [Web application (Streamlit Community Cloud)](https://aiinvestigator-o9fvk4txqc4rgqkulxvuuf.streamlit.app/)
- [Backend API (Render)](https://aiinvestigator.onrender.com/)
- [Interactive API documentation](https://aiinvestigator.onrender.com/docs)
- [Backend health check](https://aiinvestigator.onrender.com/health)

The Streamlit frontend communicates with the FastAPI backend on Render, which connects to Qdrant Cloud and Google Gemini.

> **Free-tier limitations:** Render may take 50 seconds or longer to wake after inactivity; Gemini requests are subject to quotas and rate limits.

## Features

- Ingest PDF, JSON, JSONL, and CSV documents; extract content and metadata.
- Add, list, read, update, and delete user-managed documents.
- Deduplicate documents and chunk content using `fixed_size` or `sentence_aware` strategies.
- Search with dense semantic embeddings, sparse BM25, or hybrid retrieval with custom Reciprocal Rank Fusion (RRF).
- Perform standard RAG and multi-hop investigations using Gemini tool calling.
- Return cited evidence, indicate insufficient evidence, and surface detected contradictions.
- Inspect evidence, search history, and document-management results through Streamlit.
- Evaluate retrieval, answer correctness, document-level citation relevance, and robustness.

## Architecture

```text
Streamlit frontend (Streamlit Community Cloud)
    |
    v
FastAPI backend (Render / Docker)
    |-- Document ingestion and CRUD
    |   |-- PDF / JSON / JSONL / CSV adapters
    |   |-- Deduplication and chunking
    |   `-- Dense ONNX embeddings + sparse BM25 --> Qdrant Cloud
    |
    `-- Investigation / RAG
        |-- Semantic and keyword retrieval
        |-- Custom RRF result fusion
        |-- Standard or tool-calling multi-hop retrieval
        `-- Google Gemini --> Answer with evidence and citations
```

## Technical Design Decisions

| Choice | Rationale | Trade-off |
| --- | --- | --- |
| FastAPI | Python-native API, validation, OpenAPI documentation, and ML integration. | UI is deployed separately. |
| Streamlit | Fast development of an interactive investigation interface. | Less UI customization than a dedicated frontend. |
| Qdrant | Dense and sparse vector retrieval with persistent cloud storage. | Collection and payload indexes require configuration. |
| FastEmbed / ONNX | Smaller deployment footprint than a PyTorch-based embedding stack. | Semantic similarity can miss exact terms. |
| BM25 sparse retrieval | Captures names and exact terms. | May miss paraphrases. |
| Custom RRF | Fuses semantic and keyword rankings without comparing raw scores. | Fusion parameters may require tuning. |
| Fixed-size and sentence-aware chunking | Alternative segmentation approaches for different document structures. | Chunk boundaries, size, and overlap affect retrieval. |
| Google Gemini | Evidence-based generation and structured tool calling. | Quotas, latency, and model variability. |
| HotpotQA | Multi-document bridge and comparison questions with supporting facts. | May not represent real-world uploaded-document workloads. |

These choices prioritize a working, explainable, end-to-end pipeline. Two chunking strategies are implemented; **a quantitative head-to-head chunking comparison has not yet been recorded**.

## Requirements

- Python 3.14 (the version used by the backend Docker image), or another version verified compatible with pinned dependencies.
- A Gemini API key.
- A Qdrant Cloud instance or local Qdrant server.
- Docker only if running Qdrant locally.

## Local Setup (Windows PowerShell)

```powershell
git clone https://github.com/MrPrincee/AiInvestigator.git
cd AiInvestigator
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-backend.txt
Copy-Item .env.example .env
```

Configure `.env` with your own credentials:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=your_collection_name
```

For a local Qdrant instance:

```powershell
docker run -d --name ai-investigator-qdrant -p 6333:6333 -p 6334:6334 -v ai-investigator-qdrant:/qdrant/storage qdrant/qdrant
```

Use a local Qdrant URL and a **dedicated test collection** when experimenting. Never run document-mutating evaluation against the deployed collection.

Start the backend:

```powershell
uvicorn app.main:app --reload
```

In another terminal:

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

For local Streamlit, configure `API_URL = "http://127.0.0.1:8000"` in Streamlit secrets or set the `API_URL` environment variable.

- UI: `http://localhost:8501`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Dataset and Indexing

The application uses the HotpotQA `distractor` dataset. The deployed Qdrant collection is named `hotpotqa_50k_hybrid`; **the collection name is not a verified count of unique documents**. A previous development report recorded over 78,000 indexed vector points, which may include multiple chunks per document. The number of unique indexed `document_id` values should be independently verified before claiming that 50,000 distinct documents were indexed.

The populated Qdrant database is not included in the repository. `dataset/prepare_dataset.py` prepares local dataset material; it does not populate Qdrant by itself. Evaluation fixtures in `data/evaluation/` are reproducible from the scripts, subject to the same dataset and collection configuration.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload and index documents |
| POST | `/documents` | Create a document |
| GET | `/documents` | List user-managed documents |
| GET | `/documents/{document_id}` | Read a document |
| PUT | `/documents/{document_id}` | Update a document |
| DELETE | `/documents/{document_id}` | Delete a document |
| POST | `/search` | Semantic, keyword, or hybrid retrieval |
| POST | `/rag/ask` | Evidence-based investigation |
| POST | `/llm/generate` | Direct text generation |

Full schemas: [Swagger UI](https://aiinvestigator.onrender.com/docs).

## Evaluation Summary

The retrieval benchmark uses **50 answerable HotpotQA questions** (25 bridge and 25 comparison). Metrics are computed at the **unique-document level**, after deduplicating document IDs in returned chunks.

| Retrieval metric | Semantic | Keyword | Hybrid |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.7000 | 0.6100 | **0.7200** |
| Recall@10 | 0.7200 | 0.7000 | **0.8000** |
| MRR@10 | 0.8583 | 0.7218 | **0.8687** |

Hybrid achieved the strongest retrieval scores on this benchmark. The RAG answer modes were separately compared on the **40 questions common to both answer-correctness result sets**:

| Answer metric (40 common questions) | Standard RAG | Tool Calling |
| --- | ---: | ---: |
| Mean LLM-judged answer correctness (0–1) | **0.8250** | **0.8250** |
| Incorrect or partially correct (score < 1) | 7 | 7 |
| Flagged for manual review | 1 | 0 |
| Incomplete retrieved-source coverage of gold supporting documents | 14 | 16 |

Paired comparison: **1 Standard win, 1 Tool Calling win, 38 ties**. Mean correctness is a score, not necessarily the fraction of fully correct answers.

Citation evaluation uses **48 questions common to both answer-result sets**:

| Citation metric (48 common questions) | Standard RAG | Tool Calling |
| --- | ---: | ---: |
| Citation precision proxy | 0.7750 | **0.8028** |
| Citation recall | 0.7500 | **0.7812** |
| Citation F1 proxy | 0.7310 | **0.7552** |
| Answers containing citations | 42/48 | **45/48** |
| Unresolved citation references | 0 | 0 |

Citation metrics compare cited **document IDs** with HotpotQA supporting-document IDs; they do **not** establish whether each cited passage entails the associated claim. The answer-evaluation `citation_coverage` field uses the response's `sources` collection and is a separate retrieved-source coverage measure.

Observed errors include unsupported abstention on answerable questions, answers stopping at an intermediate entity instead of completing the second hop, confusion between a company's headquarters and requested store locations, and at least one questionable reference answer (`Crambidae` is a family rather than a genus). These are discussed in the [Evaluation Report](EVALUATION_REPORT.md).

**Evaluation coverage and caveats:** The 50-question retrieval set, 48-question citation set, and 40-question answer-correctness set are not interchangeable. The stored answer-results file contains 98 records and the correctness file contains 80 records. Five unanswerable and five conflicting-evidence scenarios are defined in `evaluation/robustness_evaluation.py`, but their end-to-end pass rates have **not been verified**. The robustness script creates and deletes documents; use an isolated test collection only.

### Reproducing Offline Reports

The following commands analyze existing files without calling Gemini or changing Qdrant:

```powershell
python -m evaluation.error_analysis
python -m evaluation.citation_evaluation
```

Retrieval and answer generation evaluation scripts access live services and may consume resources or Gemini quota. The LLM-as-judge evaluation also calls Gemini. See [Evaluation Report](EVALUATION_REPORT.md) for methodology and limitations.

## Tests and API Quotas

```powershell
python -m pip install pytest
python -m pytest tests/test_adapters.py tests/test_chunking.py tests/test_rrf.py -q
```

Do not run the entire test suite indiscriminately: some integration tests call Gemini or connect to the configured Qdrant instance. Free-tier quotas and production data must be protected.

## Repository Structure

```text
app/                  FastAPI routes, schemas, configuration, and services
dataset/              Format adapters, deduplication, chunking, dataset preparation
evaluation/           Retrieval, answer, citation, and robustness evaluation
frontend/             Streamlit UI and frontend requirements
tests/                Unit and integration tests
data/evaluation/      Evaluation fixtures and recorded metrics
Dockerfile            Backend deployment definition
README.md             Project overview and setup
EVALUATION_REPORT.md  Evaluation methodology, results, and error analysis
AI_USAGE.md           AI assistance and independent development report
```

## Deployment

| Component | Hosting | Configuration |
| --- | --- | --- |
| Frontend | Streamlit Community Cloud | `frontend/streamlit_app.py`, `frontend/requirements.txt` |
| Backend | Render (Docker) | Root `Dockerfile`, `requirements-backend.txt` |
| Vector database | Qdrant Cloud | Configured collection; persisted independently of backend restarts |
| LLM | Google Gemini API | API key stored on backend |

Backend variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION`. Streamlit secrets should contain:

```toml
API_URL = "https://aiinvestigator.onrender.com"
```

**Security and operations:** Never commit `.env` or credentials. Public demo upload and document CRUD endpoints require authentication, authorization, and upload limits before unrestricted production use. Gemini calls consume the backend operator's quota; Render cold starts and free-tier resource limits may affect responsiveness.

## Documentation

- [Evaluation Report](EVALUATION_REPORT.md) — datasets, methods, metrics, results, error analysis, limitations, and reproduction.
- [AI Usage Report](AI_USAGE.md) — AI tools used during development and independently validated technical decisions.

## Project Status and Limitations

The deployed frontend, backend, Qdrant integration, CRUD, hybrid retrieval, and RAG generation have been exercised. The evaluation records show improved retrieval from hybrid fusion and slightly better citation-document matching for tool calling, **without an observed answer-correctness gain**. Outstanding validation includes quantitative chunking-strategy comparison, confirmed unique-document count, and completed isolated robustness experiments. The system's citations and contradiction flags should not be interpreted as a guarantee of factual correctness.
