# AI Investigator

**An evidence-based, multi-document AI investigation engine** built with FastAPI, Streamlit, Qdrant, and Google Gemini.

AI Investigator retrieves evidence across multiple documents, combines semantic and keyword search, supports multi-hop investigation, and generates answers with source references. It is designed to acknowledge missing evidence and report potential contradictions rather than invent unsupported conclusions.

## Live Demo

- [Web application (Streamlit Community Cloud)](https://aiinvestigator-o9fvk4txqc4rgqkulxvuuf.streamlit.app/)
- [Backend API (Render)](https://aiinvestigator.onrender.com/)
- [Interactive API documentation](https://aiinvestigator.onrender.com/docs)
- [Backend health check](https://aiinvestigator.onrender.com/health)

The Streamlit frontend communicates with the FastAPI backend hosted on Render. The backend accesses Qdrant Cloud and Google Gemini.

> **Free-tier limitations:** Render may take 50 seconds or longer to wake after inactivity; Gemini requests are subject to API quotas and rate limits.

## Features

- Ingest PDF, JSON, JSONL, and CSV documents; extract content and metadata.
- Add, list, read, update, and delete user-managed documents.
- Deduplicate documents and chunk content using `fixed_size` or `sentence_aware` strategies.
- Search with dense semantic embeddings, sparse BM25, or hybrid retrieval with custom Reciprocal Rank Fusion (RRF).
- Perform standard RAG and multi-hop investigations using Gemini tool calling.
- Return cited evidence, indicate insufficient evidence, and surface detected contradictions.
- Inspect evidence, search history, and document-management results through Streamlit.
- Evaluate retrieval quality, answer correctness, citation metrics, and robustness.

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
| FastAPI | Python-native API framework with request validation, OpenAPI documentation, and straightforward ML integration. | Requires a separately deployed UI. |
| Streamlit | Rapid development of an interactive investigation interface. | Less control over complex UI behavior than a custom frontend. |
| Qdrant | Supports dense and sparse vectors in one collection for hybrid search. | Requires careful collection configuration and separation of test and cloud environments. |
| FastEmbed / ONNX | Semantic retrieval without a large PyTorch stack in the deployed backend. | Quality depends on the embedding model; exact terms can be missed. |
| BM25 sparse retrieval | Captures exact names, identifiers, and terms semantic search can miss. | Alone, it may miss paraphrases. |
| Custom RRF | Combines rankings without requiring comparable raw scores. | Rank constant and candidate counts may require tuning. |
| Two chunking strategies | Allows comparison of fixed-size and sentence-aware boundaries. | Chunk size and overlap influence recall and storage. |
| Google Gemini | Generates evidence-grounded answers and supports tool calling. | Quotas, latency, and variability require careful handling. |
| HotpotQA | Provides multi-hop questions and supporting facts for evaluation. | Benchmark performance may not generalize to uploaded documents. |

These choices prioritize a demonstrable end-to-end investigation pipeline, reproducible retrieval evaluation, and manageable deployment complexity; they are not claims that the selected technologies are universally superior.

## Requirements

- Python 3.14 (matching the backend Docker image), or another version confirmed compatible with pinned dependencies.
- A Gemini API key.
- A Qdrant Cloud instance or a local Qdrant server.
- Docker only if running Qdrant locally.

## Local Setup (Windows PowerShell)

Clone the repository:

```powershell
git clone https://github.com/MrPrincee/AiInvestigator.git
cd AiInvestigator
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install backend dependencies (the old root `requirements.txt` is no longer used):

```powershell
python -m pip install -r requirements-backend.txt
```

Copy `.env.example` to `.env` and configure credentials:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=hotpotqa_50k_hybrid
```

The Qdrant URL and key above are for Cloud. For a local instance, start Qdrant with Docker and configure the application connection accordingly:

```powershell
docker run -d --name ai-investigator-qdrant -p 6333:6333 -p 6334:6334 -v ai-investigator-qdrant:/qdrant/storage qdrant/qdrant
```

> **Important:** The service uses the configured Cloud URL when present. Do not run destructive tests against the deployed collection. Use isolated local/test configuration and a dedicated test collection.

Start the backend:

```powershell
uvicorn app.main:app --reload
```

In another terminal, install frontend dependencies and start Streamlit:

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

For local use, set Streamlit's `API_URL` to `http://127.0.0.1:8000` through its secrets configuration.

- Local UI: `http://localhost:8501`
- API docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

## Dataset and Indexing

The deployed Qdrant collection is `hotpotqa_50k_hybrid`. The populated vector database is not stored in this repository. Connect to a compatible collection or prepare and index data into your own collection.

`dataset/prepare_dataset.py` prepares dataset material; it does not itself populate Qdrant.

> Do not run indexing or migration scripts against the live collection unless the operation is explicitly intended and verified.

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

Full request and response schemas are available in the [Swagger UI](https://aiinvestigator.onrender.com/docs).

## Evaluation

The recorded retrieval benchmark contains **50 answerable HotpotQA questions**: 25 bridge and 25 comparison questions. Three retrieval configurations were compared:

| Metric | Semantic | Keyword | Hybrid |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.7000 | 0.6100 | **0.7200** |
| Recall@10 | 0.7200 | 0.7000 | **0.8000** |
| MRR@10 | 0.8583 | 0.7218 | **0.8687** |

Hybrid retrieval performs best on all three reported metrics in this benchmark. Results are specific to this dataset and configuration.

Additional scripts cover LLM-judged answer correctness, document-level citation precision/recall/F1 proxies, error analysis, and robustness. Document-level citation proxies do not establish claim-level citation faithfulness. Final robustness checks for unanswerable and contradictory-evidence scenarios still require execution/verification due to API quota constraints.

## Tests and API Quotas

Install test tooling if needed:

```powershell
python -m pip install pytest
```

Run a small set of tests that does not intentionally call Gemini:

```powershell
python -m pytest tests/test_adapters.py tests/test_chunking.py tests/test_rrf.py -q
```

> Do not run the entire `tests/` suite indiscriminately: some integration tests make real Gemini requests (consuming quota), and others can connect to the Qdrant instance configured in `.env`. Integration tests should eventually use mocks or isolated infrastructure. Tests are not automatically run by Render merely because they are present in the repository.

## Repository Structure

```text
app/           FastAPI routes, schemas, configuration, and services
dataset/       Format adapters, deduplication, chunking, dataset preparation
evaluation/    Retrieval, answer, citation, and robustness evaluation
frontend/      Streamlit app and frontend requirements
tests/         Unit and integration tests
data/          Evaluation fixtures and generated local data
Dockerfile     Backend deployment definition
```

## Deployment

| Component | Hosting | Configuration |
| --- | --- | --- |
| Frontend | Streamlit Community Cloud | `frontend/streamlit_app.py`, `frontend/requirements.txt` |
| Backend | Render (Docker) | Root `Dockerfile`, `requirements-backend.txt` |
| Vector database | Qdrant Cloud | `hotpotqa_50k_hybrid` collection |
| LLM | Google Gemini API | API key stored on backend |

Backend environment variables:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=hotpotqa_50k_hybrid
```

Configure the frontend's Streamlit secrets:

```toml
API_URL = "https://aiinvestigator.onrender.com"
```

**Security and operations:** Never commit `.env` or API keys. The public demo's upload and document CRUD endpoints need authentication, authorization, and appropriate request/upload limits before unrestricted production use. All Gemini calls consume the backend operator's API quota. Free-tier cold starts and resource limits can affect availability.

## Project Status and Limitations

The hosted backend, Streamlit interface, Qdrant Cloud integration, document CRUD, hybrid retrieval, and RAG generation have been exercised. Retrieval benchmark results are recorded. Robustness testing for unanswerable questions and conflicting evidence remains to be verified end-to-end. The system can report insufficient or conflicting evidence, but generated answers and citations should still be checked for accuracy.
