# AI Investigator

An evidence-based multi-document investigation engine built with **FastAPI**, **Streamlit**, **Qdrant**, and **Google Gemini**.

## Features

- Upload and index PDF, JSON, JSONL, and CSV documents.
- Add, browse, update, and delete indexed documents.
- Deduplication, metadata extraction, and two chunking strategies: `fixed_size` and `sentence_aware`.
- Hybrid retrieval using dense semantic embeddings, sparse BM25 search, and custom Reciprocal Rank Fusion (RRF).
- Multi-hop RAG and Gemini tool-calling investigation modes.
- Evidence-linked answers, abstention on insufficient evidence, and contradiction reporting.
- Streamlit interface with evidence inspection and search history.
- Evaluation scripts for retrieval, answer correctness, citation relevance proxies, error analysis, and robustness.

## Architecture

```text
Streamlit frontend
       |
       v
FastAPI backend
       |---- Document ingestion / CRUD
       |          |---- PDF / JSON / JSONL / CSV adapters
       |          |---- Chunking / deduplication
       |          `---- Dense + sparse embeddings --> Qdrant
       |
       `---- RAG service
                  |---- Semantic + BM25 retrieval
                  |---- Custom RRF ranking
                  |---- Multi-hop search / tool calling
                  `---- Gemini --> Evidence-linked answer
```

## Requirements

- Python 3.11+ (ensure your Python version is supported by the ML dependencies)
- Docker or another way to run Qdrant
- A Gemini API key from Google AI Studio

## Local installation (Windows PowerShell)

Clone the repository:

```powershell
git clone https://github.com/MrPrincee/AiInvestigator.git
cd AiInvestigator
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your real Gemini API key:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Never commit `.env` or a real API key.

Start Qdrant using Docker:

```powershell
docker run -d --name ai-investigator-qdrant -p 6333:6333 -p 6334:6334 -v ai-investigator-qdrant:/qdrant/storage qdrant/qdrant
```

Start the FastAPI backend:

```powershell
uvicorn app.main:app --reload
```

In a second terminal, start the Streamlit frontend:

```powershell
streamlit run frontend/streamlit_app.py
```

- UI: `http://localhost:8501`
- API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

**Data setup:** The default Qdrant collection is `hotpotqa_50k_hybrid`. The full indexed HotpotQA dataset and populated Qdrant storage are not committed to the repository. You must prepare/index data or use an existing compatible collection. Running `dataset/prepare_dataset.py` prepares the dataset but does not itself populate Qdrant.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload and index documents |
| POST | `/documents` | Create a document |
| GET | `/documents` | List documents |
| GET | `/documents/{document_id}` | Read a document |
| PUT | `/documents/{document_id}` | Update a document |
| DELETE | `/documents/{document_id}` | Delete a document |
| POST | `/rag/ask` | Investigate a question with RAG |

Additional search and LLM endpoints are documented in Swagger.

## Evaluation

The evaluation set includes 50 answerable HotpotQA questions (25 bridge and 25 comparison). Retrieval was compared in semantic, keyword, and hybrid configurations.

| Metric | Semantic | Keyword | Hybrid |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.7000 | 0.6100 | 0.7200 |
| Recall@10 | 0.7200 | 0.7000 | 0.8000 |
| MRR@10 | 0.8583 | 0.7218 | 0.8687 |

Other evaluation code includes LLM-judged answer correctness, error analysis, and document-level citation precision/recall/F1 **proxies**. These citation proxies are not a direct measure of claim-level citation faithfulness. Robustness tests for unanswerable questions and contradictory evidence still need final execution/verification because of API quota limits.

Run tests:

```powershell
python -m pytest tests
```

Some integration tests may require Qdrant, indexed data, and API credentials.

## Repository structure

```text
app/          FastAPI endpoints and services
dataset/      Dataset adapters, chunking, and preparation
evaluation/   Retrieval, answer, citation, and robustness evaluation
frontend/     Streamlit UI
tests/        Automated tests
data/         Evaluation fixtures and generated local data
```

## Deployment notes

The current configuration is intended for local development. The Streamlit frontend uses `http://127.0.0.1:8000` as the backend URL, and backend services use `http://localhost:6333` for Qdrant. Before public deployment, make these configurable and set up hosted services, authentication, request limits, upload restrictions, and secure environment variables. All AI requests use the backend operator's Gemini API key, so usage is subject to that project's quotas and billing.

## Project status

The core backend, Streamlit UI, document CRUD, hybrid retrieval, and evaluation scripts are implemented. Public deployment and final end-to-end robustness testing remain pending.
