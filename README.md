AI Investigator
An evidence-based multi-document investigation engine built with FastAPI, Streamlit, Qdrant, and Google Gemini.
Live demo
The application is deployed and accessible online:
- Web application (Streamlit Community Cloud): https://aiinvestigator-o9fvk4txqc4rgqkulxvuuf.streamlit.app/
- Backend API (Render): https://aiinvestigator.onrender.com/
- Interactive API documentation: https://aiinvestigator.onrender.com/docs
- Backend health check: https://aiinvestigator.onrender.com/health
The Streamlit frontend communicates with the hosted FastAPI backend, which uses Qdrant Cloud for vector storage and Google Gemini for answer generation.
Free-tier note: The Render backend may spin down after inactivity. The first request after a period of inactivity can take 50 seconds or longer. Gemini requests are also subject to API rate limits and daily quotas.

Features
- Upload and index PDF, JSON, JSONL, and CSV documents.
- Add, browse, update, and delete indexed documents.
- Deduplication, metadata extraction, and two chunking strategies: fixed_size and sentence_aware.
- Hybrid retrieval using dense semantic embeddings, sparse BM25 search, and custom Reciprocal Rank Fusion (RRF).
- Multi-hop RAG and Gemini tool-calling investigation modes.
- Evidence-linked answers, abstention on insufficient evidence, and contradiction reporting.
- Streamlit interface with evidence inspection and search history.
- Evaluation scripts for retrieval, answer correctness, citation relevance proxies, error analysis, and robustness.
Architecture
Streamlit frontend (Streamlit Community Cloud)
       |
       v
FastAPI backend (Render, Docker)
       |---- Document ingestion / CRUD
       |          |---- PDF / JSON / JSONL / CSV adapters
       |          |---- Chunking / deduplication
       |          `---- Dense (ONNX) + sparse (BM25) embeddings --> Qdrant Cloud
       |
       `---- RAG service
                  |---- Semantic + BM25 retrieval
                  |---- Custom RRF ranking
                  |---- Multi-hop search / tool calling
                  `---- Google Gemini --> Evidence-linked answer
Requirements (local development)
- A Python version supported by the project's dependencies
- Docker or access to a compatible Qdrant instance
- A Gemini API key from Google AI Studio
Local installation (Windows PowerShell)
Clone the repository and install dependencies:
git clone https://github.com/MrPrincee/AiInvestigator.git
cd AiInvestigator
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy .env.example to .env and configure the required environment variables:
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
Configure QDRANT_URL and QDRANT_API_KEY if connecting to Qdrant Cloud. Never commit .env or real API keys.

To use local Qdrant instead, start it with Docker:
docker run -d --name ai-investigator-qdrant -p 6333:6333 -p 6334:6334 -v ai-investigator-qdrant:/qdrant/storage qdrant/qdrant
Start the backend:
uvicorn app.main:app --reload
In a second terminal, start the frontend:
streamlit run frontend/streamlit_app.py
- Local UI: http://localhost:8501
- Local API docs: http://127.0.0.1:8000/docs
- Local health endpoint: http://127.0.0.1:8000/health
Data setup: The default Qdrant collection is hotpotqa_50k_hybrid. The full indexed HotpotQA dataset and populated Qdrant storage are not committed to this repository. Prepare/index data or connect to an existing compatible collection. Running dataset/prepare_dataset.py prepares the dataset but does not itself populate Qdrant.
API endpoints
Method	Endpoint	Purpose
GET	/health	Health check
POST	/documents/upload	Upload and index documents
POST	/documents	Create a document
GET	/documents	List documents
GET	/documents/{document_id}	Read a document
PUT	/documents/{document_id}	Update a document
DELETE	/documents/{document_id}	Delete a document
POST	/rag/ask	Investigate a question with RAG


Additional search and LLM endpoints are documented in Swagger.
Evaluation
The evaluation set includes 50 answerable HotpotQA questions (25 bridge and 25 comparison). Retrieval was compared in semantic, keyword, and hybrid configurations.
Metric	Semantic	Keyword	Hybrid
Recall@5	0.7000	0.6100	0.7200
Recall@10	0.7200	0.7000	0.8000
MRR@10	0.8583	0.7218	0.8687


Other evaluation code includes LLM-judged answer correctness, error analysis, and document-level citation precision/recall/F1 proxies. These proxies do not directly measure claim-level citation faithfulness. Final robustness tests for unanswerable questions and contradictory evidence still require execution/verification because of API quota limits.

Run tests:
python -m pytest tests
Some integration tests require Qdrant, indexed data, and API credentials.
Repository structure
app/          FastAPI endpoints and services
dataset/      Dataset adapters, chunking, and preparation
evaluation/   Retrieval, answer, citation, and robustness evaluation
frontend/     Streamlit UI and frontend requirements
tests/        Automated tests
data/         Evaluation fixtures and generated local data
Deployment
The project uses separate hosted services:
Component	Hosting	Configuration
Frontend	Streamlit Community Cloud	frontend/streamlit_app.py; frontend/requirements.txt
Backend	Render Web Service (Docker)	Root Dockerfile; requirements-backend.txt
Vector database	Qdrant Cloud	Hosted collection hotpotqa_50k_hybrid
LLM	Google Gemini API	Backend-managed API key


Backend environment variables (Render):
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
Frontend configuration (Streamlit Community Cloud): Configure the backend address through the app's Secrets settings:
API_URL = "https://aiinvestigator.onrender.com"
Do not put Gemini or Qdrant API keys in frontend code or commit secrets to Git. The backend owns the external-service credentials.

Operational considerations: The public demo uses free-tier hosting, so cold starts, resource constraints, and Gemini quotas can affect response times and availability. Public-facing document CRUD and upload operations should be protected with authentication, authorization, and appropriate request/upload limits before production use. All Gemini calls use the backend operator's API key and associated quota.
Project status
Live deployment is complete. The FastAPI backend, Streamlit UI, Qdrant Cloud integration, document CRUD, hybrid retrieval, and RAG answer generation have been exercised through the hosted application. Evaluation scripts and retrieval benchmark results are available. Final end-to-end robustness testing for unanswerable and contradictory evidence scenarios remains to be verified.
