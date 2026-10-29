# AI Usage Report
## AI Investigator — Evidence-Based Multi-Document Investigation Engine

**Program:** Talent Growth Program — Applied AI  
**Project:** AI Investigator  
**GitHub Repository:** https://github.com/MrPrincee/AiInvestigator

---

## 1. Introduction

This report describes how Artificial Intelligence tools were used during the development of AI Investigator, an evidence-based multi-document investigation engine.

AI tools were used to support learning, architectural planning, code development, debugging, and documentation. They served as development assistants rather than replacements for technical understanding, implementation, or validation.

The project combines FastAPI, Streamlit, Qdrant, Google Gemini, and Retrieval-Augmented Generation (RAG) to retrieve information from multiple documents and generate evidence-based answers.

## 2. AI Tools Used

### 2.1 ChatGPT — Development Assistant

ChatGPT was used throughout the development process for:

- Understanding RAG architecture and multi-document retrieval.
- Discussing backend architecture and service responsibilities.
- Explaining FastAPI concepts, including application lifespan and dependency management.
- Reviewing Python implementation approaches.
- Understanding vector databases and hybrid search.
- Discussing semantic search, BM25, and Reciprocal Rank Fusion (RRF).
- Troubleshooting Python dependencies and Docker deployment issues.
- Interpreting runtime errors and API responses.
- Preparing technical documentation.

ChatGPT also helped explain unfamiliar concepts and compare alternative technical approaches before implementation.

### 2.2 Google Gemini — Application LLM

Google Gemini was integrated into AI Investigator as the language model responsible for generating answers based on retrieved evidence.

Its responsibilities include:

- Processing user investigation questions.
- Generating natural-language answers.
- Combining evidence retrieved from multiple documents.
- Supporting multi-step investigation workflows.
- Participating in tool-calling workflows.
- Producing responses with references to supporting evidence.

Unlike ChatGPT, which was used as a development assistant, Gemini is a component of the deployed application.

## 3. How AI Assisted the Development Process

### 3.1 Architecture and Technology Selection

AI assistance was used to discuss different approaches to implementing a document investigation system.

Topics included:

- FastAPI versus alternative Python web frameworks.
- Qdrant versus other vector databases.
- Semantic retrieval versus hybrid retrieval.
- Different document chunking strategies.
- Approaches to integrating an LLM with retrieved evidence.
- Separation of document ingestion, retrieval, and answer generation.

The resulting architecture separates the Streamlit user interface, FastAPI backend, document-processing services, Qdrant vector database, and Gemini integration.

### 3.2 Document Processing and Retrieval

AI assistance supported the development and understanding of:

- Document ingestion and format handling.
- Metadata extraction.
- Content-based document identifiers.
- Deduplication.
- Fixed-size and sentence-aware chunking.
- Dense and sparse embeddings.
- Hybrid retrieval.
- Custom RRF-based ranking.

The retrieval system combines semantic similarity with keyword-based search to improve the ability to find relevant evidence.

### 3.3 RAG and Tool Calling

ChatGPT was used to discuss how a RAG system can move beyond simple question answering.

Development topics included:

- Retrieving relevant passages before answer generation.
- Combining evidence from multiple sources.
- Processing multi-hop questions.
- Handling insufficient information.
- Identifying contradictory evidence.
- Associating generated answers with source documents.
- Using structured tool calls for investigation workflows.

These concepts informed the implementation of the investigation pipeline.

### 3.4 Debugging and Problem Solving

AI assistance was particularly useful when interpreting technical errors and identifying possible solutions.

Examples included:

**Python dependency optimization**

The original embedding implementation used SentenceTransformers and PyTorch.

During deployment preparation, the embedding implementation was migrated to FastEmbed with ONNX Runtime to reduce Docker Image size and avoid unnecessary PyTorch and CUDA dependencies.

The new embedding implementation was tested for vector dimensionality and compatibility with the existing vector collection.

**Docker optimization**

AI assistance helped analyze Docker build logs, identify unnecessary dependencies, and review the Dockerfile and `.dockerignore` configuration.

The resulting Docker Image was reduced from approximately 3.4 GB to 905 MB in reported disk usage.

**Qdrant Cloud integration**

During document management testing, Qdrant returned errors related to missing payload indexes.

These errors were investigated, and keyword indexes were added for fields including:

- `managed_by`
- `document_type`
- `content_hash`
- `document_id`

This allowed the application to perform the corresponding filtered operations.

**Gemini API errors**

Gemini API errors were analyzed to distinguish temporary service unavailability from quota exhaustion.

For example:

- HTTP 503 indicated temporary model unavailability.
- HTTP 429 indicated that the daily free-tier request quota had been exceeded.

These issues demonstrated the importance of handling external API limitations.

## 4. Independent Technical Decisions and Implementation

AI-generated suggestions were not treated as automatically correct.

The development process involved reviewing suggested approaches, making implementation decisions, executing commands, testing functionality, and modifying the project based on observed results.

Important decisions and activities included:

### 4.1 Project Architecture

The project was organized into separate components for:

- Dataset preparation.
- Document ingestion and CRUD.
- Embedding generation.
- Hybrid retrieval.
- RAG and LLM integration.
- Evaluation.
- User interface.

This modular organization was selected to improve maintainability and simplify testing.

### 4.2 Dataset Preparation

HotpotQA was selected as the primary dataset.

Dataset preparation included document extraction, content-based identification, and deduplication.

The project uses a Qdrant collection containing more than 78,000 indexed vector points.

### 4.3 Search Architecture

Hybrid retrieval was selected instead of relying exclusively on semantic search.

The implementation combines dense semantic embeddings and sparse BM25 retrieval with a custom Reciprocal Rank Fusion component.

This approach was evaluated against semantic-only and keyword-only configurations.

### 4.4 Embedding Optimization

The embedding implementation was changed from SentenceTransformers to FastEmbed with ONNX Runtime.

Compatibility tests confirmed that the replacement produced 384-dimensional embeddings with very high similarity to the original model outputs.

This allowed the existing Qdrant collection to be retained without rebuilding the entire index.

### 4.5 Deployment

The application was deployed using:

- Render for the FastAPI backend.
- Streamlit Community Cloud for the user interface.
- Qdrant Cloud for vector storage.
- Google Gemini API for answer generation.

Deployment configuration was tested and adjusted based on actual runtime behavior.

### 4.6 Testing and Validation

The project was tested using a combination of automated scripts and manual verification.

Activities included:

- Testing embedding generation.
- Checking Python dependency consistency.
- Building and running the Docker Image.
- Verifying FastAPI startup.
- Testing API connectivity.
- Testing document management operations.
- Checking frontend-to-backend communication.
- Comparing retrieval configurations.

The evaluation framework also includes scripts for answer quality, citation-related metrics, and robustness testing.

## 5. Evaluation and Verification

AI assistance was used to understand evaluation methodologies and interpret results.

However, reported retrieval metrics were obtained from project evaluation experiments rather than being generated as hypothetical AI outputs.

The evaluation compared three retrieval configurations using 50 answerable HotpotQA questions.

| Metric | Semantic | Keyword | Hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.7000 | 0.6100 | 0.7200 |
| Recall@10 | 0.7200 | 0.7000 | 0.8000 |
| MRR@10 | 0.8583 | 0.7218 | 0.8687 |

The results indicate that hybrid retrieval performed better than the individual retrieval methods on the evaluated question set.

Additional evaluation components include answer correctness, citation relevance proxies, and robustness testing.

Some robustness experiments require further execution or verification because of Gemini API quota limitations.

## 6. Limitations of AI Assistance

AI-generated recommendations can be incomplete, outdated, or unsuitable for a particular environment.

Several limitations were considered during development:

**Dependency compatibility:** Suggested libraries and versions required testing against the project's Python environment.

**Cloud deployment:** Local functionality did not guarantee successful execution under the memory and resource limits of free cloud services.

**External API behavior:** Gemini availability and quota restrictions could not be eliminated through code generation alone.

**Database configuration:** Qdrant Cloud required additional payload indexes for certain filtered operations.

**Evaluation:** AI-generated explanations were not used as substitutes for measured experimental results.

For these reasons, technical recommendations were checked against runtime behavior, logs, tests, and actual application outputs.

## 7. Responsible Use of AI

AI tools were used transparently as part of the development workflow.

The following principles guided their use:

1. AI suggestions were reviewed before being incorporated into the project.
2. Generated code and configuration changes were tested where practical.
3. Technical decisions were evaluated against project requirements.
4. API keys and other credentials were kept outside the public repository.
5. Experimental results were reported separately from AI-generated explanations.
6. Known limitations and incomplete evaluation work were documented.

The developer remains responsible for understanding, maintaining, explaining, and modifying the submitted implementation.

## 8. Conclusion

AI tools contributed to the development of AI Investigator by supporting technical learning, architectural discussions, implementation decisions, debugging, and documentation.

ChatGPT primarily served as an interactive development assistant, while Google Gemini was integrated as the application's language model.

The final system combines document processing, hybrid retrieval, multi-document reasoning, evidence-linked answer generation, document management, and a publicly deployed interface.

The development process involved evaluating AI suggestions, implementing and testing components, resolving integration problems, and optimizing the application for free cloud infrastructure.

AI assistance accelerated development, but practical validation, technical decision-making, and responsibility for the submitted project remained essential.