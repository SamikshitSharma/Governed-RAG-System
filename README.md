# Governed Cloud Hosted RAG System for Enterprise Knowledge Management

Unified enterprise RAG application with:

- FastAPI backend as the production source of truth
- Next.js frontend as the single front door
- ChromaDB persistence for governed retrieval
- Azure Container Apps deployment for API and UI

## Source Of Truth

- Production API: `https://rag-api.proudpebble-8567eb99.eastus.azurecontainerapps.io`
- Frontend runtime wiring: `BACKEND_API_BASE_URL`
- Azure persistence mount: `/mnt/chroma`

The frontend does not hardcode the Azure URL anymore. It calls the local Next route at `/api/query`, and that route proxies to the Azure-backed FastAPI service using `BACKEND_API_BASE_URL`.

## Local Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r requirements.txt
npm install
Copy-Item .env.template .env
```

Required environment variables:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
BACKEND_API_BASE_URL=https://rag-api.proudpebble-8567eb99.eastus.azurecontainerapps.io
```

## Run

Frontend against the Azure-backed API:

```powershell
npm run dev:ui
```

Backend locally when needed:

```powershell
npm run dev:api
```

## NDA Corpus Ingestion

Bulk-ingest the NDA corpus into the existing vector store:

```powershell
py -3.11 main.py ingest --profile nda --replace-existing --source-dir "C:\Users\milug\OneDrive\Desktop\nda\files"
```

What the NDA ingestion pipeline does:

- loads every PDF in the corpus
- removes repeated headers, footers, and page-number noise
- preserves clause-oriented sections before chunking
- tags each chunk with NDA metadata such as `document_type`, `clause_type`, `page_number`, `batch_id`, and `sensitivity_level`
- skips broken files gracefully and logs failures to `vector_db/ingest_failures.jsonl`
- logs ingest batches to `vector_db/ingest_log.jsonl`
- preserves original uploaded filenames and source paths when ingestion is performed through the Azure API

## Query Behavior

`/query` returns governed, grounded responses backed by the retrieved corpus:

- refusal when evidence is missing or weak
- trust score derived from retrieval confidence, faithfulness, and citation density
- faithfulness score from the evaluation step
- traceable sources including page/clause metadata

## Azure Deployment

Deploy both backend and frontend container apps:

```powershell
.\deploy.ps1
```

Deployment wiring:

- `api/Dockerfile` builds the FastAPI service
- `Dockerfile` builds the Next.js UI
- `infra/main.bicep` deploys both apps and injects `BACKEND_API_BASE_URL` into the UI container
- Chroma persists through the Azure Files share mounted at `/mnt/chroma`
