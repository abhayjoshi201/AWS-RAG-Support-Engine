# 🧠 RAG Support System

> **Retrieval-Augmented Generation** for automated Zendesk ticket responses — powered by AWS Bedrock, OpenSearch, and FastAPI.

When a new support ticket arrives, the system embeds the ticket text, searches a knowledge base of past tickets and Help Center articles for relevant context, generates a draft reply using Claude 3 Sonnet, and posts it back to Zendesk — all in under 2 seconds.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Zendesk   │────▶│   FastAPI    │────▶│  Bedrock (Titan) │     │  Zendesk    │
│  Webhook    │     │  /webhooks  │     │  Embed ticket    │     │  Post reply │
└─────────────┘     └──────┬──────┘     └────────┬─────────┘     └──────▲──────┘
                           │                     │                      │
                           │              ┌──────▼─────────┐           │
                           │              │   OpenSearch    │           │
                           │              │   kNN search    │           │
                           │              └──────┬─────────┘           │
                           │                     │                      │
                           │              ┌──────▼─────────┐           │
                           └──────────────│ Bedrock (Claude)│───────────┘
                                          │ Generate reply  │
                                          └────────────────┘
```

## Project Structure

```
RAG/
├── .env.example              # All required env vars
├── requirements.txt          # Python 3.11+ dependencies
├── app/
│   ├── main.py               # FastAPI app, lifespan, middleware
│   ├── core/
│   │   ├── config.py          # pydantic-settings config (incl. DEMO_MODE)
│   │   ├── logging.py         # structlog JSON logging
│   │   └── retry.py           # tenacity retry decorators
│   ├── models/
│   │   └── schemas.py         # Pydantic DTOs
│   ├── services/
│   │   ├── bedrock_embeddings.py  # Titan Embed v2 (1024-dim)
│   │   ├── bedrock_llm.py        # Claude 3 Sonnet generation
│   │   ├── vector_store.py       # OpenSearch kNN index + search
│   │   ├── zendesk.py            # Async Zendesk REST client
│   │   └── demo.py               # Fake stubs for demo mode
│   ├── api/
│   │   ├── webhooks.py        # Antigravity webhook handler
│   │   ├── ingest.py          # Bulk ingest tickets & articles
│   │   └── health.py          # Health check
│   └── utils/
│       └── text.py            # Text chunking with overlap
└── rag-system-architect/      # React dashboard (optional)
```

## Quick Start

### Prerequisites

- Python 3.11+
- (For production) AWS account with Bedrock access, an OpenSearch cluster, and a Zendesk account

### 1. Clone & Install

```bash
git clone <repo-url>
cd RAG
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🧪 Demo Mode (No Credentials Needed)

Want to explore the full pipeline without any cloud services? Set one flag:

```bash
echo "DEMO_MODE=true" > .env
uvicorn app.main:app --port 8000
```

Demo mode swaps in **fake stubs** — deterministic hash-based embeddings, canned knowledge-base articles, and a template LLM reply — so every endpoint works end-to-end:

```bash
# Simulate a webhook
curl -s -X POST http://localhost:8000/webhooks/antigravity \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": 1, "subject": "Password reset", "description": "My SSO link expired"}' \
  | python3 -m json.tool
```

```bash
# Ingest demo tickets
curl -s -X POST http://localhost:8000/ingest/tickets | python3 -m json.tool

# Ingest demo articles
curl -s -X POST http://localhost:8000/ingest/articles | python3 -m json.tool
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/ingest/tickets` | Fetch & index Zendesk tickets |
| `POST` | `/ingest/articles` | Fetch & index Help Center articles |
| `POST` | `/webhooks/antigravity` | Receive ticket event → RAG → reply |
| `GET` | `/docs` | Swagger UI |

---

## Production Setup

### Ingest Your Knowledge Base

Before the webhook can retrieve relevant context, load your data:

```bash
# Index all tickets (paginated)
curl -X POST http://localhost:8000/ingest/tickets \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 50}'

# Index Help Center articles
curl -X POST http://localhost:8000/ingest/articles \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 20}'
```

### Set Up the Webhook

Point your Antigravity webhook to:
```
https://your-domain.com/webhooks/antigravity
```

Set `ANTIGRAVITY_WEBHOOK_SECRET` in `.env` to enable HMAC-SHA256 signature verification.

### Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `false` | Run with fake stubs (no cloud needed) |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Embedding model |
| `BEDROCK_LLM_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | LLM model |
| `VECTOR_DIMENSION` | `1024` | Embedding vector size |
| `VECTOR_TOP_K` | `5` | Number of context docs retrieved |
| `OPENSEARCH_INDEX` | `rag-knowledge-base` | Index name |

See [`.env.example`](.env.example) for the full list.

---

## Design Decisions

- **Structured logging** — `structlog` with JSON output for production observability
- **Retry with backoff** — `tenacity` decorators on all external calls (Bedrock, OpenSearch, Zendesk)
- **Async where it matters** — `httpx.AsyncClient` for non-blocking Zendesk API calls
- **Pydantic v2** — `pydantic-settings` for validated, typed configuration
- **kNN search** — OpenSearch HNSW index with cosine similarity
- **Webhook security** — HMAC-SHA256 signature verification

---

## React Dashboard (Optional)

A standalone visual dashboard for exploring the architecture and simulating the pipeline:

```bash
cd rag-system-architect
npm install
npm run dev
# → http://localhost:3000
```

---

## License

MIT
