import { FileSystem } from '../types';

const REQUIREMENTS_TXT = `fastapi==0.109.2
uvicorn[standard]==0.27.1
boto3==1.34.34
opensearch-py==2.4.2
httpx==0.26.0
pydantic==2.6.1
pydantic-settings==2.1.0
python-dotenv==1.0.1
tenacity==8.2.3
structlog==24.1.0
python-json-logger==2.0.7
`;

const ENV_EXAMPLE = `# ── AWS / Bedrock ────────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# ── OpenSearch ───────────────────────────────────────────
OPENSEARCH_HOST=https://localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=
OPENSEARCH_INDEX=rag-knowledge-base
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=false

# ── Zendesk ──────────────────────────────────────────────
ZENDESK_SUBDOMAIN=yourcompany
ZENDESK_EMAIL=support@yourcompany.com
ZENDESK_API_TOKEN=

# ── Antigravity Webhook ─────────────────────────────────
ANTIGRAVITY_WEBHOOK_SECRET=

# ── App ──────────────────────────────────────────────────
LOG_LEVEL=INFO
VECTOR_DIMENSION=1024
VECTOR_TOP_K=5

# Set to true to run with fake data (no AWS / OpenSearch / Zendesk needed)
DEMO_MODE=false
`;

const CONFIG_PY = `"""Centralised configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All env vars consumed by the RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── AWS / Bedrock ───────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # ── OpenSearch ──────────────────────────────────────
    opensearch_host: str = "https://localhost"
    opensearch_port: int = 9200
    opensearch_username: str = "admin"
    opensearch_password: str = ""
    opensearch_index: str = "rag-knowledge-base"
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = False

    # ── Zendesk ─────────────────────────────────────────
    zendesk_subdomain: str = ""
    zendesk_email: str = ""
    zendesk_api_token: str = ""

    # ── Antigravity Webhook ─────────────────────────────
    antigravity_webhook_secret: str = ""

    # ── App tunables ────────────────────────────────────
    log_level: str = "INFO"
    vector_dimension: int = 1024
    vector_top_k: int = 5
    demo_mode: bool = False

    @property
    def zendesk_base_url(self) -> str:
        return f"https://{self.zendesk_subdomain}.zendesk.com/api/v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
`;

const LOGGING_PY = `"""Structured logging with structlog + JSON output."""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.stdlib import add_log_level, filter_by_level

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure structlog for JSON structured logging."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            filter_by_level,
            structlog.contextvars.merge_contextvars,
            add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger with the given name."""
    return structlog.get_logger(name)
`;

const RETRY_PY = `"""Reusable retry decorator with exponential backoff + jitter."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.logging import get_logger

_log = get_logger(__name__)


def _log_retry(retry_state) -> None:
    _log.warning(
        "retrying_call",
        attempt=retry_state.attempt_number,
        wait=round(retry_state.next_action.sleep, 2) if retry_state.next_action else 0,
        fn=getattr(retry_state.fn, "__qualname__", str(retry_state.fn)),
        error=str(retry_state.outcome.exception()) if retry_state.outcome and retry_state.outcome.failed else None,
    )


bedrock_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)

opensearch_retry = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=20, jitter=1),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)

http_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=10, jitter=1),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)
`;

const SCHEMAS_PY = `"""Pydantic models / DTOs used across the application."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentSource(str, Enum):
    TICKET = "ticket"
    ARTICLE = "article"


class ZendeskTicketEvent(BaseModel):
    ticket_id: int
    subject: str = ""
    description: str = ""
    requester_email: str = ""
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    priority: str | None = None
    status: str | None = None


class IngestRequest(BaseModel):
    max_pages: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    doc_id: str
    source: DocumentSource
    title: str = ""
    content: str = ""
    score: float = 0.0


class DraftReplyResponse(BaseModel):
    ticket_id: int
    draft_reply: str
    context_docs: list[SearchResult] = Field(default_factory=list)
    posted_to_zendesk: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestResponse(BaseModel):
    source: DocumentSource
    documents_ingested: int = 0
    message: str = ""
`;

const BEDROCK_EMBEDDINGS_PY = `"""AWS Bedrock embedding service using Titan Embed v2."""

from __future__ import annotations

import json

import boto3

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import bedrock_retry

_log = get_logger(__name__)
_client = None


def _get_client():
    global _client
    if _client is None:
        settings = get_settings()
        _client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
    return _client


@bedrock_retry
def embed_text(text: str) -> list[float]:
    """Embed a single text string and return the embedding vector."""
    settings = get_settings()
    client = _get_client()

    body = json.dumps({
        "inputText": text,
        "dimensions": settings.vector_dimension,
        "normalize": True,
    })

    response = client.invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts sequentially."""
    return [embed_text(t) for t in texts]
`;

const BEDROCK_LLM_PY = `"""AWS Bedrock LLM generation service using Claude 3 Sonnet."""

from __future__ import annotations

import json

import boto3

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import bedrock_retry

_log = get_logger(__name__)
_client = None


def _get_client():
    global _client
    if _client is None:
        settings = get_settings()
        _client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
    return _client


_SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant. "
    "Use the provided context documents to draft a clear, empathetic, and accurate "
    "reply to the customer's support ticket."
)


@bedrock_retry
def generate_reply(ticket_text: str, context_docs: list[str]) -> str:
    settings = get_settings()
    client = _get_client()

    context_block = "\\n---\\n".join(context_docs) if context_docs else "(no context)"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Context:\\n{context_block}\\n\\nTicket:\\n{ticket_text}"}],
        "temperature": 0.3,
    })

    response = client.invoke_model(
        modelId=settings.bedrock_llm_model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    return "".join(b["text"] for b in result.get("content", []) if b.get("type") == "text").strip()
`;

const VECTOR_STORE_PY = `"""OpenSearch vector store — indexing and kNN similarity search."""

from __future__ import annotations

from opensearchpy import OpenSearch, RequestsHttpConnection

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import opensearch_retry
from app.models.schemas import SearchResult, DocumentSource

_log = get_logger(__name__)
_os_client = None


def _get_client() -> OpenSearch:
    global _os_client
    if _os_client is not None:
        return _os_client

    settings = get_settings()
    _os_client = OpenSearch(
        hosts=[{"host": settings.opensearch_host.replace("https://", "").replace("http://", ""),
                "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_username, settings.opensearch_password),
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )
    return _os_client


@opensearch_retry
def ensure_index() -> None:
    settings = get_settings()
    client = _get_client()
    index = settings.opensearch_index

    if client.indices.exists(index=index):
        return

    client.indices.create(index=index, body={
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.vector_dimension,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "nmslib"},
                },
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
            }
        },
    })


@opensearch_retry
def index_document(doc_id, embedding, source, title, content, metadata=None):
    settings = get_settings()
    _get_client().index(index=settings.opensearch_index, id=doc_id, body={
        "doc_id": doc_id, "embedding": embedding,
        "source": source, "title": title, "content": content,
    }, refresh="wait_for")


@opensearch_retry
def search_similar(embedding, top_k=None) -> list[SearchResult]:
    settings = get_settings()
    k = top_k or settings.vector_top_k

    response = _get_client().search(index=settings.opensearch_index, body={
        "size": k,
        "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
        "_source": ["doc_id", "source", "title", "content"],
    })

    return [
        SearchResult(
            doc_id=hit["_source"]["doc_id"],
            source=DocumentSource(hit["_source"].get("source", "article")),
            title=hit["_source"].get("title", ""),
            content=hit["_source"].get("content", ""),
            score=hit.get("_score", 0.0),
        )
        for hit in response["hits"]["hits"]
    ]
`;

const ZENDESK_PY = `"""Zendesk REST API client — tickets, articles, replies."""

from __future__ import annotations

import base64
import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import http_retry

_log = get_logger(__name__)


def _auth_header() -> dict[str, str]:
    settings = get_settings()
    credential = f"{settings.zendesk_email}/token:{settings.zendesk_api_token}"
    encoded = base64.b64encode(credential.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@http_retry
async def fetch_tickets(page: int = 1, per_page: int = 100) -> list[dict]:
    url = f"{get_settings().zendesk_base_url}/tickets.json"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_auth_header(), params={"page": page, "per_page": per_page})
        resp.raise_for_status()
    return resp.json().get("tickets", [])


@http_retry
async def fetch_articles(page: int = 1, per_page: int = 100) -> list[dict]:
    settings = get_settings()
    url = f"https://{settings.zendesk_subdomain}.zendesk.com/api/v2/help_center/articles.json"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_auth_header(), params={"page": page, "per_page": per_page})
        resp.raise_for_status()
    return resp.json().get("articles", [])


@http_retry
async def post_reply(ticket_id: int, body: str) -> dict:
    url = f"{get_settings().zendesk_base_url}/tickets/{ticket_id}.json"
    payload = {"ticket": {"comment": {"body": body, "public": True}}}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(url, headers={**_auth_header(), "Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
    return resp.json()
`;

const DEMO_PY = `"""Demo stubs — fake embeddings, LLM, vector search, Zendesk.

Activated when DEMO_MODE=true. Explore the RAG pipeline without real credentials.
"""

from __future__ import annotations

import hashlib
import random
import time

from app.core.logging import get_logger

_log = get_logger(__name__)

DEMO_ARTICLES = [
    {"id": 9001, "title": "How to Reset Your SSO Password",
     "body": "Visit the SSO portal → Forgot Password → link valid 15 min."},
    {"id": 9002, "title": "Billing FAQ — Refunds and Credits",
     "body": "Submit refund within 30 days via Account → Billing → Request Refund."},
    {"id": 9003, "title": "Getting Started With Our API",
     "body": "Generate key at Settings → Developer. Bearer auth. 100 req/min free."},
]

DEMO_TICKETS = [
    {"id": 42001, "subject": "Can't log in after password reset",
     "description": "I keep getting 'invalid credentials' after resetting.", "tags": ["login"]},
    {"id": 42002, "subject": "Refund for double charge",
     "description": "Was charged twice this month.", "tags": ["billing"]},
]


def demo_embed_text(text: str, dimension: int = 1024) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dimension)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


def demo_search_similar(embedding, top_k=3) -> list[dict]:
    return [{"doc_id": f"demo-{a['id']}", "source": "article",
             "title": a["title"], "content": a["body"],
             "score": round(0.95 - i * 0.05, 2)}
            for i, a in enumerate(DEMO_ARTICLES[:top_k])]


def demo_generate_reply(ticket_text: str, context_docs: list[str]) -> str:
    time.sleep(0.3)
    return (
        "Hi there,\\n\\n"
        "Thank you for reaching out. Based on our knowledge base:\\n\\n"
        + "\\n".join(f"  {i}. {d[:80]}..." for i, d in enumerate(context_docs[:3], 1))
        + "\\n\\nPlease try the steps above. Reply if you need more help.\\n\\n"
        "Best regards,\\nSupport (AI-drafted — demo mode)"
    )


async def demo_fetch_tickets(**kw): return DEMO_TICKETS
async def demo_fetch_articles(**kw): return DEMO_ARTICLES
async def demo_post_reply(ticket_id, body):
    _log.info("demo_post_reply", ticket_id=ticket_id)
    return {"status": "ok", "demo": True}
`;

const WEBHOOKS_PY = `"""Antigravity webhook handler."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import APIRouter, Header, HTTPException, Request

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import DraftReplyResponse, ZendeskTicketEvent, SearchResult, DocumentSource

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
_log = get_logger(__name__)


def _verify_signature(payload: bytes, signature: str) -> bool:
    secret = get_settings().antigravity_webhook_secret
    if not secret:
        return True  # allow in dev
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/antigravity", response_model=DraftReplyResponse)
async def handle_antigravity_webhook(
    request: Request,
    x_antigravity_signature: str = Header(default=""),
) -> DraftReplyResponse:
    settings = get_settings()
    raw_body = await request.body()

    if not settings.demo_mode and not _verify_signature(raw_body, x_antigravity_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = ZendeskTicketEvent.model_validate_json(raw_body)
    ticket_text = f"{event.subject}\\n\\n{event.description}"

    if settings.demo_mode:
        from app.services.demo import demo_embed_text, demo_search_similar, demo_generate_reply, demo_post_reply
        embedding = demo_embed_text(ticket_text)
        raw = demo_search_similar(embedding)
        results = [SearchResult(**r) for r in raw]
        draft = demo_generate_reply(ticket_text, [r.content for r in results])
        await demo_post_reply(event.ticket_id, draft)
        posted = True
    else:
        from app.services.bedrock_embeddings import embed_text
        from app.services.bedrock_llm import generate_reply
        from app.services.vector_store import search_similar
        from app.services import zendesk

        embedding = embed_text(ticket_text)
        results = search_similar(embedding)
        draft = generate_reply(ticket_text, [r.content for r in results])
        try:
            await zendesk.post_reply(event.ticket_id, draft)
            posted = True
        except Exception:
            posted = False

    return DraftReplyResponse(
        ticket_id=event.ticket_id, draft_reply=draft,
        context_docs=results, posted_to_zendesk=posted,
    )
`;

const MAIN_PY = `"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.vector_store import ensure_index
from app.api import health, ingest, webhooks

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    _log.info("app_starting", demo_mode=settings.demo_mode)

    if not settings.demo_mode:
        try:
            ensure_index()
        except Exception:
            _log.exception("opensearch_index_init_failed")
    else:
        _log.info("demo_mode_active", msg="Using demo stubs")

    yield
    _log.info("app_shutting_down")


app = FastAPI(
    title="RAG Support System",
    description="Retrieval-Augmented Generation for Zendesk tickets",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(webhooks.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    _log.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    _log.info("http_request", method=request.method, path=request.url.path)
    response = await call_next(request)
    return response
`;

const TEXT_PY = `"""Text chunking utilities."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        split_pos = text.rfind("\\n\\n", start, end)
        if split_pos <= start:
            split_pos = text.rfind(". ", start, end)
        if split_pos <= start:
            split_pos = end
        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        start = max(split_pos - overlap, start + 1)
    return chunks
`;

export const pythonFileSystem: FileSystem = [
    {
        name: "requirements.txt",
        language: "text",
        content: REQUIREMENTS_TXT
    },
    {
        name: ".env.example",
        language: "text",
        content: ENV_EXAMPLE
    },
    {
        name: "app",
        files: [
            {
                name: "main.py",
                language: "python",
                content: MAIN_PY
            },
            {
                name: "core",
                files: [
                    {
                        name: "config.py",
                        language: "python",
                        content: CONFIG_PY
                    },
                    {
                        name: "logging.py",
                        language: "python",
                        content: LOGGING_PY
                    },
                    {
                        name: "retry.py",
                        language: "python",
                        content: RETRY_PY
                    }
                ]
            },
            {
                name: "models",
                files: [
                    {
                        name: "schemas.py",
                        language: "python",
                        content: SCHEMAS_PY
                    }
                ]
            },
            {
                name: "services",
                files: [
                    {
                        name: "bedrock_embeddings.py",
                        language: "python",
                        content: BEDROCK_EMBEDDINGS_PY
                    },
                    {
                        name: "bedrock_llm.py",
                        language: "python",
                        content: BEDROCK_LLM_PY
                    },
                    {
                        name: "vector_store.py",
                        language: "python",
                        content: VECTOR_STORE_PY
                    },
                    {
                        name: "zendesk.py",
                        language: "python",
                        content: ZENDESK_PY
                    },
                    {
                        name: "demo.py",
                        language: "python",
                        content: DEMO_PY
                    }
                ]
            },
            {
                name: "api",
                files: [
                    {
                        name: "webhooks.py",
                        language: "python",
                        content: WEBHOOKS_PY
                    },
                    {
                        name: "health.py",
                        language: "python",
                        content: `"""Health-check endpoint."""

from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()
`
                    }
                ]
            },
            {
                name: "utils",
                files: [
                    {
                        name: "text.py",
                        language: "python",
                        content: TEXT_PY
                    }
                ]
            }
        ]
    }
];
