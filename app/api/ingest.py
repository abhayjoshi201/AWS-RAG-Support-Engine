"""Ingestion endpoints — bulk-load Zendesk tickets and articles into the vector store."""

from __future__ import annotations

import hashlib
import re

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import DocumentSource, IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])
_log = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text) if text else ""


def _doc_id(source: str, external_id: str | int, chunk_idx: int) -> str:
    raw = f"{source}:{external_id}:{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Tickets ──────────────────────────────────────────────


@router.post("/tickets", response_model=IngestResponse)
async def ingest_tickets(body: IngestRequest | None = None) -> IngestResponse:
    """Fetch tickets from Zendesk, chunk, embed, and index."""
    settings = get_settings()
    max_pages = body.max_pages if body else 10
    _log.info("ingest_tickets_start", max_pages=max_pages, demo=settings.demo_mode)

    from app.utils.text import chunk_text

    if settings.demo_mode:
        from app.services.demo import demo_fetch_tickets, demo_embed_batch
        fetch_fn = demo_fetch_tickets
        embed_fn = demo_embed_batch
    else:
        from app.services import zendesk
        from app.services.bedrock_embeddings import embed_batch
        fetch_fn = zendesk.fetch_tickets
        embed_fn = embed_batch

    all_docs: list[dict] = []

    for page in range(1, max_pages + 1):
        tickets = await fetch_fn(page=page)
        if not tickets:
            break

        for ticket in tickets:
            text = f"{ticket.get('subject', '')}\n\n{ticket.get('description', '')}"
            text = _strip_html(text).strip()
            if not text:
                continue

            chunks = chunk_text(text)
            embeddings = embed_fn(chunks)

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                all_docs.append({
                    "doc_id": _doc_id("ticket", ticket["id"], idx),
                    "embedding": emb,
                    "source": DocumentSource.TICKET.value,
                    "title": ticket.get("subject", ""),
                    "content": chunk,
                    "metadata": {
                        "ticket_id": ticket["id"],
                        "chunk_index": idx,
                        "tags": ticket.get("tags", []),
                    },
                })

    # Index (skip in demo mode — no real vector store)
    if settings.demo_mode:
        indexed = len(all_docs)
        _log.info("ingest_tickets_demo_done", docs=indexed)
    else:
        from app.services.vector_store import index_documents
        indexed = index_documents(all_docs)

    _log.info("ingest_tickets_done", indexed=indexed)
    return IngestResponse(
        source=DocumentSource.TICKET,
        documents_ingested=indexed,
        message=f"Ticket ingestion complete{' (demo)' if settings.demo_mode else ''}",
    )


# ── Articles ─────────────────────────────────────────────


@router.post("/articles", response_model=IngestResponse)
async def ingest_articles(body: IngestRequest | None = None) -> IngestResponse:
    """Fetch Help Center articles from Zendesk, chunk, embed, and index."""
    settings = get_settings()
    max_pages = body.max_pages if body else 10
    _log.info("ingest_articles_start", max_pages=max_pages, demo=settings.demo_mode)

    from app.utils.text import chunk_text

    if settings.demo_mode:
        from app.services.demo import demo_fetch_articles, demo_embed_batch
        fetch_fn = demo_fetch_articles
        embed_fn = demo_embed_batch
    else:
        from app.services import zendesk
        from app.services.bedrock_embeddings import embed_batch
        fetch_fn = zendesk.fetch_articles
        embed_fn = embed_batch

    all_docs: list[dict] = []

    for page in range(1, max_pages + 1):
        articles = await fetch_fn(page=page)
        if not articles:
            break

        for article in articles:
            text = f"{article.get('title', '')}\n\n{_strip_html(article.get('body', ''))}"
            text = text.strip()
            if not text:
                continue

            chunks = chunk_text(text)
            embeddings = embed_fn(chunks)

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                all_docs.append({
                    "doc_id": _doc_id("article", article["id"], idx),
                    "embedding": emb,
                    "source": DocumentSource.ARTICLE.value,
                    "title": article.get("title", ""),
                    "content": chunk,
                    "metadata": {
                        "article_id": article["id"],
                        "chunk_index": idx,
                        "section_id": article.get("section_id"),
                    },
                })

    if settings.demo_mode:
        indexed = len(all_docs)
        _log.info("ingest_articles_demo_done", docs=indexed)
    else:
        from app.services.vector_store import index_documents
        indexed = index_documents(all_docs)

    _log.info("ingest_articles_done", indexed=indexed)
    return IngestResponse(
        source=DocumentSource.ARTICLE,
        documents_ingested=indexed,
        message=f"Article ingestion complete{' (demo)' if settings.demo_mode else ''}",
    )
