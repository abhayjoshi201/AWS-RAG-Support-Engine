"""Pydantic models / DTOs used across the application."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────


class DocumentSource(str, Enum):
    TICKET = "ticket"
    ARTICLE = "article"


# ── Zendesk Webhook Payload ──────────────────────────────


class ZendeskTicketEvent(BaseModel):
    """Payload received from Antigravity webhook for a new Zendesk ticket."""

    ticket_id: int
    subject: str = ""
    description: str = ""
    requester_email: str = ""
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    priority: str | None = None
    status: str | None = None


# ── Ingestion ────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for manual ingestion triggers."""

    max_pages: int = Field(default=10, ge=1, le=100)


class IngestedDocument(BaseModel):
    doc_id: str
    source: DocumentSource
    title: str = ""
    content: str = ""
    metadata: dict = Field(default_factory=dict)


# ── Vector Search ────────────────────────────────────────


class SearchResult(BaseModel):
    doc_id: str
    source: DocumentSource
    title: str = ""
    content: str = ""
    score: float = 0.0


# ── Response DTOs ────────────────────────────────────────


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
