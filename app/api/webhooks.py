"""Antigravity webhook handler — processes incoming Zendesk ticket events."""

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
    """Validate HMAC-SHA256 signature from Antigravity."""
    secret = get_settings().antigravity_webhook_secret
    if not secret:
        _log.warning("webhook_signature_skip", reason="no secret configured")
        return True  # allow in dev
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/antigravity", response_model=DraftReplyResponse)
async def handle_antigravity_webhook(
    request: Request,
    x_antigravity_signature: str = Header(default=""),
) -> DraftReplyResponse:
    """Receive a new-ticket event, generate a draft reply, and post it back."""
    settings = get_settings()
    raw_body = await request.body()

    # ── Signature verification ───────────────────────────
    if not settings.demo_mode and not _verify_signature(raw_body, x_antigravity_signature):
        _log.warning("webhook_signature_invalid")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # ── Parse event payload ──────────────────────────────
    event = ZendeskTicketEvent.model_validate_json(raw_body)
    _log.info("webhook_received", ticket_id=event.ticket_id, subject=event.subject, demo=settings.demo_mode)

    ticket_text = f"{event.subject}\n\n{event.description}"

    # ── Select real or demo services ─────────────────────
    if settings.demo_mode:
        from app.services.demo import demo_embed_text, demo_search_similar, demo_generate_reply, demo_post_reply

        _log.info("webhook_demo_mode", ticket_id=event.ticket_id)
        embedding = demo_embed_text(ticket_text)
        raw_results = demo_search_similar(embedding)
        results = [
            SearchResult(doc_id=r["doc_id"], source=DocumentSource(r["source"]),
                         title=r["title"], content=r["content"], score=r["score"])
            for r in raw_results
        ]
        context_docs = [r.content for r in results if r.content]
        draft = demo_generate_reply(ticket_text, context_docs)

        posted = False
        try:
            await demo_post_reply(event.ticket_id, draft)
            posted = True
        except Exception:
            _log.exception("webhook_demo_reply_failed", ticket_id=event.ticket_id)
    else:
        from app.services.bedrock_embeddings import embed_text
        from app.services.bedrock_llm import generate_reply
        from app.services.vector_store import search_similar
        from app.services import zendesk

        # ── Embed ticket ─────────────────────────────────
        _log.info("webhook_embedding_ticket", ticket_id=event.ticket_id)
        embedding = embed_text(ticket_text)

        # ── Retrieve similar documents ───────────────────
        _log.info("webhook_searching", ticket_id=event.ticket_id)
        results = search_similar(embedding)
        context_docs = [r.content for r in results if r.content]

        # ── Generate draft reply ─────────────────────────
        _log.info("webhook_generating_reply", ticket_id=event.ticket_id, context_count=len(context_docs))
        draft = generate_reply(ticket_text, context_docs)

        # ── Post reply back to Zendesk ───────────────────
        posted = False
        try:
            await zendesk.post_reply(event.ticket_id, draft)
            posted = True
            _log.info("webhook_reply_posted", ticket_id=event.ticket_id)
        except Exception:
            _log.exception("webhook_reply_post_failed", ticket_id=event.ticket_id)

    return DraftReplyResponse(
        ticket_id=event.ticket_id,
        draft_reply=draft,
        context_docs=results,
        posted_to_zendesk=posted,
    )
