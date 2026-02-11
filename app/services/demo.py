"""Demo/dry-run stubs — fake embeddings, LLM generation, vector search, and Zendesk API.

Activated when DEMO_MODE=true. Lets anyone explore the full RAG pipeline
without AWS credentials, an OpenSearch cluster, or a Zendesk account.
"""

from __future__ import annotations

import hashlib
import random
import time

from app.core.logging import get_logger

_log = get_logger(__name__)

# ── Fake Zendesk Data ────────────────────────────────────

DEMO_ARTICLES = [
    {
        "id": 9001,
        "title": "How to Reset Your SSO Password",
        "body": (
            "If your SSO password link has expired, visit the SSO portal and click "
            "'Forgot Password' to receive a new link. The link is valid for 15 minutes. "
            "Check your spam folder if you don't see the email. Contact IT if the issue persists."
        ),
    },
    {
        "id": 9002,
        "title": "Billing FAQ — Refunds and Credits",
        "body": (
            "Refund requests must be submitted within 30 days of purchase. Navigate to "
            "Account → Billing → Request Refund. Credits are applied within 3-5 business days. "
            "Enterprise customers should contact their account manager directly."
        ),
    },
    {
        "id": 9003,
        "title": "Getting Started With Our API",
        "body": (
            "Generate an API key from Settings → Developer → API Keys. All endpoints require "
            "Bearer token authentication. Rate limits are 100 requests/minute for free-tier "
            "and 1000/minute for paid plans. See our OpenAPI spec at /docs for details."
        ),
    },
    {
        "id": 9004,
        "title": "Two-Factor Authentication Setup",
        "body": (
            "Enable 2FA under Account → Security → Two-Factor Authentication. We support "
            "TOTP apps (Google Authenticator, Authy) and SMS. Recovery codes are generated "
            "during setup — store them securely. Admins can enforce 2FA for all users."
        ),
    },
    {
        "id": 9005,
        "title": "Troubleshooting Login Errors",
        "body": (
            "Common login errors: 'Account locked' — wait 30 minutes or contact support. "
            "'Invalid credentials' — ensure Caps Lock is off and try resetting your password. "
            "'SSO redirect loop' — clear cookies and try an incognito window."
        ),
    },
]

DEMO_TICKETS = [
    {
        "id": 42001,
        "subject": "Can't log in after password reset",
        "description": "I reset my password yesterday but now I keep getting 'invalid credentials' when I try to log in.",
        "tags": ["login", "password"],
        "requester_email": "alice@example.com",
    },
    {
        "id": 42002,
        "subject": "Refund for double charge",
        "description": "I was charged twice for my subscription this month. Please issue a refund for the duplicate charge.",
        "tags": ["billing", "refund"],
        "requester_email": "bob@example.com",
    },
    {
        "id": 42003,
        "subject": "API rate limit question",
        "description": "We're hitting rate limits on the free tier. What are the limits for paid plans?",
        "tags": ["api", "rate-limit"],
        "requester_email": "carol@example.com",
    },
]


# ── Demo Embedding ───────────────────────────────────────


def demo_embed_text(text: str, dimension: int = 1024) -> list[float]:
    """Return a deterministic pseudo-random embedding based on text hash."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dimension)]
    # L2 normalise
    norm = sum(v * v for v in vec) ** 0.5
    vec = [v / norm for v in vec]
    _log.info("demo_embed", text_len=len(text), dim=dimension)
    return vec


def demo_embed_batch(texts: list[str], dimension: int = 1024) -> list[list[float]]:
    return [demo_embed_text(t, dimension) for t in texts]


# ── Demo Vector Search ───────────────────────────────────


def demo_search_similar(embedding: list[float], top_k: int = 3) -> list[dict]:
    """Return top_k demo articles as fake search results with mock scores."""
    results = []
    for i, article in enumerate(DEMO_ARTICLES[:top_k]):
        results.append({
            "doc_id": f"demo-article-{article['id']}",
            "source": "article",
            "title": article["title"],
            "content": article["body"],
            "score": round(0.95 - i * 0.05, 2),
        })
    _log.info("demo_search", top_k=top_k, results=len(results))
    return results


# ── Demo LLM Generation ─────────────────────────────────


def demo_generate_reply(ticket_text: str, context_docs: list[str]) -> str:
    """Return a canned but realistic-looking draft reply."""
    # Simulate a short delay like a real LLM call
    time.sleep(0.3)

    context_summary = ""
    if context_docs:
        context_summary = (
            "Based on our knowledge base, I found the following information that may help:\n\n"
        )
        for i, doc in enumerate(context_docs[:3], 1):
            snippet = doc[:120].replace("\n", " ")
            context_summary += f"  {i}. {snippet}...\n"
        context_summary += "\n"

    reply = (
        f"Hi there,\n\n"
        f"Thank you for reaching out to us.\n\n"
        f"{context_summary}"
        f"Here are the recommended next steps:\n\n"
        f"1. Please try the solution described in the relevant article above.\n"
        f"2. If the issue persists, reply to this ticket with any error messages you see.\n"
        f"3. Our team will follow up within 24 hours.\n\n"
        f"Best regards,\n"
        f"Support Team (AI-drafted via RAG system — demo mode)"
    )
    _log.info("demo_generate", reply_len=len(reply))
    return reply


# ── Demo Zendesk Client ──────────────────────────────────


async def demo_fetch_tickets(page: int = 1, per_page: int = 100) -> list[dict]:
    _log.info("demo_fetch_tickets", page=page)
    return DEMO_TICKETS


async def demo_fetch_articles(page: int = 1, per_page: int = 100) -> list[dict]:
    _log.info("demo_fetch_articles", page=page)
    return DEMO_ARTICLES


async def demo_post_reply(ticket_id: int, body: str) -> dict:
    _log.info("demo_post_reply", ticket_id=ticket_id, reply_len=len(body))
    return {"status": "ok", "demo": True, "ticket_id": ticket_id}
