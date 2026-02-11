"""Zendesk REST API client — tickets, articles, replies."""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import http_retry

_log = get_logger(__name__)


def _auth_header() -> dict[str, str]:
    settings = get_settings()
    # Zendesk API token auth: {email}/token:{api_token}
    import base64

    credential = f"{settings.zendesk_email}/token:{settings.zendesk_api_token}"
    encoded = base64.b64encode(credential.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _base_url() -> str:
    return get_settings().zendesk_base_url


# ── Tickets ──────────────────────────────────────────────


@http_retry
async def fetch_tickets(page: int = 1, per_page: int = 100) -> list[dict]:
    """Fetch a page of recent tickets from Zendesk."""
    url = f"{_base_url()}/tickets.json"
    params = {"page": page, "per_page": per_page, "sort_by": "created_at", "sort_order": "desc"}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_auth_header(), params=params)
        resp.raise_for_status()

    data = resp.json()
    tickets = data.get("tickets", [])
    _log.info("zendesk_fetch_tickets", page=page, count=len(tickets))
    return tickets


# ── Help Center Articles ─────────────────────────────────


@http_retry
async def fetch_articles(page: int = 1, per_page: int = 100) -> list[dict]:
    """Fetch a page of Help Center articles."""
    settings = get_settings()
    url = f"https://{settings.zendesk_subdomain}.zendesk.com/api/v2/help_center/articles.json"
    params = {"page": page, "per_page": per_page, "sort_by": "created_at", "sort_order": "desc"}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_auth_header(), params=params)
        resp.raise_for_status()

    data = resp.json()
    articles = data.get("articles", [])
    _log.info("zendesk_fetch_articles", page=page, count=len(articles))
    return articles


# ── Post Reply ───────────────────────────────────────────


@http_retry
async def post_reply(ticket_id: int, body: str) -> dict:
    """Post a public reply (comment) on a Zendesk ticket."""
    url = f"{_base_url()}/tickets/{ticket_id}.json"
    payload = {
        "ticket": {
            "comment": {
                "body": body,
                "public": True,
            }
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(url, headers={**_auth_header(), "Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()

    _log.info("zendesk_post_reply", ticket_id=ticket_id, reply_len=len(body))
    return resp.json()
