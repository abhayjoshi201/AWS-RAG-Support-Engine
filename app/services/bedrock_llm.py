"""AWS Bedrock LLM generation service using Claude 3 Sonnet."""

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
    "reply to the customer's support ticket. If the context does not contain enough "
    "information, say so honestly. Keep the reply concise and professional."
)


@bedrock_retry
def generate_reply(ticket_text: str, context_docs: list[str]) -> str:
    """Generate a draft reply for a support ticket using retrieved context."""
    settings = get_settings()
    client = _get_client()

    context_block = "\n---\n".join(context_docs) if context_docs else "(no context available)"

    user_message = (
        f"## Retrieved Context\n{context_block}\n\n"
        f"## Customer Ticket\n{ticket_text}\n\n"
        "Draft a helpful reply to the customer."
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
    })

    _log.info(
        "bedrock_llm_request",
        model=settings.bedrock_llm_model_id,
        ticket_len=len(ticket_text),
        context_docs=len(context_docs),
    )

    response = client.invoke_model(
        modelId=settings.bedrock_llm_model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())

    # Claude Messages API returns content as a list of blocks
    reply_text = ""
    for block in result.get("content", []):
        if block.get("type") == "text":
            reply_text += block["text"]

    _log.info("bedrock_llm_response", reply_len=len(reply_text))
    return reply_text.strip()
