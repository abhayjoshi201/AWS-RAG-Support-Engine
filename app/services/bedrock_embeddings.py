"""AWS Bedrock embedding service using Titan Embed v2."""

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

    _log.debug("bedrock_embed_request", model=settings.bedrock_embedding_model_id, text_len=len(text))

    response = client.invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    embedding = result["embedding"]

    _log.debug("bedrock_embed_response", dim=len(embedding))
    return embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts sequentially.

    Titan Embed v2 does not support native batching, so we iterate.
    """
    _log.info("bedrock_embed_batch", count=len(texts))
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        emb = embed_text(text)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            _log.info("bedrock_embed_batch_progress", done=i + 1, total=len(texts))
    return embeddings
