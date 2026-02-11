"""OpenSearch vector store — indexing and kNN similarity search."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.retry import opensearch_retry
from app.models.schemas import SearchResult, DocumentSource

_log = get_logger(__name__)
_os_client: OpenSearch | None = None


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


# ── Index Management ─────────────────────────────────────


@opensearch_retry
def ensure_index() -> None:
    """Create the kNN index if it does not already exist."""
    settings = get_settings()
    client = _get_client()
    index = settings.opensearch_index

    if client.indices.exists(index=index):
        _log.info("opensearch_index_exists", index=index)
        return

    body: dict[str, Any] = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 1,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 48,
                        },
                    },
                },
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "metadata": {"type": "object", "enabled": False},
            }
        },
    }

    client.indices.create(index=index, body=body)
    _log.info("opensearch_index_created", index=index, dim=settings.vector_dimension)


# ── Indexing ─────────────────────────────────────────────


@opensearch_retry
def index_document(
    doc_id: str,
    embedding: list[float],
    source: str,
    title: str,
    content: str,
    metadata: dict | None = None,
) -> None:
    """Index a single document with its embedding vector."""
    settings = get_settings()
    client = _get_client()

    body = {
        "doc_id": doc_id,
        "embedding": embedding,
        "source": source,
        "title": title,
        "content": content,
        "metadata": metadata or {},
    }

    client.index(index=settings.opensearch_index, id=doc_id, body=body, refresh="wait_for")
    _log.debug("opensearch_indexed", doc_id=doc_id)


def index_documents(docs: list[dict]) -> int:
    """Bulk-index a list of document dicts.

    Each dict must have: doc_id, embedding, source, title, content.
    Returns the number of successfully indexed documents.
    """
    count = 0
    for doc in docs:
        try:
            index_document(
                doc_id=doc["doc_id"],
                embedding=doc["embedding"],
                source=doc["source"],
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata"),
            )
            count += 1
        except Exception:
            _log.exception("opensearch_index_failed", doc_id=doc.get("doc_id"))
    _log.info("opensearch_bulk_indexed", total=len(docs), success=count)
    return count


# ── Search ───────────────────────────────────────────────


@opensearch_retry
def search_similar(embedding: list[float], top_k: int | None = None) -> list[SearchResult]:
    """Return top-k most similar documents for the given embedding."""
    settings = get_settings()
    client = _get_client()
    k = top_k or settings.vector_top_k

    query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": k,
                }
            }
        },
        "_source": ["doc_id", "source", "title", "content"],
    }

    response = client.search(index=settings.opensearch_index, body=query)

    results: list[SearchResult] = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append(
            SearchResult(
                doc_id=src["doc_id"],
                source=DocumentSource(src.get("source", "article")),
                title=src.get("title", ""),
                content=src.get("content", ""),
                score=hit.get("_score", 0.0),
            )
        )

    _log.info("opensearch_search", top_k=k, results=len(results))
    return results
