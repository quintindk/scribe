"""
Scribe — The Record Keeper.

A thin FastMCP server that exposes a single `search_archives` tool. It embeds
the query via an OpenAI-compatible endpoint (LM Studio nomic) and searches
across the Chamberlain Qdrant collections written by Miller.

Multi-collection by design: collections matching CHAMBERLAIN_COLLECTION_PREFIX
are discovered at request time. Pass `collection="all"` to fan out, or a bare
suffix like `catchpole` (we'll prefix it).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastmcp import FastMCP
from openai import OpenAI
from qdrant_client import QdrantClient

LOG = logging.getLogger("scribe")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_PREFIX = os.getenv("CHAMBERLAIN_COLLECTION_PREFIX", "chamberlain-")

EMBED_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE", "http://localhost:1234/v1")
EMBED_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed")
EMBED_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "text-embedding-nomic-embed-text-v1.5")

DEFAULT_LIMIT = int(os.getenv("SCRIBE_DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.getenv("SCRIBE_MAX_LIMIT", "25"))

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
embedder = OpenAI(base_url=EMBED_BASE, api_key=EMBED_KEY)

mcp = FastMCP(
    name="scribe",
    instructions=(
        "Scribe is the Chamberlain knowledge archive. Use `search_archives` "
        "to retrieve repository chunks (code, docs, configs) indexed from the "
        "Chamberlain pillars (catchpole, scribe, miller, bailiff, meta)."
    ),
)


def _embed(query: str) -> list[float]:
    resp = embedder.embeddings.create(model=EMBED_MODEL, input=query)
    return resp.data[0].embedding


def _resolve_collections(collection: str) -> list[str]:
    if collection in ("", "all", "*"):
        names = [c.name for c in qdrant.get_collections().collections]
        return sorted(n for n in names if n.startswith(COLLECTION_PREFIX))
    if collection.startswith(COLLECTION_PREFIX):
        return [collection]
    return [f"{COLLECTION_PREFIX}{collection}"]


def _format_hit(hit: Any, collection: str) -> dict[str, Any]:
    payload = hit.payload or {}
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "score": round(float(hit.score), 4),
        "collection": collection,
        "repository": meta.get("repository") or payload.get("repository"),
        "file_path": meta.get("file_path") or payload.get("file_path"),
        "chunk_index": meta.get("chunk_index"),
        "content": payload.get("content") or payload.get("text") or payload.get("page_content", ""),
    }


@mcp.tool
def search_archives(
    query: str,
    collection: str = "all",
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Search the Chamberlain archives for relevant chunks.

    Args:
        query: Natural-language question or keywords.
        collection: "all" (default) to fan out across every chamberlain-* collection,
                    a bare pillar name ("catchpole"), or a full collection name.
        limit: Total chunks to return (capped at SCRIBE_MAX_LIMIT).

    Returns:
        Ranked list of hits with score, repository, file_path and content.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    collections = _resolve_collections(collection)
    if not collections:
        LOG.warning("No matching collections for %r", collection)
        return []

    LOG.info("Query=%r across %d collections (limit=%d)", query[:80], len(collections), limit)
    vector = _embed(query)
    per_collection = max(1, limit if len(collections) == 1 else limit)

    hits: list[dict[str, Any]] = []
    for name in collections:
        try:
            results = qdrant.query_points(
                collection_name=name,
                query=vector,
                limit=per_collection,
                with_payload=True,
            ).points
        except Exception as exc:
            LOG.warning("Search failed for %s: %s", name, exc)
            continue
        hits.extend(_format_hit(h, name) for h in results)

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:limit]


@mcp.tool
def list_collections() -> list[dict[str, Any]]:
    """List all Chamberlain collections with point counts."""
    out = []
    for c in qdrant.get_collections().collections:
        if not c.name.startswith(COLLECTION_PREFIX):
            continue
        info = qdrant.get_collection(c.name)
        out.append({"name": c.name, "points": info.points_count, "status": str(info.status)})
    return sorted(out, key=lambda x: x["name"])


if __name__ == "__main__":
    transport = os.getenv("SCRIBE_TRANSPORT", "sse")
    host = os.getenv("SCRIBE_HOST", "0.0.0.0")
    port = int(os.getenv("SCRIBE_PORT", "8000"))
    LOG.info("Starting Scribe (%s) on %s:%s", transport, host, port)
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=host, port=port)
