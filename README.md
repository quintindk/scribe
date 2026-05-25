# Scribe

Scribe is a small FastMCP server that exposes a `search_archives` tool and a `list_collections` tool over a Qdrant vector database. It is designed to be attached to an inference engine (e.g. LM Studio) as an MCP tool provider so the model itself can pull repository context out of indexed archives without burning client-side context tokens.

## Architecture

```
Inference engine  ──(MCP / HTTP or SSE)──►  Scribe  ──►  OpenAI-compatible /v1/embeddings
                                                │
                                                └──►  Qdrant collections (matching COLLECTION_PREFIX)
```

The same embedding model used to ingest the data must be used for query embedding, so vector geometry matches end to end.

## Why not vanilla `mcp-server-qdrant`?

The upstream `qdrant/mcp-server-qdrant` project supports only:

1. FastEmbed for embeddings (no OpenAI-compatible endpoint).
2. Named vectors (`fast-<model>`) — many ingesters write unnamed default vectors.
3. A single collection per server instance.

Scribe is ~140 lines of Python that solves all three: OpenAI-compatible embeddings, unnamed-vector search, and multi-collection fan-out.

## Tools

| Tool | Description |
| --- | --- |
| `search_archives(query, collection="all", limit=5)` | Embed the query and search across one or all collections matching `COLLECTION_PREFIX`, ranked by score. |
| `list_collections()` | Enumerate the indexed collections with point counts and status. |

`collection` accepts:

- `"all"` / `"*"` — fan out across every collection whose name starts with the configured prefix (default).
- A bare suffix like `"docs"` — the prefix is added automatically.
- A fully qualified collection name like `"archive-docs"`.

## Quick start

```bash
cp .env.example .env
# edit the embedder + Qdrant URL if needed
docker compose up -d
docker compose logs -f scribe
```

Scribe listens on `http://localhost:8000/mcp` (streamable-http). Set `SCRIBE_TRANSPORT=sse` for legacy SSE on `/sse`, or `SCRIBE_TRANSPORT=stdio` for direct embedding.

## Attaching to LM Studio

Add to LM Studio's `mcp.json`:

```json
{
  "mcpServers": {
    "scribe": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Restart LM Studio. When a chat request comes in, LM Studio will see the `search_archives` tool and can call it autonomously when context is needed.

## Configuration

All settings come from environment variables (see `.env.example`):

| Var | Default | Notes |
| --- | --- | --- |
| `SCRIBE_TRANSPORT` | `http` | `http`, `sse`, or `stdio`. |
| `SCRIBE_HOST` / `SCRIBE_PORT` / `SCRIBE_PATH` | `0.0.0.0` / `8000` / `/mcp` | HTTP bind. |
| `QDRANT_URL` | `http://localhost:6333` | Point at the Qdrant instance holding the indexed data. |
| `QDRANT_API_KEY` | _unset_ | Leave blank for local. |
| `COLLECTION_PREFIX` | `archive-` | Only collections starting with this string are discovered. |
| `OPENAI_COMPATIBLE_API_BASE` | `http://localhost:1234/v1` | Embedding endpoint. |
| `OPENAI_COMPATIBLE_API_KEY` | `not-needed` | Most local servers ignore this. |
| `OPENAI_COMPATIBLE_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Must match the ingestion model. |
| `SCRIBE_DEFAULT_LIMIT` / `SCRIBE_MAX_LIMIT` | `5` / `25` | Result count guards. |
| `LOG_LEVEL` | `INFO` | Standard Python log level. |

## Local smoke test

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as c:
        r = await c.call_tool("search_archives", {"query": "your question here", "limit": 3})
        for h in r.data:
            print(f"[{h['score']:.3f}] {h['collection']} :: {h['file_path']}")

asyncio.run(main())
```

## Requirements

- Python 3.10+ (deps in `requirements.txt`).
- A Qdrant instance reachable from Scribe.
- An OpenAI-compatible embeddings endpoint that exposes the same model used to write the collections.
