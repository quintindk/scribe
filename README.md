# Scribe — The Record Keeper

Scribe is the inference-layer memory of [Chamberlain](../README.md). It exposes a small FastMCP server that LM Studio attaches to as a tool provider, so the model itself can pull repository context out of the Qdrant archives that [Miller](../miller/) keeps fresh.

> Scribe is NOT called by clients directly. The flow is:
> `Bailiff → Catchpole → LM Studio → Scribe → Qdrant`.
> Scribe sits next to the inference engine, not next to the agent.

## Architecture

```
LM Studio  ──(MCP/SSE)──►  Scribe  ──►  LM Studio /v1/embeddings  (nomic-embed-text-v1.5)
                              │
                              └──►  Qdrant (chamberlain-* collections written by Miller)
```

The same `nomic-embed-text-v1.5` model Miller used for ingestion is used for query embedding, so vector geometry matches end to end.

## Why not vanilla `mcp-server-qdrant`?

The upstream qdrant/mcp-server-qdrant project supports only:

1. FastEmbed for embeddings (no OpenAI-compatible endpoint, so no LM Studio).
2. Named vectors (`fast-<model>`) — Miller wrote unnamed default vectors.
3. A single collection per server instance — we have one collection per pillar.

Scribe is ~120 lines of Python that solves all three: OpenAI-compatible embeddings, unnamed-vector search, and multi-collection fan-out.

## Tools

| Tool | Description |
| --- | --- |
| `search_archives(query, collection="all", limit=5)` | Embed the query and search across one or all `chamberlain-*` collections, ranked by score. |
| `list_collections()` | Enumerate the Chamberlain collections with point counts and status. |

`collection` accepts:

- `"all"` / `"*"` — fan out across every `chamberlain-*` collection (default).
- A bare pillar name like `"catchpole"` — prefixed automatically.
- A fully qualified collection name like `"chamberlain-catchpole"`.

## Quick start

```bash
cp .env.example .env
# Set OPENAI_COMPATIBLE_API_KEY to your LM Studio key, leave the rest as-is
docker compose up -d
docker compose logs -f scribe
```

Scribe will be listening on `http://localhost:8000/sse`.

## LM Studio wiring

Add the following to LM Studio's `mcp.json`:

```json
{
  "mcpServers": {
    "scribe": {
      "url": "http://localhost:8000/sse",
      "transport": "sse"
    }
  }
}
```

Restart LM Studio. When a chat request comes in, LM Studio will see the `search_archives` tool and can call it autonomously when context is needed.

## Configuration

All settings come from environment variables (see `.env.example`):

| Var | Default | Notes |
| --- | --- | --- |
| `SCRIBE_TRANSPORT` | `sse` | `sse` or `stdio`. |
| `SCRIBE_HOST` / `SCRIBE_PORT` | `0.0.0.0` / `8000` | Bind address for SSE. |
| `QDRANT_URL` | `http://localhost:6333` | Must point at the same Qdrant Miller writes to. |
| `QDRANT_API_KEY` | _unset_ | Leave blank for local. |
| `CHAMBERLAIN_COLLECTION_PREFIX` | `chamberlain-` | Discovery filter. |
| `OPENAI_COMPATIBLE_API_BASE` | `http://localhost:1234/v1` | LM Studio. |
| `OPENAI_COMPATIBLE_API_KEY` | `not-needed` | LM Studio key. |
| `OPENAI_COMPATIBLE_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Must match Miller's embedding model. |
| `SCRIBE_DEFAULT_LIMIT` / `SCRIBE_MAX_LIMIT` | `5` / `25` | Result count guards. |
| `LOG_LEVEL` | `INFO` | Standard Python log level. |

## Local smoke test (no LM Studio MCP client needed)

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/sse") as c:
        r = await c.call_tool("search_archives", {"query": "how does the router decide local vs cloud", "limit": 3})
        for h in r.data:
            print(f"[{h['score']:.3f}] {h['collection']} :: {h['file_path']}")

asyncio.run(main())
```

## See also

- The [Chamberlain Architecture specification](../specification.md).
- [Miller](../miller/) — the ingester that writes the collections Scribe reads.
- [Catchpole](../catchpole/) — the gateway upstream of LM Studio.
