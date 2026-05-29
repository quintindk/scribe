# Scribe

Scribe is a FastMCP server that gives an inference engine a knowledge layer over Qdrant. It exposes three groups of tools — **search**, **ingest**, and **memory** — so the model itself can pull repository context, load ad-hoc URLs or local folders into ephemeral scratch pads, and persist small curated facts for future recall, all without burning client-side context tokens.

The architectural contract for Scribe lives in [`../specification.md`](../specification.md) §4.

## Architecture

```
Inference engine  ──(MCP / HTTP)──►  Scribe  ──►  OpenAI-compatible /v1/embeddings
                                        │
                                        ├──►  archive-*      (Miller-managed, persistent)
                                        ├──►  scratch-*      (ingest_*, ephemeral)
                                        └──►  memory-default (remember/recall, curated)
```

The same embedding model used to write a collection must be used to query it.

## Collection layout

| Prefix          | Lifetime              | Source                          | Granularity        | Mutability                     |
| --------------- | --------------------- | ------------------------------- | ------------------ | ------------------------------ |
| `archive-*`     | Persistent            | Miller (cron, GitHub repos)     | Chunks of files    | Rebuilt wholesale by Miller    |
| `scratch-*`     | Ephemeral (manual GC) | Scribe `ingest_*` tools         | Chunks of pages    | Delete-whole-collection        |
| `memory-default`| Persistent, curated   | Scribe `remember` tool          | One fact per point | Per-entry insert / forget      |

`search_archives` fans out across `archive-*` and `scratch-*`. Memories are queried explicitly via `recall`.

## Tools

### Search (read-only)

| Tool | Description |
| --- | --- |
| `search_archives(query, collection="all", limit=5)` | Embed the query and search across archive-\* and scratch-\* collections. |
| `list_collections()` | Enumerate archive and scratch collections with point counts, kind, and status. |

### Ingest (writes `scratch-*`)

| Tool | Description |
| --- | --- |
| `ingest_url(url, collection, max_pages=1)` | Fetch a single URL, extract main text via trafilatura, chunk, embed, upsert. SSRF-gated. `max_pages > 1` is rejected in v1. |
| `ingest_path(path, collection)` | Read text files under `/drop/<path>`, chunk, embed, upsert. Path must stay inside the `/drop` mount. |
| `forget_collection(collection)` | Delete a `scratch-*` collection in full. Refuses to touch `archive-*` or `memory-*`. |

### Memory (writes `memory-default`, point-granular)

| Tool | Description |
| --- | --- |
| `remember(fact, subject, reason, citations)` | Store one curated fact. All four fields mandatory; `fact` capped at 200 chars. Refuses secrets and GDPR Art. 9 categories. |
| `recall(query, subject=None, limit=5)` | Semantic search over `memory-default`, optionally filtered by subject. Returns full payloads, not synthesised prose. |
| `forget(memory_id, reason)` | Delete one memory by id. `reason` is written to the audit log on stderr, not stored. |
| `list_memories(subject=None, limit=100)` | Enumerate memories, optionally filtered by subject. |

`collection` arguments accept either a bare suffix (e.g. `docs`) or a fully prefixed name (e.g. `scratch-docs`). For `search_archives` only, a bare suffix defaults to the archive prefix for back-compat.

## Safety gates (non-negotiable)

These are spec-mandated and live in Scribe itself, not the caller:

- **SSRF allowlist on `ingest_url`.** Scheme must be `http`/`https`. Resolved IPs must be globally routable (`ipaddress.is_global`). Redirects are not followed in v1. Response size and timeout are capped. Override with `SCRIBE_INGEST_ALLOW_PRIVATE=true` for dev only.
- **Path containment on `ingest_path`.** Resolved real path must be a descendant of `/drop` (rejects symlink escapes).
- **Secret and PII refusal on `remember`.** Regex set covers PEM keys, JWTs, GitHub PATs, OpenAI/Anthropic/Google/Slack/Stripe tokens, Azure storage keys, DB URLs with embedded credentials, and bearer tokens. A heuristic GDPR Art. 9 keyword gate refuses health, religion, ethnicity, political, sexual orientation, union, and biometric content. Refusal is loud (error string visible to the model).

The audit log for `forget` and `forget_collection` is stderr (captured by Docker), grep-friendly via the `[AUDIT]` prefix.

## Quick start

```bash
cp .env.example .env
# Edit the embedder + Qdrant URL if needed.
mkdir -p drop                 # host directory mounted read-only at /drop
docker compose up -d
docker compose logs -f scribe
```

Scribe listens on `http://localhost:8000/mcp`. Use `SCRIBE_TRANSPORT=sse` for legacy SSE or `SCRIBE_TRANSPORT=stdio` for direct embedding.

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

Restart LM Studio. The model sees all eight Scribe tools. **Usage policy** (when to call `recall` vs `search_archives`, when `remember` is appropriate) lives in the calling agent's system prompt (Bailiff's `BAILIFF_SYSTEM_PROMPT_FILE`), not in Scribe.

## Configuration

| Var | Default | Notes |
| --- | --- | --- |
| `SCRIBE_TRANSPORT` | `http` | `http`, `sse`, or `stdio`. |
| `SCRIBE_HOST` / `SCRIBE_PORT` / `SCRIBE_PATH` | `0.0.0.0` / `8000` / `/mcp` | HTTP bind. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance. |
| `QDRANT_API_KEY` | _unset_ | Leave blank for local. |
| `SCRIBE_ARCHIVE_PREFIX` | `archive-` | Falls back to legacy `COLLECTION_PREFIX` if unset. |
| `SCRIBE_SCRATCH_PREFIX` | `scratch-` | Prefix for ingest-created collections. |
| `SCRIBE_MEMORY_COLLECTION` | `memory-default` | Curated memory collection name. |
| `OPENAI_COMPATIBLE_API_BASE` / `_KEY` / `_MODEL` | `http://localhost:1234/v1` / `not-needed` / `text-embedding-nomic-embed-text-v1.5` | Embedding endpoint. Must match Miller. |
| `SCRIBE_DEFAULT_LIMIT` / `SCRIBE_MAX_LIMIT` | `5` / `25` | Search result count guards. |
| `SCRIBE_DROP_DIR` | `./drop` | Host directory mounted read-only at `/drop`. |
| `SCRIBE_INGEST_ALLOW_PRIVATE` | `false` | Disable SSRF gate. **Dev only.** |
| `SCRIBE_INGEST_MAX_BYTES` | `5242880` (5 MB) | `ingest_url` response size cap. |
| `SCRIBE_INGEST_TIMEOUT` | `30` | `ingest_url` per-call timeout in seconds. |
| `SCRIBE_INGEST_PATH_MAX_FILES` | `500` | `ingest_path` file count cap. |
| `SCRIBE_INGEST_PATH_MAX_BYTES` | `26214400` (25 MB) | `ingest_path` total bytes cap. |
| `SCRIBE_CHUNK_CHARS` / `SCRIBE_CHUNK_OVERLAP` | `1000` / `200` | Chunker window. |
| `LOG_LEVEL` | `INFO` | Standard Python log level. |

## Out of scope for v1

By design (see specification.md §4.4): `update_memory`, memory voting, multi-namespace memory, automatic age-based GC, multi-tenant isolation, and multi-page (BFS) crawling for `ingest_url`.

## Local smoke test

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as c:
        # Search the corpus.
        r = await c.call_tool("search_archives", {"query": "your question", "limit": 3})
        for h in r.data:
            print(f"[{h['score']:.3f}] {h['collection']} :: {h['file_path']}")

        # Store a memory.
        m = await c.call_tool("remember", {
            "fact": "Bailiff owns the upstream system prompt for LM Studio.",
            "subject": "architecture",
            "reason": "Future tasks editing system-prompt policy should look at Bailiff first.",
            "citations": "specification.md §6.1",
        })
        print("stored:", m.data["id"])

        # Recall it.
        r = await c.call_tool("recall", {"query": "who owns the system prompt?"})
        for hit in r.data:
            print(f"[{hit['score']:.3f}] {hit['fact']}  (cites: {hit['citations']})")

asyncio.run(main())
```

## Requirements

- Python 3.10+ (deps in `requirements.txt`).
- A Qdrant instance reachable from Scribe.
- An OpenAI-compatible embeddings endpoint exposing the same model used to write the collections.
