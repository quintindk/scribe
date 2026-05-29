"""
Scribe — a knowledge MCP server backed by Qdrant.

A FastMCP server exposing three groups of tools:

* Search (read-only):  `search_archives`, `list_collections`
* Ingest (writes scratch-*):  `ingest_url`, `ingest_path`, `forget_collection`
* Memory (writes memory-default, point-granular):
    `remember`, `recall`, `forget`, `list_memories`

Architectural contract lives in chamberlain/specification.md §4. Tool docstrings
are purely descriptive; any "recall before search" or similar usage policy
belongs in the calling agent's system prompt, not here.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import socket
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from fastmcp import FastMCP
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

LOG = logging.getLogger("scribe")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --- Config ----------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

ARCHIVE_PREFIX = os.getenv(
    "SCRIBE_ARCHIVE_PREFIX",
    os.getenv("COLLECTION_PREFIX", "archive-"),  # back-compat
)
SCRATCH_PREFIX = os.getenv("SCRIBE_SCRATCH_PREFIX", "scratch-")
MEMORY_COLLECTION = os.getenv("SCRIBE_MEMORY_COLLECTION", "memory-default")
SEARCH_PREFIXES = (ARCHIVE_PREFIX, SCRATCH_PREFIX)

EMBED_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE", "http://localhost:1234/v1")
EMBED_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed")
EMBED_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "text-embedding-nomic-embed-text-v1.5")

DEFAULT_LIMIT = int(os.getenv("SCRIBE_DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.getenv("SCRIBE_MAX_LIMIT", "25"))

# Ingest
INGEST_DROP_DIR = Path(os.getenv("SCRIBE_DROP_DIR", "/drop")).resolve()
INGEST_ALLOW_PRIVATE = os.getenv("SCRIBE_INGEST_ALLOW_PRIVATE", "false").lower() == "true"
INGEST_MAX_BYTES = int(os.getenv("SCRIBE_INGEST_MAX_BYTES", str(5 * 1024 * 1024)))
INGEST_TIMEOUT = float(os.getenv("SCRIBE_INGEST_TIMEOUT", "30"))
INGEST_PATH_MAX_FILES = int(os.getenv("SCRIBE_INGEST_PATH_MAX_FILES", "500"))
INGEST_PATH_MAX_BYTES = int(os.getenv("SCRIBE_INGEST_PATH_MAX_BYTES", str(25 * 1024 * 1024)))

CHUNK_CHARS = int(os.getenv("SCRIBE_CHUNK_CHARS", "1000"))
CHUNK_OVERLAP = int(os.getenv("SCRIBE_CHUNK_OVERLAP", "200"))

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".htm", ".css", ".sh", ".bash", ".zsh",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".cs", ".swift", ".kt", ".scala",
    ".sql", ".bicep", ".tf", ".tfvars",
}

ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")

# Slug for collection-name validation: lowercase alnum + hyphen, 1-63 chars
COLLECTION_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

# --- Globals ---------------------------------------------------------------

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
embedder = OpenAI(base_url=EMBED_BASE, api_key=EMBED_KEY)

_memory_lock = threading.Lock()
_memory_ready = False
_embed_dim: int | None = None

mcp = FastMCP(
    name="scribe",
    instructions=(
        "Scribe is a knowledge layer over Qdrant. It exposes three groups of "
        "tools: search (search_archives, list_collections), ingest "
        "(ingest_url, ingest_path, forget_collection), and memory "
        "(remember, recall, forget, list_memories). Search covers archive-* "
        "and scratch-* collections; memories must be queried explicitly via "
        "recall."
    ),
)


# --- Embedding -------------------------------------------------------------

def _embed(text: str) -> list[float]:
    resp = embedder.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _embed_dimension() -> int:
    global _embed_dim
    if _embed_dim is None:
        _embed_dim = len(_embed("dim-probe"))
        LOG.info("Embedder dimension: %d", _embed_dim)
    return _embed_dim


# --- Collection naming -----------------------------------------------------

def _resolve_search_collections(collection: str) -> list[str]:
    """Resolve user-facing `collection` arg to a list of real Qdrant names.

    Fan-out covers archive-* and scratch-*; memory-* is excluded.
    """
    names = [c.name for c in qdrant.get_collections().collections]
    if collection in ("", "all", "*"):
        return sorted(n for n in names if n.startswith(SEARCH_PREFIXES))
    if collection.startswith(SEARCH_PREFIXES):
        return [collection]
    # Back-compat: bare suffix defaults to archive prefix.
    return [f"{ARCHIVE_PREFIX}{collection}"]


def _scratch_collection_name(collection: str) -> str:
    """Validate and canonicalise a user-supplied scratch collection name."""
    if not collection:
        raise ValueError("collection name must not be empty")
    if collection.startswith(ARCHIVE_PREFIX) or collection.startswith(MEMORY_COLLECTION):
        raise ValueError(
            f"refusing to touch non-scratch collection {collection!r}; "
            f"ingest/forget only operate on {SCRATCH_PREFIX}* collections"
        )
    bare = collection[len(SCRATCH_PREFIX):] if collection.startswith(SCRATCH_PREFIX) else collection
    if not COLLECTION_SLUG_RE.match(bare):
        raise ValueError(
            f"invalid collection name {bare!r}: must match {COLLECTION_SLUG_RE.pattern}"
        )
    return f"{SCRATCH_PREFIX}{bare}"


# --- Memory collection lifecycle ------------------------------------------

def _ensure_memory_collection() -> None:
    """Create the memory collection on first use, idempotently."""
    global _memory_ready
    if _memory_ready:
        return
    with _memory_lock:
        if _memory_ready:
            return
        dim = _embed_dimension()
        existing = {c.name for c in qdrant.get_collections().collections}
        if MEMORY_COLLECTION not in existing:
            try:
                qdrant.create_collection(
                    collection_name=MEMORY_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )
                LOG.info("Created memory collection %r (dim=%d)", MEMORY_COLLECTION, dim)
            except Exception as exc:
                # Tolerate "already exists" races.
                LOG.warning("create_collection raced or failed: %s", exc)
        else:
            info = qdrant.get_collection(MEMORY_COLLECTION)
            try:
                existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing_dim != dim:
                    LOG.error(
                        "Memory collection dim %d != embedder dim %d; recall results will be garbage",
                        existing_dim, dim,
                    )
            except Exception:
                pass
        _ensure_memory_indexes()
        _memory_ready = True


def _ensure_memory_indexes() -> None:
    """Create payload indexes used by recall/list_memories filters."""
    for field in ("subject",):
        try:
            qdrant.create_payload_index(
                collection_name=MEMORY_COLLECTION,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # idempotent


# --- Chunking --------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy paragraph-aware chunker with bounded overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            paragraphs.append(para)
        else:
            # Split oversized paragraph by hard windows.
            for i in range(0, len(para), max_chars - overlap):
                paragraphs.append(para[i:i + max_chars])

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= max_chars:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = f"{tail}\n\n{para}" if tail else para
            if len(buf) > max_chars:
                # Tail+para overflowed; emit tail-less chunk to keep bounded.
                chunks.append(buf[:max_chars])
                buf = buf[max_chars - overlap:] if overlap > 0 else ""
    if buf:
        chunks.append(buf)
    return chunks


# --- SSRF gate -------------------------------------------------------------

def _check_url_safe(url: str) -> None:
    """Reject URLs that resolve to non-global IPs (loopback, RFC1918, metadata)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme {parsed.scheme!r}; use http or https")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for {host!r}: {exc}") from exc

    ips = {info[4][0] for info in infos}
    if not ips:
        raise ValueError(f"no IPs resolved for {host!r}")

    for raw in ips:
        ip = ipaddress.ip_address(raw)
        # Normalise IPv4-mapped IPv6.
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if INGEST_ALLOW_PRIVATE:
            continue
        if not ip.is_global:
            raise ValueError(
                f"refusing to fetch {url!r}: host {host} resolved to non-global IP {ip}. "
                "Set SCRIBE_INGEST_ALLOW_PRIVATE=true to override (dev only)."
            )


def _stream_fetch(url: str) -> tuple[bytes, str]:
    """Fetch a URL with strict safety: no redirects, no env proxies, size cap.

    Returns (bytes, content_type). Raises ValueError on any policy violation.
    """
    _check_url_safe(url)
    with httpx.Client(
        follow_redirects=False,
        trust_env=False,
        timeout=INGEST_TIMEOUT,
        headers={"User-Agent": "Scribe/1.0 (+https://github.com/quintindk/chamberlain)"},
    ) as client:
        with client.stream("GET", url) as resp:
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location", "")
                raise ValueError(
                    f"redirects are not followed in v1; got {resp.status_code} → {loc!r}"
                )
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code} for {url!r}")
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and not any(content_type.startswith(ct) for ct in ALLOWED_CONTENT_TYPES):
                raise ValueError(
                    f"unsupported content-type {content_type!r}; "
                    f"allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
                )
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf.extend(chunk)
                if len(buf) > INGEST_MAX_BYTES:
                    raise ValueError(
                        f"response exceeded {INGEST_MAX_BYTES} bytes; aborting"
                    )
            return bytes(buf), content_type


# --- Secret / PII gate -----------------------------------------------------

# Loud, deliberately conservative. False positives are the point.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PEM private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abporsu]-[A-Za-z0-9-]{10,}\b")),
    ("Stripe live key", re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b")),
    ("Azure storage account key", re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}")),
    ("DB URL with credentials", re.compile(r"\b(?:postgres|postgresql|mysql|mongodb|redis|amqp)://[^\s/]+:[^\s/@]+@", re.IGNORECASE)),
    ("Bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b")),
    ("Generic password assignment", re.compile(r"\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9._\-/+=]{8,}", re.IGNORECASE)),
]

# Heuristic GDPR Article 9 keyword gate. Not a classifier; intentionally
# trips on obvious mentions so the model is forced to reconsider what it is
# trying to store as a "memory".
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Health / medical", re.compile(r"\b(?:HIV|AIDS|diagnos(?:is|ed)|prescription|medication|illness|disease|disorder|psychiatric|therapy session)\b", re.IGNORECASE)),
    ("Religion", re.compile(r"\b(?:Christian|Muslim|Jewish|Hindu|Buddhist|atheist|religion is|believes in (?:God|Allah|Jesus))\b", re.IGNORECASE)),
    ("Ethnicity / race", re.compile(r"\b(?:race is|ethnicity|ethnically|racial origin)\b", re.IGNORECASE)),
    ("Political views", re.compile(r"\b(?:votes? (?:for|Labour|Tory|ANC|DA|Republican|Democrat)|political (?:view|affiliation|party))\b", re.IGNORECASE)),
    ("Sexual orientation", re.compile(r"\b(?:sexual orientation|sexuality is|is (?:gay|lesbian|bisexual|straight|queer))\b", re.IGNORECASE)),
    ("Trade union", re.compile(r"\b(?:trade union member|union membership)\b", re.IGNORECASE)),
    ("Biometric", re.compile(r"\b(?:fingerprint|retina scan|facial recognition data|DNA sequence)\b", re.IGNORECASE)),
]


def _check_remember_safe(*parts: str) -> None:
    blob = "\n".join(p for p in parts if p)
    for label, pat in _SECRET_PATTERNS:
        if pat.search(blob):
            raise ValueError(
                f"refusing to store memory: looks like a {label}. "
                "Memories must not contain credentials, tokens, or secrets."
            )
    for label, pat in _PII_PATTERNS:
        if pat.search(blob):
            raise ValueError(
                f"refusing to store memory: looks like sensitive personal data "
                f"({label}). GDPR Article 9 categories are not permitted; "
                "rephrase or drop the fact."
            )


# --- Hit formatting --------------------------------------------------------

def _format_hit(hit: Any, collection: str) -> dict[str, Any]:
    payload = hit.payload or {}
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "score": round(float(hit.score), 4),
        "collection": collection,
        "repository": meta.get("repository") or payload.get("repository"),
        "file_path": meta.get("file_path") or payload.get("file_path") or payload.get("source"),
        "chunk_index": meta.get("chunk_index") or payload.get("chunk_index"),
        "content": payload.get("content") or payload.get("text") or payload.get("page_content", ""),
    }


# --- Tools: Search ---------------------------------------------------------

@mcp.tool
def search_archives(
    query: str,
    collection: str = "all",
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Search indexed archives and scratch pads for relevant chunks.

    Fan-out covers `archive-*` (Miller-managed) and `scratch-*` (ingested
    on demand) collections. Memories (`memory-default`) are NOT searched
    here; use `recall` to query them.

    Args:
        query: Natural-language question or keywords.
        collection: "all" (default), a full collection name, or a bare
            suffix (defaults to the archive prefix for back-compat).
        limit: Total chunks to return (capped at SCRIBE_MAX_LIMIT).
    """
    limit = max(1, min(limit, MAX_LIMIT))
    collections = _resolve_search_collections(collection)
    if not collections:
        LOG.warning("No matching collections for %r", collection)
        return []

    LOG.info("search_archives %r across %d collections (limit=%d)", query[:80], len(collections), limit)
    vector = _embed(query)

    hits: list[dict[str, Any]] = []
    for name in collections:
        try:
            results = qdrant.query_points(
                collection_name=name,
                query=vector,
                limit=limit,
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
    """List indexed `archive-*` and `scratch-*` collections with point counts and kind."""
    out: list[dict[str, Any]] = []
    for c in qdrant.get_collections().collections:
        if c.name.startswith(ARCHIVE_PREFIX):
            kind = "archive"
        elif c.name.startswith(SCRATCH_PREFIX):
            kind = "scratch"
        else:
            continue
        try:
            info = qdrant.get_collection(c.name)
            out.append({
                "name": c.name,
                "kind": kind,
                "points": info.points_count,
                "status": str(info.status),
            })
        except Exception as exc:
            LOG.warning("get_collection failed for %s: %s", c.name, exc)
    return sorted(out, key=lambda x: x["name"])


# --- Tools: Ingest ---------------------------------------------------------

def _stable_chunk_id(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(f"{source}|{idx}|{content}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(h[:32]))


def _ensure_scratch_collection(name: str) -> None:
    """Create a scratch collection on demand, idempotently."""
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        return
    dim = _embed_dimension()
    try:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        LOG.info("Created scratch collection %r (dim=%d)", name, dim)
    except Exception as exc:
        LOG.warning("create_collection raced or failed for %s: %s", name, exc)


def _upsert_chunks(
    collection: str,
    source: str,
    source_type: str,
    chunks: list[str],
) -> int:
    if not chunks:
        return 0
    _ensure_scratch_collection(collection)
    now = datetime.now(timezone.utc).isoformat()
    points: list[qmodels.PointStruct] = []
    for idx, content in enumerate(chunks):
        vec = _embed(content)
        points.append(qmodels.PointStruct(
            id=_stable_chunk_id(source, idx, content),
            vector=vec,
            payload={
                "content": content,
                "source": source,
                "source_type": source_type,
                "chunk_index": idx,
                "created_at": now,
            },
        ))
    qdrant.upsert(collection_name=collection, points=points)
    return len(points)


@mcp.tool
def ingest_url(url: str, collection: str, max_pages: int = 1) -> dict[str, Any]:
    """Fetch a single URL, extract main text, chunk, embed, upsert into a scratch collection.

    Multi-page crawling is not implemented in v1; `max_pages` must be 1.
    The fetch is gated by an SSRF allowlist (no private/loopback/metadata
    IPs) and a response-size cap. Redirects are not followed.

    Args:
        url: http(s) URL to ingest.
        collection: Scratch collection name (bare suffix or already-prefixed).
        max_pages: Reserved for future multi-page crawling. Must be 1.

    Returns:
        Dict with `collection`, `points`, `source`, `bytes`.
    """
    if max_pages != 1:
        raise ValueError("max_pages > 1 is not implemented in v1")
    name = _scratch_collection_name(collection)
    LOG.info("ingest_url url=%r collection=%r", url, name)

    raw, content_type = _stream_fetch(url)
    if content_type.startswith("text/plain"):
        text = raw.decode("utf-8", errors="replace")
    else:
        extracted = trafilatura.extract(raw.decode("utf-8", errors="replace"), url=url) or ""
        text = extracted

    if not text.strip():
        raise ValueError(f"no extractable text from {url!r}")

    chunks = _chunk_text(text)
    n = _upsert_chunks(name, source=url, source_type="url", chunks=chunks)
    return {"collection": name, "points": n, "source": url, "bytes": len(raw)}


@mcp.tool
def ingest_path(path: str, collection: str) -> dict[str, Any]:
    """Read text files under `/drop/<path>`, chunk, embed, upsert into a scratch collection.

    Only paths inside the container's `/drop` mount (host-bound, read-only)
    are accepted. Symlinks escaping `/drop` are rejected. File and total
    byte counts are capped.

    Args:
        path: Path relative to `/drop` (or absolute under `/drop`).
        collection: Scratch collection name (bare suffix or already-prefixed).

    Returns:
        Dict with `collection`, `points`, `files`, `bytes`.
    """
    name = _scratch_collection_name(collection)
    p = (INGEST_DROP_DIR / path).resolve() if not path.startswith("/") else Path(path).resolve()
    if not p.is_relative_to(INGEST_DROP_DIR):
        raise ValueError(f"path {p!s} is not inside the drop directory {INGEST_DROP_DIR!s}")
    if not p.exists():
        raise ValueError(f"path does not exist: {p!s}")

    files: list[Path] = []
    if p.is_file():
        files = [p]
    else:
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if not f.resolve().is_relative_to(INGEST_DROP_DIR):
                continue  # symlink escape
            files.append(f)
            if len(files) > INGEST_PATH_MAX_FILES:
                raise ValueError(f"too many files (> {INGEST_PATH_MAX_FILES}); narrow the path")

    LOG.info("ingest_path path=%r files=%d collection=%r", str(p), len(files), name)
    total_bytes = 0
    total_points = 0
    for f in files:
        try:
            data = f.read_bytes()
        except Exception as exc:
            LOG.warning("read failed for %s: %s", f, exc)
            continue
        total_bytes += len(data)
        if total_bytes > INGEST_PATH_MAX_BYTES:
            raise ValueError(f"total bytes exceeded {INGEST_PATH_MAX_BYTES}; aborting")
        text = data.decode("utf-8", errors="replace")
        chunks = _chunk_text(text)
        rel = str(f.relative_to(INGEST_DROP_DIR))
        total_points += _upsert_chunks(name, source=rel, source_type="path", chunks=chunks)

    return {"collection": name, "points": total_points, "files": len(files), "bytes": total_bytes}


@mcp.tool
def forget_collection(collection: str) -> dict[str, Any]:
    """Delete a `scratch-*` collection in full. Refuses to touch `archive-*` or `memory-*`."""
    name = _scratch_collection_name(collection)
    existing = {c.name for c in qdrant.get_collections().collections}
    if name not in existing:
        return {"ok": False, "collection": name, "reason": "not found"}
    qdrant.delete_collection(collection_name=name)
    LOG.info("[AUDIT] forget_collection collection=%r", name)
    return {"ok": True, "collection": name}


# --- Tools: Memory ---------------------------------------------------------

@mcp.tool
def remember(fact: str, subject: str, reason: str, citations: str) -> dict[str, Any]:
    """Store a single curated fact in `memory-default`. All fields are mandatory.

    Args:
        fact: The claim itself (<= 200 chars).
        subject: 1-2 word topical tag (e.g. "deployment", "architecture").
        reason: Why this fact is worth storing and which future tasks it serves.
        citations: Provenance: file references (`path/file.go:123`) or exact
            user quotations (`User input: "<exact quote>"`).

    Returns:
        Dict with `id` (UUID4 string) and the stored payload echo.
    """
    fact = (fact or "").strip()
    subject = (subject or "").strip()
    reason = (reason or "").strip()
    citations = (citations or "").strip()

    if not fact or not subject or not reason or not citations:
        raise ValueError("fact, subject, reason, and citations are all mandatory")
    if len(fact) > 200:
        raise ValueError(f"fact must be <= 200 chars (got {len(fact)})")

    _check_remember_safe(fact, reason, citations)
    _ensure_memory_collection()

    point_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "fact": fact,
        "subject": subject,
        "reason": reason,
        "citations": citations,
        "created_at": now,
    }
    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[qmodels.PointStruct(id=point_id, vector=_embed(fact), payload=payload)],
    )
    LOG.info("[AUDIT] remember id=%s subject=%r", point_id, subject)
    return {"id": point_id, **payload}


@mcp.tool
def recall(query: str, subject: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Semantic search over stored memories, optionally filtered by subject.

    Returns full memory payloads (fact, subject, reason, citations, score),
    not synthesised prose.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    _ensure_memory_collection()
    query_filter = None
    if subject:
        query_filter = qmodels.Filter(must=[
            qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject)),
        ])
    try:
        results = qdrant.query_points(
            collection_name=MEMORY_COLLECTION,
            query=_embed(query),
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points
    except Exception as exc:
        LOG.warning("recall failed: %s", exc)
        return []

    out = []
    for hit in results:
        payload = hit.payload or {}
        out.append({
            "id": str(hit.id),
            "score": round(float(hit.score), 4),
            "fact": payload.get("fact"),
            "subject": payload.get("subject"),
            "reason": payload.get("reason"),
            "citations": payload.get("citations"),
            "created_at": payload.get("created_at"),
        })
    return out


@mcp.tool
def forget(memory_id: str, reason: str) -> dict[str, Any]:
    """Delete a single memory point by id. The `reason` is logged to stderr for audit, not stored."""
    if not memory_id or not reason:
        raise ValueError("memory_id and reason are both mandatory")
    _ensure_memory_collection()
    try:
        qdrant.delete(
            collection_name=MEMORY_COLLECTION,
            points_selector=qmodels.PointIdsList(points=[memory_id]),
        )
    except Exception as exc:
        raise ValueError(f"delete failed: {exc}") from exc
    LOG.info("[AUDIT] forget id=%s reason=%r", memory_id, reason)
    return {"ok": True, "id": memory_id}


@mcp.tool
def list_memories(subject: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Enumerate stored memories, optionally filtered by subject. Returns up to `limit` entries."""
    limit = max(1, min(limit, 1000))
    _ensure_memory_collection()
    scroll_filter = None
    if subject:
        scroll_filter = qmodels.Filter(must=[
            qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject)),
        ])
    try:
        points, _ = qdrant.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        LOG.warning("list_memories failed: %s", exc)
        return []
    out = []
    for p in points:
        payload = p.payload or {}
        out.append({
            "id": str(p.id),
            "fact": payload.get("fact"),
            "subject": payload.get("subject"),
            "reason": payload.get("reason"),
            "citations": payload.get("citations"),
            "created_at": payload.get("created_at"),
        })
    return out


# --- Entrypoint ------------------------------------------------------------

if __name__ == "__main__":
    transport = os.getenv("SCRIBE_TRANSPORT", "http")
    host = os.getenv("SCRIBE_HOST", "0.0.0.0")
    port = int(os.getenv("SCRIBE_PORT", "8000"))
    path = os.getenv("SCRIBE_PATH", "/mcp" if transport == "http" else "/sse")
    LOG.info(
        "Starting Scribe (%s) on %s:%s%s | archive=%r scratch=%r memory=%r drop=%s",
        transport, host, port, path, ARCHIVE_PREFIX, SCRATCH_PREFIX, MEMORY_COLLECTION, INGEST_DROP_DIR,
    )
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=host, port=port, path=path)
