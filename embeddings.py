from __future__ import annotations

import array
import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

DEFAULT_EMBEDDING_MODEL = 'qwen3-embedding:4b'
DEFAULT_EMBEDDING_BASE_URL = 'http://ollama:11434'


@dataclass
class EmbeddingItem:
    entity_type: str
    entity_id: int
    text: str
    text_hash: str


@dataclass
class EmbeddingSyncStats:
    processed: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            'processed': self.processed,
            'inserted': self.inserted,
            'updated': self.updated,
            'skipped': self.skipped,
            'failed': self.failed,
        }


def text_hash(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    data = f'{model}\0{text}'.encode('utf-8', errors='replace')
    return hashlib.sha256(data).hexdigest()


def normalize_vector(vec: Iterable[float]) -> list[float]:
    out = [float(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def vector_to_blob(vec: Iterable[float]) -> bytes:
    arr = array.array('f', (float(v) for v in vec))
    return arr.tobytes()


def blob_to_vector(blob: bytes | memoryview | None) -> list[float]:
    if blob is None:
        return []
    raw = blob.tobytes() if isinstance(blob, memoryview) else blob
    arr = array.array('f')
    arr.frombytes(raw)
    return list(arr)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if not av or not bv:
        return 0.0
    if len(av) != len(bv):
        m = min(len(av), len(bv))
        av = av[:m]
        bv = bv[:m]
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(y * y for y in bv)) or 1.0
    return dot / (na * nb)


def mean_vector(vectors: Iterable[Iterable[float]], weights: Iterable[float] | None = None) -> list[float]:
    vecs = [list(v) for v in vectors if v]
    if not vecs:
        return []
    if weights is None:
        weights_list = [1.0] * len(vecs)
    else:
        weights_list = list(weights)
        if len(weights_list) != len(vecs):
            raise ValueError('weights length must match vectors length')
    dim = max(len(v) for v in vecs)
    acc = [0.0] * dim
    total = 0.0
    for vec, weight in zip(vecs, weights_list):
        total += weight
        for i, value in enumerate(vec):
            acc[i] += value * weight
    if total == 0:
        return []
    out = [v / total for v in acc]
    return normalize_vector(out)


def candidate_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get('title', ''),
        candidate.get('author', ''),
        candidate.get('description', ''),
        ' '.join(candidate.get('tags', []) if isinstance(candidate.get('tags'), list) else []),
        ' '.join(candidate.get('themes', []) if isinstance(candidate.get('themes'), list) else []),
    ]
    return ' '.join(p for p in parts if p)


class OllamaEmbedder:
    def __init__(self, base_url: str = DEFAULT_EMBEDDING_BASE_URL, model: str = DEFAULT_EMBEDDING_MODEL, timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({'model': self.model, 'input': texts}).encode('utf-8')
        req = urllib.request.Request(
            f'{self.base_url}/api/embed',
            data=payload,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8', errors='replace'))
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace') if hasattr(e, 'read') else ''
            raise RuntimeError(f'Ollama embedding request failed: {e.code} {body}') from e
        except Exception as e:
            raise RuntimeError(f'Ollama embedding request failed: {e}') from e

        vectors = data.get('embeddings') or []
        if not isinstance(vectors, list):
            raise RuntimeError(f'Unexpected Ollama embedding response: {data!r}')
        if len(vectors) != len(texts):
            if 'embedding' in data and len(texts) == 1:
                vectors = [data['embedding']]
            else:
                raise RuntimeError(f'Ollama returned {len(vectors)} embeddings for {len(texts)} inputs')
        return [normalize_vector(v) for v in vectors]


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def upsert_embedding(conn, *, entity_type: str, entity_id: int, model: str, text_hash_value: str, vector: Iterable[float]) -> None:
    vec = [float(v) for v in vector]
    blob = vector_to_blob(vec)
    dim = len(vec)
    conn.execute(
        """
        INSERT INTO embeddings (entity_type, entity_id, model, text_hash, dim, vector_blob, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET
          text_hash=excluded.text_hash,
          dim=excluded.dim,
          vector_blob=excluded.vector_blob,
          updated_at=CURRENT_TIMESTAMP
        """,
        (entity_type, entity_id, model, text_hash_value, dim, blob),
    )


def load_embeddings_map(conn, *, entity_type: str, model: str, entity_ids: Iterable[int] | None = None) -> dict[int, list[float]]:
    q = 'SELECT entity_id, vector_blob, text_hash FROM embeddings WHERE entity_type=? AND model=?'
    args: list[Any] = [entity_type, model]
    if entity_ids is not None:
        ids = list(dict.fromkeys(int(v) for v in entity_ids))
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        q += f' AND entity_id IN ({placeholders})'
        args.extend(ids)
    out: dict[int, list[float]] = {}
    for row in conn.execute(q, args):
        out[int(row['entity_id'])] = blob_to_vector(row['vector_blob'])
    return out


def load_embedding_hashes(conn, *, entity_type: str, model: str, entity_ids: Iterable[int] | None = None) -> dict[int, str]:
    q = 'SELECT entity_id, text_hash FROM embeddings WHERE entity_type=? AND model=?'
    args: list[Any] = [entity_type, model]
    if entity_ids is not None:
        ids = list(dict.fromkeys(int(v) for v in entity_ids))
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        q += f' AND entity_id IN ({placeholders})'
        args.extend(ids)
    return {int(row['entity_id']): row['text_hash'] for row in conn.execute(q, args)}


def sync_embeddings(
    conn,
    *,
    entity_type: str,
    items: Iterable[EmbeddingItem],
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    batch_size: int = 16,
) -> dict[str, int]:
    embedder = OllamaEmbedder(base_url=base_url, model=model)
    existing_hashes = load_embedding_hashes(conn, entity_type=entity_type, model=model)
    pending: list[EmbeddingItem] = []
    stats = EmbeddingSyncStats()

    for item in items:
        stats.processed += 1
        if not item.text.strip():
            stats.skipped += 1
            continue
        if existing_hashes.get(int(item.entity_id)) == item.text_hash:
            stats.skipped += 1
            continue
        pending.append(item)

    for chunk in chunked(pending, batch_size):
        texts = [item.text for item in chunk]
        try:
            vectors = embedder.embed_batch(texts)
        except Exception:
            stats.failed += len(chunk)
            continue
        for item, vec in zip(chunk, vectors):
            upsert_embedding(
                conn,
                entity_type=entity_type,
                entity_id=int(item.entity_id),
                model=model,
                text_hash_value=item.text_hash,
                vector=vec,
            )
            if item.entity_id in existing_hashes:
                stats.updated += 1
            else:
                stats.inserted += 1
            existing_hashes[int(item.entity_id)] = item.text_hash
    conn.commit()
    return stats.as_dict()
