from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_uid TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    rating INTEGER,
    goodreads_id TEXT,
    storygraph_id TEXT,
    isbn TEXT,
    note_path TEXT,
    tags_json TEXT,
    themes_json TEXT,
    summary TEXT,
    review TEXT,
    body TEXT,
    raw_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_uid)
);
CREATE TABLE IF NOT EXISTS read_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_uid TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    goodreads_id TEXT,
    url TEXT,
    read_at TEXT,
    raw_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_uid)
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_uid TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    url TEXT,
    cover_url TEXT,
    media_type TEXT,
    published_at TEXT,
    description TEXT,
    raw_json TEXT,
    score REAL,
    score_breakdown TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    librarr_id TEXT,
    decided_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_uid)
);
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    kind TEXT,
    url TEXT,
    enabled INTEGER DEFAULT 1,
    weight REAL DEFAULT 0.0,
    config_json TEXT,
    last_seen_at TEXT,
    last_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    action TEXT NOT NULL,
    note TEXT,
    channel TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector_blob BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, model)
);
CREATE TABLE IF NOT EXISTS discord_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    channel_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'posted',
    reaction TEXT,
    reaction_counts_json TEXT,
    librarr_id TEXT,
    posted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    reacted_at TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id),
    UNIQUE(message_id)
);
"""


def connect(path: str) -> sqlite3.Connection:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_rating(rating: Any) -> int | None:
    """Convert empty/zero ratings to NULL. Returns int or None."""
    if rating is None or rating == '' or rating == 0:
        return None
    try:
        r = int(rating)
        return r if 1 <= r <= 5 else None
    except (TypeError, ValueError):
        return None


def upsert_book(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO books (
          source, source_uid, title, author, rating, goodreads_id, storygraph_id, isbn,
          note_path, tags_json, themes_json, summary, review, body, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, source_uid) DO UPDATE SET
          title=excluded.title,
          author=excluded.author,
          rating=excluded.rating,
          goodreads_id=excluded.goodreads_id,
          storygraph_id=excluded.storygraph_id,
          isbn=excluded.isbn,
          note_path=excluded.note_path,
          tags_json=excluded.tags_json,
          themes_json=excluded.themes_json,
          summary=excluded.summary,
          review=excluded.review,
          body=excluded.body,
          raw_json=excluded.raw_json,
          updated_at=CURRENT_TIMESTAMP
        RETURNING id
        """,
        (
            row.get('source', 'obsidian'),
            row['source_uid'],
            row['title'],
            row.get('author', ''),
            _normalize_rating(row.get('rating')),
            row.get('goodreads_id'),
            row.get('storygraph_id'),
            row.get('isbn'),
            row.get('note_path'),
            _json(row.get('tags', [])),
            _json(row.get('themes', [])),
            row.get('summary', ''),
            row.get('review', ''),
            row.get('body', ''),
            _json(row.get('raw', {})),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    return int(row['id'])


def upsert_read_event(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO read_events (source, source_uid, title, author, goodreads_id, url, read_at, raw_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, source_uid) DO UPDATE SET
          title=excluded.title,
          author=excluded.author,
          goodreads_id=excluded.goodreads_id,
          url=excluded.url,
          read_at=excluded.read_at,
          raw_json=excluded.raw_json,
          updated_at=CURRENT_TIMESTAMP
        RETURNING id
        """,
        (
            row.get('source', 'goodreads-rss'),
            row['source_uid'],
            row['title'],
            row.get('author', ''),
            row.get('goodreads_id'),
            row.get('url'),
            row.get('read_at'),
            _json(row.get('raw', {})),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    return int(row['id'])


def upsert_candidate(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO candidates (
          source, source_uid, title, author, url, cover_url, media_type, published_at,
          description, raw_json, score, score_breakdown, status, librarr_id, decided_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, 'new'), ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, source_uid) DO UPDATE SET
          title=excluded.title,
          author=excluded.author,
          url=excluded.url,
          cover_url=excluded.cover_url,
          media_type=excluded.media_type,
          published_at=excluded.published_at,
          description=excluded.description,
          raw_json=excluded.raw_json,
          score=COALESCE(excluded.score, candidates.score),
          score_breakdown=COALESCE(excluded.score_breakdown, candidates.score_breakdown),
          status=CASE WHEN candidates.status IN ('approved', 'rejected', 'imported') THEN candidates.status ELSE COALESCE(excluded.status, candidates.status) END,
          librarr_id=COALESCE(excluded.librarr_id, candidates.librarr_id),
          decided_at=COALESCE(excluded.decided_at, candidates.decided_at),
          updated_at=CURRENT_TIMESTAMP
        RETURNING id
        """,
        (
            row['source'],
            row['source_uid'],
            row['title'],
            row.get('author', ''),
            row.get('url'),
            row.get('cover_url'),
            row.get('media_type', 'audiobook'),
            row.get('published_at'),
            row.get('description', ''),
            _json(row.get('raw', {})),
            row.get('score'),
            _json(row.get('score_breakdown', {})),
            row.get('status'),
            row.get('librarr_id'),
            row.get('decided_at'),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    return int(row['id'])


def record_feedback(conn: sqlite3.Connection, candidate_id: int | None, action: str, note: str = '', channel: str = '') -> None:
    conn.execute(
        'INSERT INTO feedback (candidate_id, action, note, channel) VALUES (?, ?, ?, ?)',
        (candidate_id, action, note, channel),
    )
    conn.commit()


def record_event(conn: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
    conn.execute('INSERT INTO events (kind, payload_json) VALUES (?, ?)', (kind, _json(payload)))
    conn.commit()


def set_candidate_status(conn: sqlite3.Connection, candidate_id: int, status: str, librarr_id: str | None = None, note: str | None = None) -> None:
    conn.execute(
        'UPDATE candidates SET status=?, librarr_id=COALESCE(?, librarr_id), decided_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE id=?',
        (status, librarr_id, candidate_id),
    )
    if note:
        record_feedback(conn, candidate_id, status, note, 'system')
    conn.commit()


def get_books(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute('SELECT * FROM books ORDER BY COALESCE(rating, 0) DESC, updated_at DESC'))


def get_candidates(conn: sqlite3.Connection, status: str | None = None, limit: int | None = None) -> list[sqlite3.Row]:
    q = 'SELECT * FROM candidates'
    args: list[Any] = []
    if status:
        q += ' WHERE status = ?'
        args.append(status)
    q += ' ORDER BY COALESCE(score, 0) DESC, updated_at DESC'
    if limit:
        q += f' LIMIT {int(limit)}'
    return list(conn.execute(q, args))


def get_source(conn: sqlite3.Connection, name: str):
    return conn.execute('SELECT * FROM sources WHERE name = ?', (name,)).fetchone()


def get_embeddings_map(conn: sqlite3.Connection, entity_type: str, model: str, entity_ids: Iterable[int] | None = None) -> dict[int, dict[str, Any]]:
    q = 'SELECT entity_id, text_hash, dim, vector_blob FROM embeddings WHERE entity_type=? AND model=?'
    args: list[Any] = [entity_type, model]
    if entity_ids is not None:
        ids = list(dict.fromkeys(int(v) for v in entity_ids))
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        q += f' AND entity_id IN ({placeholders})'
        args.extend(ids)
    out: dict[int, dict[str, Any]] = {}
    for row in conn.execute(q, args):
        out[int(row['entity_id'])] = {'text_hash': row['text_hash'], 'dim': row['dim'], 'vector_blob': row['vector_blob']}
    return out


def get_embedding(conn: sqlite3.Connection, entity_type: str, entity_id: int, model: str):
    return conn.execute('SELECT * FROM embeddings WHERE entity_type=? AND entity_id=? AND model=?', (entity_type, entity_id, model)).fetchone()


def upsert_embedding(conn: sqlite3.Connection, *, entity_type: str, entity_id: int, model: str, text_hash: str, dim: int, vector_blob: bytes) -> None:
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
        (entity_type, entity_id, model, text_hash, dim, vector_blob),
    )
    conn.commit()


def upsert_source(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO sources (name, kind, url, enabled, weight, config_json, last_seen_at, last_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name) DO UPDATE SET
          kind=excluded.kind,
          url=excluded.url,
          enabled=excluded.enabled,
          weight=excluded.weight,
          config_json=excluded.config_json,
          last_seen_at=COALESCE(excluded.last_seen_at, sources.last_seen_at),
          last_hash=COALESCE(excluded.last_hash, sources.last_hash),
          updated_at=CURRENT_TIMESTAMP
        """,
        (
            row['name'],
            row.get('kind', 'rss'),
            row.get('url'),
            1 if row.get('enabled', True) else 0,
            row.get('weight', 0.0),
            _json(row.get('config', {})),
            row.get('last_seen_at'),
            row.get('last_hash'),
        ),
    )
    conn.commit()


def upsert_discord_message(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    channel_id: str,
    message_id: str,
    content: str,
    status: str = 'posted',
    reaction: str | None = None,
    reaction_counts_json: str | None = None,
    librarr_id: str | None = None,
    reacted_at: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO discord_messages (
            candidate_id, channel_id, message_id, content, status, reaction,
            reaction_counts_json, librarr_id, reacted_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(candidate_id) DO UPDATE SET
          channel_id=excluded.channel_id,
          message_id=excluded.message_id,
          content=excluded.content,
          status=excluded.status,
          reaction=COALESCE(excluded.reaction, discord_messages.reaction),
          reaction_counts_json=COALESCE(excluded.reaction_counts_json, discord_messages.reaction_counts_json),
          librarr_id=COALESCE(excluded.librarr_id, discord_messages.librarr_id),
          reacted_at=COALESCE(excluded.reacted_at, discord_messages.reacted_at),
          updated_at=CURRENT_TIMESTAMP
        """,
        (candidate_id, channel_id, message_id, content, status, reaction, reaction_counts_json, librarr_id, reacted_at),
    )
    conn.commit()


def get_discord_message_for_candidate(conn: sqlite3.Connection, candidate_id: int):
    return conn.execute('SELECT * FROM discord_messages WHERE candidate_id=?', (candidate_id,)).fetchone()


def get_discord_message_by_message_id(conn: sqlite3.Connection, message_id: str):
    return conn.execute('SELECT * FROM discord_messages WHERE message_id=?', (message_id,)).fetchone()


def get_pending_discord_messages(conn: sqlite3.Connection):
    return list(
        conn.execute(
            """
            SELECT dm.*, c.title, c.author, c.url, c.cover_url, c.media_type, c.published_at, c.description, c.raw_json, c.score, c.score_breakdown, c.status AS candidate_status
            FROM discord_messages dm
            JOIN candidates c ON c.id = dm.candidate_id
            WHERE dm.status = 'posted' AND c.status = 'discord_pending'
            ORDER BY dm.posted_at ASC
            """
        )
    )


def update_discord_message_status(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    status: str,
    reaction: str | None = None,
    reaction_counts_json: str | None = None,
    librarr_id: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE discord_messages
        SET status=?, reaction=COALESCE(?, reaction), reaction_counts_json=COALESCE(?, reaction_counts_json),
            librarr_id=COALESCE(?, librarr_id), reacted_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
        WHERE candidate_id=?
        """,
        (status, reaction, reaction_counts_json, librarr_id, candidate_id),
    )
    conn.commit()


def normalize_book_key(title: str, author: str) -> str:
    t = re.sub(r'\s+', ' ', (title or '').strip().lower())
    a = re.sub(r'\s+', ' ', (author or '').strip().lower())
    return f'{t}||{a}'


def normalize_candidate_match_key(title: str, author: str) -> str:
    """Looser match key for source candidates vs existing library books.

    This intentionally strips edition/series metadata so variants like
    "The Martian (Deluxe Edition)" or a source title lacking a series suffix
    can still match the canonical book row already in the library.
    It should NOT be used for book-table dedupe because it can collapse distinct
    series volumes (e.g. 1Q84 #2 vs #3).
    """
    t = re.sub(r'\s+', ' ', (title or '').strip().lower())
    m = re.search(r'\s*\(([^)]*)\)\s*$', t)
    if m:
        inside = m.group(1)
        if ('#' in inside) or any(word in inside for word in ('edition', 'deluxe', 'special', 'anniversary', 'unabridged', 'illustrated', 'expanded', 'collector', 'audiobook')):
            t = t[: m.start()].strip()
    t = re.sub(r'\s*:\s*(?:deluxe|special|anniversary|unabridged|illustrated|expanded|collector)\s+edition\s*$', '', t)
    t = re.sub(r'\s+(?:deluxe|special|anniversary|unabridged|illustrated|expanded|collector)\s+edition\s*$', '', t)
    a = re.sub(r'\s+', ' ', (author or '').strip().lower())
    return f'{re.sub(r"\s+", " ", t).strip()}||{a}'


def candidate_match_keys(title: str, author: str, source: str = '') -> set[str]:
    """Return candidate match key variants for library dedupe/filtering.

    Apple audiobooks often encode a canonical title plus a marketing / series
    suffix after a colon, so we include a source-specific prefix variant there.
    This keeps the broader normalization conservative for other sources.
    """
    keys = {normalize_candidate_match_key(title, author)}
    source_name = (source or '').strip().lower()
    if source_name == 'apple-top-audiobooks':
        title_norm = re.sub(r'\s+', ' ', (title or '').strip())
        if ':' in title_norm:
            prefix = title_norm.split(':', 1)[0].strip()
            if prefix:
                keys.add(normalize_candidate_match_key(prefix, author))
        stripped = re.sub(r'\s*,\s*(?:book|vol\.?|volume|part|chapter)\s*\d+[\w\.-]*.*$', '', title_norm, flags=re.I).strip()
        if stripped and stripped != title_norm:
            keys.add(normalize_candidate_match_key(stripped, author))
    return keys


def get_book_title_author_keys(conn: sqlite3.Connection) -> set[str]:
    """Return a set of normalized title||author keys for all books."""
    rows = conn.execute('SELECT title, author FROM books').fetchall()
    return {normalize_book_key(r['title'], r['author']) for r in rows}


def _row_text(row, field: str) -> str:
    value = row[field]
    return '' if value is None else str(value)


def _book_richness(row) -> tuple:
    return (
        1 if row['goodreads_id'] else 0,
        1 if row['storygraph_id'] else 0,
        1 if row['isbn'] else 0,
        1 if row['rating'] is not None else 0,
        len(_row_text(row, 'summary')),
        len(_row_text(row, 'review')),
        len(_row_text(row, 'body')),
        len(_row_text(row, 'tags_json')),
        len(_row_text(row, 'themes_json')),
        len(_row_text(row, 'raw_json')),
    )


def dedupe_books_by_title_author(conn: sqlite3.Connection) -> dict[str, int]:
    rows = list(conn.execute('SELECT * FROM books ORDER BY id ASC'))
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        key = normalize_book_key(row['title'], row['author'])
        groups.setdefault(key, []).append(row)

    merged_groups = 0
    deleted_rows = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        keep = max(group, key=_book_richness)
        keep_id = int(keep['id'])
        tags = []
        themes = []
        raw: dict[str, Any] = {}
        merged = dict(keep)
        for row in group:
            row_tags = json.loads(row['tags_json'] or '[]')
            row_themes = json.loads(row['themes_json'] or '[]')
            for item in row_tags:
                if item not in tags:
                    tags.append(item)
            for item in row_themes:
                if item not in themes:
                    themes.append(item)
            try:
                row_raw = json.loads(row['raw_json'] or '{}')
            except Exception:
                row_raw = {}
            if isinstance(row_raw, dict):
                for k, v in row_raw.items():
                    raw.setdefault(k, v)
            for field in ('goodreads_id', 'storygraph_id', 'isbn', 'note_path'):
                if not merged.get(field) and row[field]:
                    merged[field] = row[field]
            for field in ('summary', 'review', 'body'):
                cur = _row_text(merged, field)
                new = _row_text(row, field)
                if len(new) > len(cur):
                    merged[field] = new
            if not merged.get('rating') and row['rating'] is not None:
                merged['rating'] = row['rating']
        conn.execute(
            """
            UPDATE books
            SET goodreads_id=?, storygraph_id=?, isbn=?, note_path=?, tags_json=?, themes_json=?,
                summary=?, review=?, body=?, raw_json=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                merged.get('goodreads_id'),
                merged.get('storygraph_id'),
                merged.get('isbn'),
                merged.get('note_path'),
                _json(tags),
                _json(themes),
                merged.get('summary') or '',
                merged.get('review') or '',
                merged.get('body') or '',
                _json(raw),
                keep_id,
            ),
        )
        for row in group:
            if int(row['id']) == keep_id:
                continue
            conn.execute('DELETE FROM books WHERE id=?', (int(row['id']),))
            deleted_rows += 1
        merged_groups += 1
    conn.commit()
    return {'groups': merged_groups, 'deleted_rows': deleted_rows}
