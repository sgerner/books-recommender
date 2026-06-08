from __future__ import annotations

from typing import Any

from .db import record_event, upsert_read_event
from .feeds import normalized_feed_items


def sync_goodreads_rss(conn, rss_url: str) -> dict[str, int]:
    items = normalized_feed_items(rss_url)
    inserted = 0
    updated = 0
    seen = 0
    for item in items:
        seen += 1
        uid = item.get('guid') or item.get('link') or f"{item.get('goodreads_id') or ''}:{item.get('title') or ''}"
        row = {
            'source': 'goodreads-rss',
            'source_uid': uid,
            'title': item.get('title', ''),
            'author': item.get('author', ''),
            'goodreads_id': item.get('goodreads_id') or '',
            'url': item.get('link') or item.get('url') or '',
            'read_at': item.get('published_at'),
            'raw': item,
        }
        existing = conn.execute('SELECT title, author, goodreads_id, url, read_at, raw_json FROM read_events WHERE source=? AND source_uid=?', ('goodreads-rss', uid)).fetchone()
        fingerprint = (
            str(row['title']), str(row['author']), str(row['goodreads_id']), str(row['url']), str(row['read_at']),
            __import__('json').dumps(row['raw'], sort_keys=True, ensure_ascii=False),
        )
        if existing:
            existing_fp = (
                str(existing['title']), str(existing['author']), str(existing['goodreads_id'] or ''), str(existing['url'] or ''), str(existing['read_at'] or ''), str(existing['raw_json'] or '{}'),
            )
            if existing_fp == fingerprint:
                continue
            updated += 1
        else:
            inserted += 1
        upsert_read_event(conn, row)
    record_event(conn, 'goodreads_sync', {'seen': seen, 'inserted': inserted, 'updated': updated, 'rss_url': rss_url})
    return {'seen': seen, 'inserted': inserted, 'updated': updated}
