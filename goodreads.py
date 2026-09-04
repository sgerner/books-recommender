from __future__ import annotations

import re
from typing import Any

from .db import record_event, upsert_read_event
from .feeds import normalized_feed_items


def _read_event_uid(item: dict[str, Any]) -> str:
    """Build a stable per-book/per-review identity for Goodreads RSS rows."""
    goodreads_id = str(item.get('goodreads_id') or '').strip()
    review_ref = str(item.get('guid') or item.get('link') or '').strip()
    if goodreads_id:
        review_match = re.search(r'/review/show/(\d+)', review_ref, re.I)
        if review_match:
            return f'goodreads:{goodreads_id}:review:{review_match.group(1)}'
        return f'goodreads:{goodreads_id}'
    return review_ref or f"title:{item.get('title') or ''}"


def sync_goodreads_rss(conn, rss_url: str) -> dict[str, int]:
    items = normalized_feed_items(rss_url)
    inserted = 0
    updated = 0
    seen = 0
    for item in items:
        seen += 1
        uid = _read_event_uid(item)
        goodreads_id = item.get('goodreads_id') or ''
        row = {
            'source': 'goodreads-rss',
            'source_uid': uid,
            'title': item.get('title', ''),
            'author': item.get('author', ''),
            'goodreads_id': goodreads_id,
            'url': item.get('url') or item.get('link') or '',
            'read_at': item.get('published_at'),
            'raw': item,
        }
        existing = conn.execute('SELECT title, author, goodreads_id, url, read_at, raw_json FROM read_events WHERE source=? AND source_uid=?', ('goodreads-rss', uid)).fetchone()
        if existing is None and goodreads_id:
            # Migrate the old review-URL identity in place when possible so a
            # normal sync does not duplicate the historical read event.
            review_match = re.search(r'/review/show/(\d+)', str(item.get('guid') or item.get('link') or ''), re.I)
            legacy_query = (
                'SELECT id FROM read_events WHERE source=? AND goodreads_id=? AND source_uid<>? '
                + ('AND source_uid LIKE ? ' if review_match else '')
                + 'ORDER BY id LIMIT 1'
            )
            legacy_args: list[Any] = ['goodreads-rss', goodreads_id, uid]
            if review_match:
                legacy_args.append(f'%/review/show/{review_match.group(1)}%')
            legacy = conn.execute(legacy_query, legacy_args).fetchone()
            if legacy:
                conn.execute('UPDATE read_events SET source_uid=? WHERE id=?', (uid, legacy['id']))
                conn.commit()
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
