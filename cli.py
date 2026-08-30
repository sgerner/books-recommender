from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_librarr_api_key
from .db import connect, get_books, get_candidates, get_embeddings_map, get_book_title_author_keys, init_db, record_event, record_feedback, set_candidate_status, upsert_candidate, upsert_source, dedupe_books_by_title_author, get_source, normalize_book_key, normalize_candidate_match_key, candidate_match_keys
from .discord_workflow import cmd_discord_post, cmd_discord_sync_reactions
from .embeddings import EmbeddingItem, candidate_text, sync_embeddings, text_hash as embedding_text_hash, blob_to_vector
from .enrichment import enrich_book_metadata
from .goodreads import sync_goodreads_rss
from .librarr import LibrarrClient
from .obsidian import import_obsidian, book_text
from .scoring import build_profile, candidate_is_banned, score_candidate
from .source_discovery import discover_source_items
from .sources import classify_source, combined_sources, default_weight, archive_sources_in_inbox, source_name_from_url


def _candidate_dict(row):
    raw = json.loads(row['raw_json']) if row['raw_json'] else {}
    return {
        'id': row['id'],
        'source': row['source'],
        'source_uid': row['source_uid'],
        'title': row['title'],
        'author': row['author'],
        'url': row['url'],
        'cover_url': row['cover_url'],
        'media_type': row['media_type'] or 'audiobook',
        'published_at': row['published_at'],
        'description': row['description'] or '',
        'raw': raw,
        'score': row['score'],
        'score_breakdown': json.loads(row['score_breakdown']) if row['score_breakdown'] else {},
        'status': row['status'],
    }


def _candidate_dict(row):
    raw = json.loads(row['raw_json']) if row['raw_json'] else {}
    return {
        'id': row['id'],
        'source': row['source'],
        'source_uid': row['source_uid'],
        'title': row['title'],
        'author': row['author'],
        'url': row['url'],
        'cover_url': row['cover_url'],
        'media_type': row['media_type'] or 'audiobook',
        'published_at': row['published_at'],
        'description': row['description'] or '',
        'raw': raw,
        'score': row['score'],
        'score_breakdown': json.loads(row['score_breakdown']) if row['score_breakdown'] else {},
        'status': row['status'],
    }


def _goodreads_id_for_candidate(row: dict[str, Any]) -> str:
    raw = row.get('raw') or {}
    if isinstance(raw, dict) and raw.get('goodreads_id'):
        match = re.search(r'\d+', str(raw['goodreads_id']))
        if match:
            return match.group(0)
    for value in (row.get('url'), row.get('source_uid')):
        match = re.search(r'goodreads(?:\.com)?[/:]?(?:book/show/)?(\d+)', str(value or ''), re.I)
        if match:
            return match.group(1)
        match = re.search(r'goodreads\.com/book/show/(\d+)', str(value or ''), re.I)
        if match:
            return match.group(1)
    return ''


def _strip_resolved_author_from_title(title: str, author: str) -> str:
    title = re.sub(r'\s+', ' ', str(title or '')).strip()
    author = re.sub(r'\s+', ' ', str(author or '')).strip()
    if not title or not author:
        return title
    return re.sub(r'\s+by\s+' + re.escape(author) + r'\s*$', '', title, flags=re.I).strip(' \t-–—:') or title


def cmd_init(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    if args.import_obsidian:
        stats = import_obsidian(conn, cfg['vault_path'])
        dedupe = dedupe_books_by_title_author(conn)
        print(json.dumps({'obsidian': stats, 'dedupe': dedupe}, indent=2))


def cmd_sync_goodreads_to_obsidian(args):
    """Sync Goodreads RSS read books to Obsidian notes and database with embeddings."""
    from datetime import datetime
    from .feeds import normalized_feed_items
    from .text import html_to_text, slugify
    from .embeddings import OllamaEmbedder, text_hash, upsert_embedding, vector_to_blob
    from .db import normalize_book_key, get_book_title_author_keys, upsert_book
    from .goodreads_scraper import fetch_book_metadata

    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)

    vault_path = Path(cfg['vault_path'])
    vault_path.mkdir(parents=True, exist_ok=True)

    # Fetch Goodreads RSS
    rss_url = args.goodreads_url or cfg['goodreads_rss_url']
    items = normalized_feed_items(rss_url)

    emb_cfg = cfg.get('embeddings', {})
    embedder = OllamaEmbedder(
        base_url=emb_cfg.get('base_url', 'http://ollama:11434'),
        model=emb_cfg.get('model', 'qwen3-embedding:4b'),
    )

    # Build lookup tables for existing books
    existing_by_goodreads_id = {}
    existing_by_isbn = {}
    existing_by_title_author = {}
    for row in conn.execute('SELECT id, title, author, goodreads_id, isbn FROM books').fetchall():
        if row['goodreads_id']:
            existing_by_goodreads_id[str(row['goodreads_id'])] = row['id']
        if row['isbn']:
            existing_by_isbn[str(row['isbn'])] = row['id']
        key = normalize_book_key(row['title'], row['author'])
        existing_by_title_author[key] = row['id']

    stats = {'seen': len(items), 'new': 0, 'skipped': 0, 'notes_created': 0, 'embeddings_created': 0, 'scraped': 0, 'scrape_errors': 0, 'errors': 0}

    for item in items:
        raw = item.get('raw', {})
        inner = raw if isinstance(raw, dict) else {}

        # Extract clean data
        title = inner.get('title') or item.get('title', '')
        author = inner.get('author_name') or item.get('author', '')
        # Clean up author field if it has "name: Steven" suffix
        if ' name: ' in author:
            author = author.split(' name: ')[0].strip()

        goodreads_id = str(inner.get('book_id') or item.get('goodreads_id', ''))
        if not goodreads_id:
            stats['skipped'] += 1
            continue

        # Check if already in books table by goodreads_id
        if goodreads_id in existing_by_goodreads_id:
            stats['skipped'] += 1
            continue

        # Check by ISBN
        isbn = inner.get('isbn', '') or ''
        if isbn and isbn in existing_by_isbn:
            stats['skipped'] += 1
            continue

        # Check by title+author
        key = normalize_book_key(title, author)
        if key in existing_by_title_author:
            stats['skipped'] += 1
            continue

        # Extract additional data from RSS
        rating = inner.get('user_rating')
        if rating:
            try:
                rating = int(rating)
                if rating < 1 or rating > 5:
                    rating = None
            except (ValueError, TypeError):
                rating = None

        book_description = inner.get('book_description', '') or ''
        summary = html_to_text(book_description)
        user_review = inner.get('user_review') or 'No review provided.'
        review = html_to_text(user_review) if user_review and user_review != 'None' else 'No review provided.'

        # Parse read date
        read_at = item.get('published_at') or ''
        date_read = ''
        if read_at:
            try:
                # Try to parse ISO format
                dt = datetime.fromisoformat(read_at.replace('Z', '+00:00'))
                date_read = dt.strftime('%Y-%m-%d')
            except:
                date_read = ''

        # Scrape Goodreads page for enriched metadata
        pages = ''
        series = ''
        themes = []
        setting = ''
        scraped_isbn13 = ''
        
        try:
            metadata = fetch_book_metadata(goodreads_id, delay=1.5)
            if 'error' not in metadata:
                stats['scraped'] += 1
                if metadata.get('pages'):
                    pages = str(metadata['pages'])
                if metadata.get('series'):
                    series = metadata['series']
                if metadata.get('genres'):
                    themes = metadata['genres']
                if metadata.get('isbn13') and not isbn:
                    isbn = metadata['isbn13']
                    scraped_isbn13 = isbn
            else:
                stats['scrape_errors'] += 1
        except Exception as e:
            stats['scrape_errors'] += 1

        # Create Obsidian note
        safe_title = title.replace('/', '-').replace('\\', '-').replace(':', '-')
        note_filename = f'{safe_title}.md'
        note_path = vault_path / note_filename

        # Avoid overwriting existing notes
        if note_path.exists():
            # Try with goodreads_id suffix
            note_filename = f'{safe_title} ({goodreads_id}).md'
            note_path = vault_path / note_filename

        # Format themes as YAML list
        themes_yaml = json.dumps(themes) if themes else '[]'
        
        # Build note content
        frontmatter = f'''---
title: "{title}"
goodreads_id: {goodreads_id}
author: "{author}"
rating: {rating if rating else ''}
date_read: {date_read}
date_added: {date_read or datetime.now().strftime('%Y-%m-%d')}
pages: "{pages}"
isbn: "{isbn}"
themes: {themes_yaml}
setting: "{setting}"
series: "{series}"
tags: ["read"]
---

# {title}
By **{author}**

## Summary
{summary}

## Review
{review}
'''
        note_path.write_text(frontmatter, encoding='utf-8')
        stats['notes_created'] += 1

        # Add to books table
        source_uid = f'goodreads_id:{goodreads_id}'
        row = {
            'source': 'obsidian',
            'source_uid': source_uid,
            'title': title,
            'author': author,
            'rating': rating,
            'goodreads_id': goodreads_id,
            'storygraph_id': '',
            'isbn': isbn,
            'note_path': str(note_path),
            'tags': ['read'] + themes[:5],  # Include top genres as tags
            'themes': themes,
            'summary': summary,
            'review': review,
            'body': '',
            'raw': {'goodreads_id': goodreads_id, 'synced_from': 'goodreads-rss', 'scraped': True},
        }
        book_id = upsert_book(conn, row)
        stats['new'] += 1

        # Update lookup tables
        existing_by_goodreads_id[goodreads_id] = book_id
        if isbn:
            existing_by_isbn[isbn] = book_id
        existing_by_title_author[key] = book_id

        # Create embedding
        try:
            text = f'{title} {author} {summary}'
            hash_val = text_hash(text, embedder.model)
            vectors = embedder.embed_batch([text])
            if vectors:
                upsert_embedding(
                    conn,
                    entity_type='book',
                    entity_id=book_id,
                    model=embedder.model,
                    text_hash_value=hash_val,
                    vector=vectors[0],
                )
                stats['embeddings_created'] += 1
        except Exception as e:
            print(f'Warning: embedding failed for {title}: {e}')
            stats['errors'] += 1

    print(json.dumps(stats, indent=2))


def cmd_sync_goodreads(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    stats = sync_goodreads_rss(conn, args.url or cfg['goodreads_rss_url'])
    print(json.dumps(stats, indent=2))


def cmd_preview_source(args):
    cfg = load_config(args.config)
    kind = args.kind or classify_source(args.url)
    name = args.name or source_name_from_url(args.url)
    source = {
        'name': name,
        'kind': kind,
        'url': args.url,
        'enabled': True,
        'weight': args.weight if args.weight is not None else default_weight(kind),
        'media_type': args.media_type,
    }
    result = discover_source_items(source, cfg)
    if args.format == 'json':
        payload = {
            'source': source,
            'skipped': result.skipped,
            'reason': result.reason,
            'content_hash': result.content_hash,
            'items_seen': len(result.items),
            'items': result.items[: args.limit],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    print(f"# Preview: {source['name']}")
    print(f"Kind: {source['kind']}")
    print(f"URL: {source['url']}")
    if result.skipped:
        print(f"Skipped: {result.reason}")
        print(f"Hash: {result.content_hash}")
        return
    print(f"Hash: {result.content_hash}")
    print(f"Items: {len(result.items)}")
    for idx, item in enumerate(result.items[: args.limit], 1):
        title = item.get('title') or '(untitled)'
        author = item.get('author') or 'Unknown'
        url = item.get('url') or ''
        print(f"{idx}. {title} — {author}")
        if url:
            print(f"   {url}")
        desc = (item.get('description') or '').strip()
        if desc:
            print(f"   {desc[:220]}")


def _scan_one_source(conn, source, cfg):
    prior = get_source(conn, source['name'])
    if prior and prior['last_hash']:
        source = {**source, 'last_hash': prior['last_hash']}
    result = discover_source_items(source, cfg)
    upsert_source(conn, {
        **source,
        'last_hash': result.content_hash,
        'last_seen_at': datetime.now(timezone.utc).isoformat(),
    })
    if result.skipped:
        return {'source': source['name'], 'skipped': True, 'inserted': 0, 'reason': result.reason}

    existing_keys = _candidate_match_keys(conn)
    for row in conn.execute('SELECT title, author, source FROM candidates').fetchall():
        existing_keys.update(candidate_match_keys(str(row['title']), str(row['author'] or ''), str(row['source'] or '')))

    inserted = 0
    skipped_dupe = 0
    for item in result.items:
        uid = item.get('source_uid') or item.get('guid') or item.get('url') or f"{source['name']}:{item.get('title', '')}"
        # Normalize Goodreads IDs even if a caller supplied an older indexed UID.
        if str(uid).startswith('goodreads:'):
            uid = str(uid).split(':', 2)[0] + ':' + str(uid).split(':', 2)[1]
        raw = item.get('raw') or {}
        if item.get('source_meta'):
            raw = {**raw, 'source_meta': item.get('source_meta')}
        if item.get('author'):
            raw.setdefault('author_provider', 'source')
        row = {
            'source': source['name'],
            'source_uid': uid,
            'title': item.get('title', '') or source['name'],
            'author': item.get('author', ''),
            'url': item.get('url') or '',
            'cover_url': item.get('cover_url', ''),
            'media_type': item.get('media_type') or source.get('media_type', 'audiobook'),
            'published_at': item.get('published_at'),
            'description': item.get('description') or '',
            'raw': raw,
            'status': 'new',
        }
        # Migrate the old ``goodreads:<id>:<page-index>`` key when a stable
        # Goodreads key is first seen again. This prevents a repaired row from
        # being inserted beside its historical blank-author duplicate.
        if str(uid).startswith('goodreads:'):
            stable_exists = conn.execute(
                'SELECT 1 FROM candidates WHERE source=? AND source_uid=?',
                (source['name'], uid),
            ).fetchone()
            if not stable_exists:
                legacy = conn.execute(
                    'SELECT source_uid FROM candidates WHERE source=? AND source_uid LIKE ? ORDER BY id LIMIT 1',
                    (source['name'], f'{uid}:%'),
                ).fetchone()
                if legacy:
                    conn.execute(
                        'UPDATE candidates SET source_uid=? WHERE source=? AND source_uid=?',
                        (uid, source['name'], legacy['source_uid']),
                    )
                    conn.commit()
        if candidate_is_banned(row, cfg.get('recommendation', {})):
            row['status'] = 'excluded'
            row['score'] = 0
            row['score_breakdown'] = {'score': 0, 'reasons': ['blocked format: graphic novel']}
            row['decided_at'] = datetime.now(timezone.utc).isoformat()
        keys = candidate_match_keys(row['title'], row['author'], row['source'])
        if keys & existing_keys:
            skipped_dupe += 1
            continue
        existing = conn.execute('SELECT title, author, url, cover_url, media_type, published_at, description, raw_json FROM candidates WHERE source=? AND source_uid=?', (source['name'], uid)).fetchone()
        fingerprint = (
            row['title'], row['author'], row['url'], row['cover_url'], row['media_type'], row['published_at'], row['description'],
            json.dumps(row['raw'], sort_keys=True, ensure_ascii=False),
        )
        if existing:
            existing_fp = (
                existing['title'], existing['author'], existing['url'], existing['cover_url'], existing['media_type'], existing['published_at'], existing['description'] or '', existing['raw_json'] or '{}',
            )
            if existing_fp == fingerprint:
                continue
        upsert_candidate(conn, row)
        existing_keys.update(keys)
        inserted += 1
    if source.get('origin') and source.get('kind') not in {'rss', 'atom', 'goodreads-rss'}:
        archive_sources_in_inbox(source['origin'], [source['url']])
    return {'source': source['name'], 'kind': source.get('kind'), 'inserted': inserted, 'items_seen': len(result.items), 'skipped_duplicates': skipped_dupe, 'hash': result.content_hash[:12]}


def _normalize_source_media_types(conn, sources):
    changed = 0
    for source in sources:
        media_type = source.get('media_type', 'audiobook')
        if not media_type:
            continue
        cur = conn.execute(
            'UPDATE candidates SET media_type=?, updated_at=CURRENT_TIMESTAMP WHERE source=? AND lower(coalesce(media_type, "")) <> lower(?)',
            (media_type, source['name'], media_type),
        )
        changed += cur.rowcount or 0
    if changed:
        conn.commit()
    return changed


def cmd_scan_sources(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    results = []
    sources = combined_sources(cfg)
    for source in sources:
        if args.source and source['name'] != args.source:
            continue
        results.append(_scan_one_source(conn, source, cfg))
    normalized = _normalize_source_media_types(conn, sources)
    print(json.dumps({'results': results, 'normalized_media_types': normalized}, indent=2))


def _book_rows(conn):
    rows = []
    for row in get_books(conn):
        tags = json.loads(row['tags_json']) if row['tags_json'] else []
        themes = json.loads(row['themes_json']) if row['themes_json'] else []
        rows.append({
            'id': row['id'],
            'title': row['title'],
            'author': row['author'],
            'rating': row['rating'],
            'goodreads_id': row['goodreads_id'],
            'storygraph_id': row['storygraph_id'],
            'isbn': row['isbn'],
            'tags': tags,
            'themes': themes,
            'summary': row['summary'] or '',
            'review': row['review'] or '',
            'body': row['body'] or '',
            'raw': json.loads(row['raw_json']) if row['raw_json'] else {},
        })
    return rows


def _sync_book_embeddings(conn, cfg):
    emb_cfg = cfg.get('embeddings', {})
    if not emb_cfg.get('enabled', True):
        return {'skipped': True, 'processed': 0, 'inserted': 0, 'updated': 0, 'failed': 0}
    model = emb_cfg.get('model', 'qwen3-embedding:4b')
    batch_size = int(emb_cfg.get('batch_size', 16))
    base_url = emb_cfg.get('base_url', 'http://ollama:11434')
    items = []
    for row in _book_rows(conn):
        text = book_text(row)
        items.append(EmbeddingItem('book', int(row['id']), text, embedding_text_hash(text, model)))
    stats = sync_embeddings(conn, entity_type='book', items=items, model=model, base_url=base_url, batch_size=batch_size)
    stats['skipped'] = len(items) - stats.get('inserted', 0) - stats.get('updated', 0) - stats.get('failed', 0)
    return stats


def _prepare_candidate_embeddings(conn, cfg, candidate_rows):
    emb_cfg = cfg.get('embeddings', {})
    if not emb_cfg.get('enabled', True):
        return {}
    model = emb_cfg.get('model', 'qwen3-embedding:4b')
    batch_size = int(emb_cfg.get('batch_size', 16))
    base_url = emb_cfg.get('base_url', 'http://ollama:11434')
    items = []
    candidate_ids = []
    for row in candidate_rows:
        candidate = _candidate_dict(row)
        text = candidate_text(candidate)
        items.append(EmbeddingItem('candidate', int(candidate['id']), text, embedding_text_hash(text, model)))
        candidate_ids.append(int(candidate['id']))
    sync_embeddings(conn, entity_type='candidate', items=items, model=model, base_url=base_url, batch_size=batch_size)
    if not candidate_ids:
        return {}
    emb_map = get_embeddings_map(conn, 'candidate', model, candidate_ids)
    out = {}
    for cid, emb in emb_map.items():
        out[cid] = blob_to_vector(emb['vector_blob'])
    return out


def _candidate_match_keys(conn) -> set[str]:
    keys = {
        normalize_candidate_match_key(str(row['title']), str(row['author'] or ''))
        for row in conn.execute('SELECT title, author FROM books').fetchall()
    }
    return keys


def _dedupe_candidates(conn) -> dict[str, int]:
    """Drop redundant candidates: already-read books and duplicate source rows.

    Keeps only the richest/highest-scored row for a candidate title+author. This
    is intentionally candidate-only; books use stricter dedupe to avoid merging
    distinct volumes.
    """
    rows = [dict(r) for r in conn.execute('SELECT * FROM candidates').fetchall()]
    book_keys = _candidate_match_keys(conn)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = normalize_candidate_match_key(row.get('title') or '', row.get('author') or '')
        groups.setdefault(key, []).append(row)

    def richness(row: dict) -> tuple:
        status_rank = {'approved': 5, 'imported': 5, 'new': 4, 'rejected': 2, 'excluded': 1}.get(row.get('status'), 0)
        score = row.get('score') if row.get('score') is not None else -1
        audiobook = 1 if str(row.get('media_type') or '').lower() == 'audiobook' else 0
        return (status_rank, float(score), audiobook, len(row.get('description') or ''), 1 if row.get('url') else 0, -int(row['id']))

    delete_ids: list[int] = []
    removed_library_matches = 0
    removed_duplicate_rows = 0
    for key, group in groups.items():
        if any(candidate_match_keys(r.get('title') or '', r.get('author') or '', r.get('source') or '') & book_keys for r in group):
            delete_ids.extend(int(r['id']) for r in group)
            removed_library_matches += len(group)
            continue
        if len(group) > 1:
            keep = max(group, key=richness)
            doomed = [int(r['id']) for r in group if int(r['id']) != int(keep['id'])]
            delete_ids.extend(doomed)
            removed_duplicate_rows += len(doomed)
    if delete_ids:
        conn.execute('DELETE FROM candidates WHERE id IN (%s)' % ','.join('?' for _ in delete_ids), delete_ids)
        conn.commit()
    return {'removed_library_matches': removed_library_matches, 'removed_duplicate_rows': removed_duplicate_rows, 'deleted': len(delete_ids)}


def _active_candidate_rows(conn):
    return list(conn.execute("SELECT * FROM candidates WHERE status='new' ORDER BY updated_at DESC"))


def _enrich_candidates(conn, candidates: list, cfg: dict[str, Any]) -> dict[str, int]:
    """Enrich candidates with missing metadata via Goodreads/OpenLibrary fallback.

    Enrichment is field-aware: a row with pages/genres but no author still
    needs repair. Only eligible rows count against the per-run rate limit.
    """
    enrich_cfg = cfg.get('enrichment', {})
    if not enrich_cfg.get('enabled', True):
        return {'skipped': True, 'enriched': 0, 'failed': 0}

    enriched = 0
    failed = 0
    delay = enrich_cfg.get('delay', 2.0)
    max_enrichments = int(enrich_cfg.get('max_per_run', 15))
    eligible: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for row in candidates:
        candidate = row if isinstance(row, dict) else _candidate_dict(row)
        raw = candidate.get('raw') or {}
        if not isinstance(raw, dict):
            raw = {}
        has_metadata = bool(raw.get('pages') or raw.get('genres') or raw.get('series'))
        missing_author = not str(candidate.get('author') or '').strip()
        if missing_author or not has_metadata:
            eligible.append((row, candidate, raw))
            if len(eligible) >= max_enrichments:
                break

    for row, candidate, raw in eligible:
        goodreads_id = raw.get('goodreads_id')
        if not goodreads_id:
            url_match = re.search(r'goodreads\.com/book/show/(\d+)', str(candidate.get('url') or ''), re.I)
            goodreads_id = url_match.group(1) if url_match else None
            if goodreads_id:
                raw['goodreads_id'] = goodreads_id
        isbn = raw.get('isbn') or raw.get('isbn13') or candidate.get('isbn13')

        try:
            meta = enrich_book_metadata(
                title=candidate.get('title', ''),
                author=candidate.get('author', ''),
                goodreads_id=goodreads_id,
                isbn=isbn,
                delay=delay,
            )
        except Exception:
            failed += 1
            continue

        if meta.get('provider') == 'none':
            failed += 1
            continue

        enriched_data: dict[str, Any] = {}
        if meta.get('pages'):
            enriched_data['pages'] = meta['pages']
        if meta.get('series'):
            enriched_data['series'] = meta['series']
        if meta.get('genres'):
            enriched_data['genres'] = meta['genres']
        if meta.get('isbn13'):
            enriched_data['isbn13'] = meta['isbn13']
        if meta.get('language'):
            enriched_data['language'] = meta['language']
        if meta.get('first_published'):
            enriched_data['first_published'] = meta['first_published']
        if meta.get('cover_url') and not candidate.get('cover_url'):
            enriched_data['cover_url'] = meta['cover_url']

        resolved_author = str(meta.get('author') or '').strip()
        if resolved_author and not candidate.get('author') and resolved_author.casefold() != str(candidate.get('title') or '').strip().casefold():
            enriched_data['resolved_author'] = resolved_author
            enriched_data['author_provider'] = meta.get('author_provider') or meta.get('provider')
        if meta.get('provider'):
            enriched_data['enrichment_provider'] = meta['provider']
        if meta.get('providers'):
            enriched_data['enrichment_providers'] = meta['providers']

        if enriched_data:
            raw.update(enriched_data)
            conn.execute(
                '''UPDATE candidates
                   SET raw_json=?, cover_url=coalesce(?, cover_url),
                       author=CASE WHEN trim(coalesce(?, '')) <> '' THEN ? ELSE author END,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE id=?''',
                (
                    json.dumps(raw, ensure_ascii=False),
                    meta.get('cover_url'),
                    resolved_author if not candidate.get('author') else '',
                    resolved_author if not candidate.get('author') else '',
                    candidate['id'],
                ),
            )
            enriched += 1

    if enriched:
        conn.commit()

    return {'enriched': enriched, 'failed': failed, 'total_checked': len(eligible)}


def _repair_historical_goodreads_candidates(
    conn,
    cfg: dict[str, Any],
    *,
    limit: int = 0,
    delay: float = 0.2,
    reopen: bool = True,
) -> dict[str, Any]:
    """Repair pre-fix Goodreads candidates without replaying source scans.

    Only non-terminal rows with blank authors and an identifiable Goodreads book
    ID are considered. Goodreads-by-ID is the authoritative repair source;
    OpenLibrary suggestions are not silently promoted when Goodreads cannot
    verify the author. Approval/rejection/Discord rows are never modified or
    deleted.
    """
    terminal_statuses = {'approved', 'rejected', 'imported', 'discord_pending'}
    rows = conn.execute(
        "SELECT * FROM candidates WHERE trim(COALESCE(author, '')) = '' ORDER BY id"
    ).fetchall()
    groups: dict[tuple[str, str], list[Any]] = {}
    skipped_no_id = 0
    for row in rows:
        if row['status'] in terminal_statuses:
            continue
        candidate = _candidate_dict(row)
        goodreads_id = _goodreads_id_for_candidate(candidate)
        if not goodreads_id:
            skipped_no_id += 1
            continue
        groups.setdefault((str(row['source']), goodreads_id), []).append(row)

    selected_groups = list(groups.items())
    if limit > 0:
        selected_groups = selected_groups[:limit]
    stats: dict[str, Any] = {
        'groups_considered': len(selected_groups),
        'rows_considered': sum(len(group) for _, group in selected_groups),
        'repaired_groups': 0,
        'repaired_rows': 0,
        'unresolved_groups': 0,
        'fallback_only_groups': 0,
        'errors': 0,
        'skipped_without_goodreads_id': skipped_no_id,
        'duplicate_rows_removed': 0,
        'protected_duplicate_groups': 0,
        'rescored': 0,
        'reopened': 0,
        'library_matches': 0,
        'below_threshold': 0,
    }
    repaired_groups: list[tuple[str, str]] = []
    repaired_ids: list[int] = []
    now = datetime.now(timezone.utc).isoformat()

    for (source, goodreads_id), group in selected_groups:
        representative = max(group, key=lambda row: (len(row['title'] or ''), -int(row['id'])))
        candidate = _candidate_dict(representative)
        try:
            metadata = enrich_book_metadata(
                title=candidate.get('title', ''),
                author='',
                goodreads_id=goodreads_id,
                delay=delay,
            )
        except Exception:
            stats['errors'] += 1
            continue

        resolved_author = str(metadata.get('author') or '').strip()
        author_provider = str(metadata.get('author_provider') or '').strip()
        if not resolved_author:
            stats['unresolved_groups'] += 1
            continue
        if author_provider != 'goodreads':
            # OpenLibrary is useful as an investigative fallback, but a title
            # match alone is not strong enough to rewrite historical rows.
            stats['fallback_only_groups'] += 1
            continue

        repaired_groups.append((source, goodreads_id))
        for row in group:
            raw = json.loads(row['raw_json']) if row['raw_json'] else {}
            if not isinstance(raw, dict):
                raw = {}
            repair_info = raw.get('historical_repair')
            if not isinstance(repair_info, dict):
                repair_info = {
                    'previous_status': row['status'],
                    'previous_score': row['score'],
                    'first_repaired_at': now,
                }
            raw['goodreads_id'] = goodreads_id
            raw['resolved_author'] = resolved_author
            raw['author_provider'] = 'goodreads'
            raw['enrichment_provider'] = metadata.get('provider') or 'goodreads'
            if metadata.get('providers'):
                raw['enrichment_providers'] = metadata['providers']
            for key in ('pages', 'series', 'genres', 'isbn13', 'language', 'format', 'first_published'):
                value = metadata.get(key)
                if value not in (None, '', []):
                    raw[key] = value
            repair_info.update({
                'repaired_at': now,
                'method': 'goodreads-book-id',
                'goodreads_id': goodreads_id,
                'author': resolved_author,
                'author_provider': 'goodreads',
            })
            raw['historical_repair'] = repair_info
            title = _strip_resolved_author_from_title(row['title'], resolved_author)
            cover_url = metadata.get('cover_url') or row['cover_url']
            conn.execute(
                """UPDATE candidates
                   SET title=?, author=?, raw_json=?, cover_url=?, updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status NOT IN ('approved','rejected','imported','discord_pending')""",
                (
                    title,
                    resolved_author,
                    json.dumps(raw, ensure_ascii=False),
                    cover_url,
                    row['id'],
                ),
            )
            repaired_ids.append(int(row['id']))
            stats['repaired_rows'] += 1
        stats['repaired_groups'] += 1
        if len(repaired_groups) % 25 == 0:
            conn.commit()
    conn.commit()

    # Collapse the old anchor/index duplicates for only the Goodreads IDs just
    # repaired. Do not run the broad candidate dedupe, which is allowed to
    # remove unrelated historical rows.
    for source, goodreads_id in repaired_groups:
        source_rows = conn.execute('SELECT * FROM candidates WHERE source=?', (source,)).fetchall()
        group = [
            row for row in source_rows
            if _goodreads_id_for_candidate(_candidate_dict(row)) == goodreads_id
        ]
        if len(group) <= 1:
            if group and group[0]['source_uid'] != f'goodreads:{goodreads_id}':
                conn.execute(
                    'UPDATE candidates SET source_uid=? WHERE id=?',
                    (f'goodreads:{goodreads_id}', group[0]['id']),
                )
            continue
        protected = False
        for row in group:
            if row['status'] in terminal_statuses:
                protected = True
                break
            if conn.execute('SELECT 1 FROM discord_messages WHERE candidate_id=? LIMIT 1', (row['id'],)).fetchone():
                protected = True
                break
            if conn.execute('SELECT 1 FROM feedback WHERE candidate_id=? LIMIT 1', (row['id'],)).fetchone():
                protected = True
                break
        if protected:
            stats['protected_duplicate_groups'] += 1
            continue
        keep = max(
            group,
            key=lambda row: (
                len(row['title'] or ''),
                len(row['description'] or ''),
                1 if str(row['url'] or '').startswith('http') else 0,
                -int(row['id']),
            ),
        )
        doomed = [row['id'] for row in group if row['id'] != keep['id']]
        if doomed:
            placeholders = ','.join('?' for _ in doomed)
            conn.execute(f'DELETE FROM candidates WHERE id IN ({placeholders})', doomed)
            stats['duplicate_rows_removed'] += len(doomed)
        conn.execute(
            'UPDATE candidates SET source_uid=? WHERE id=?',
            (f'goodreads:{goodreads_id}', keep['id']),
        )
    conn.commit()

    survivors = []
    for candidate_id in repaired_ids:
        row = conn.execute('SELECT * FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        if row and row['author'] and row['status'] not in terminal_statuses:
            survivors.append(row)
    if not survivors:
        record_event(conn, 'historical_candidate_repair', stats)
        return stats

    # Re-score repaired rows directly. They remain excluded unless they clear
    # the normal threshold, so a metadata repair cannot flood the next digest.
    profile = build_profile(_book_rows(conn), conn=conn, embedding_cfg=cfg.get('embeddings', {}))
    book_keys = _candidate_match_keys(conn)
    threshold = int(cfg.get('recommendation', {}).get('minimum_score', 69))
    source_weights = cfg.get('source_weights', {})
    for row in survivors:
        candidate = _candidate_dict(row)
        weight = source_weights.get(candidate['source'], source_weights.get('rss', 0.0))
        scored = score_candidate(
            candidate,
            profile,
            source_weight=weight,
            explain=True,
            recommendation_cfg=cfg.get('recommendation', {}),
        )
        raw = candidate.get('raw') or {}
        repair_info = raw.get('historical_repair') if isinstance(raw, dict) else {}
        if isinstance(repair_info, dict):
            repair_info['rescored_score'] = scored['score']
            repair_info['rescored_at'] = now
            raw['historical_repair'] = repair_info
        library_match = bool(candidate_match_keys(candidate.get('title', ''), candidate.get('author', ''), candidate.get('source', '')) & book_keys)
        conn.execute(
            'UPDATE candidates SET score=?, score_breakdown=?, raw_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?',
            (scored['score'], json.dumps(scored, ensure_ascii=False), json.dumps(raw, ensure_ascii=False), row['id']),
        )
        stats['rescored'] += 1
        if library_match:
            stats['library_matches'] += 1
        elif int(scored.get('score', 0)) >= threshold and not candidate_is_banned(candidate, cfg.get('recommendation', {})):
            if reopen and row['status'] == 'excluded':
                repair_info['reopened_at'] = now
                raw['historical_repair'] = repair_info
                conn.execute(
                    "UPDATE candidates SET status='new', decided_at=NULL, raw_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='excluded'",
                    (json.dumps(raw, ensure_ascii=False), row['id']),
                )
                stats['reopened'] += 1
        else:
            stats['below_threshold'] += 1
    conn.commit()
    record_event(conn, 'historical_candidate_repair', stats)
    return stats


def cmd_repair_candidates(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    stats = _repair_historical_goodreads_candidates(
        conn,
        cfg,
        limit=args.limit,
        delay=args.delay,
        reopen=not args.no_reopen,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _retire_low_scoring_candidates(conn, changed, cfg) -> dict[str, int]:
    rec_cfg = cfg.get('recommendation', {})
    threshold = int(rec_cfg.get('minimum_score', 82))
    updated = 0
    for candidate, scored in changed:
        if candidate.get('status') != 'new':
            continue
        if int(scored.get('score', 0)) < threshold:
            conn.execute(
                "UPDATE candidates SET status='excluded', decided_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='new'",
                (candidate['id'],),
            )
            updated += conn.total_changes  # approximate, reset not needed for final stats below
    conn.commit()
    actual = conn.execute("SELECT changes()").fetchone()[0] if changed else 0
    # sqlite changes() only returns last statement, so report counted candidates instead.
    return {'minimum_score': threshold, 'excluded': sum(1 for _, s in changed if int(s.get('score', 0)) < threshold)}


def _score_pending(conn, cfg, *, sync_books: bool = True, explain: bool = False, enrich: bool = False):
    if sync_books:
        _sync_book_embeddings(conn, cfg)
    _dedupe_candidates(conn)

    # Enrichment can change the title/author identity, so de-duplicate again
    # after repairing metadata before embeddings and scoring are prepared.
    if enrich:
        candidate_rows_for_enrich = _active_candidate_rows(conn)
        if candidate_rows_for_enrich:
            _enrich_candidates(conn, candidate_rows_for_enrich, cfg)
        _dedupe_candidates(conn)

    profile = build_profile(_book_rows(conn), conn=conn, embedding_cfg=cfg.get('embeddings', {}))
    source_weights = cfg.get('source_weights', {})
    changed = []
    candidate_rows = _active_candidate_rows(conn)
    candidate_embeddings = _prepare_candidate_embeddings(conn, cfg, candidate_rows)
    for row in candidate_rows:
        candidate = _candidate_dict(row)
        if candidate['id'] in candidate_embeddings:
            candidate['embedding'] = candidate_embeddings[candidate['id']]
        weight = source_weights.get(candidate['source'], source_weights.get('rss', 0.0))
        scored = score_candidate(candidate, profile, source_weight=weight, explain=explain, recommendation_cfg=cfg.get('recommendation', {}))
        conn.execute('UPDATE candidates SET score=?, score_breakdown=?, updated_at=CURRENT_TIMESTAMP WHERE id=?', (scored['score'], json.dumps(scored, ensure_ascii=False), candidate['id']))
        changed.append((candidate, scored))
    conn.commit()
    return profile, changed


def cmd_score(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    profile, changed = _score_pending(conn, cfg, explain=True)
    top = sorted(changed, key=lambda x: x[1]['score'], reverse=True)[: args.limit]
    out = []
    for candidate, scored in top:
        out.append({
            'id': candidate['id'], 'title': candidate['title'], 'author': candidate['author'],
            'score': scored['score'], 'reasons': scored['reasons'], 'similar_books': scored['similar_books'],
            'semantic_books': scored.get('semantic_books', []),
        })
    print(json.dumps({'count': len(changed), 'top': out}, indent=2, ensure_ascii=False))


def cmd_embed(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    book_stats = _sync_book_embeddings(conn, cfg)
    candidate_rows = _active_candidate_rows(conn)
    candidate_stats = {'skipped': True, 'processed': 0, 'inserted': 0, 'updated': 0, 'failed': 0}
    if candidate_rows:
        _prepare_candidate_embeddings(conn, cfg, candidate_rows)
        candidate_stats = {'processed': len(candidate_rows)}
    print(json.dumps({'books': book_stats, 'candidates': candidate_stats}, indent=2, ensure_ascii=False))


def cmd_digest(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    # Weekly digest: enrich missing metadata (Goodreads/OpenLibrary fallback),
    # use cached book embeddings/profile inputs. Nightly handles
    # new candidate embeddings and low-score retirement.
    profile, changed = _score_pending(conn, cfg, sync_books=False, explain=True, enrich=True)
    # Filter out candidates whose title+author match an existing book.
    book_keys = _candidate_match_keys(conn)
    changed = [(c, s) for c, s in changed if normalize_candidate_match_key(c.get('title', ''), c.get('author', '')) not in book_keys]
    rows = sorted(changed, key=lambda x: x[1]['score'], reverse=True)[: args.limit]

    if args.format == 'json':
        print(json.dumps([{
            'id': c['id'], 'title': c['title'], 'author': c['author'], 'score': s['score'], 'reasons': s['reasons'], 'status': c['status'],
            'semantic_books': s.get('semantic_books', []),
        } for c, s in rows], indent=2, ensure_ascii=False))
        return
    print('# Weekly Book Recommendations')
    print()
    if not rows:
        print('WARNING: 0 candidates are eligible after metadata and library-quality filters.')
        print()
    for c, s in rows:
        print(f"## {c['title']} — {c.get('author') or 'Unknown'}")
        print(f"Score: {s['score']}/100")
        print(f"Candidate ID: {c['id']}")
        if c.get('url'):
            print(f"URL: {c['url']}")
        # Show enrichment source if available.
        raw = c.get('raw') or {}
        if isinstance(raw, dict) and raw.get('enrichment_provider'):
            print(f"Enriched via: {raw['enrichment_provider']}")
        if s['reasons']:
            print('Why:')
            for reason in s['reasons']:
                print(f'- {reason}')
        if s['similar_books']:
            print('Closest matches:')
            for b in s['similar_books'][:3]:
                print(f"- {b['title']} — {b.get('author') or 'Unknown'} ({b['rating']} stars, {b['similarity']})")
        if s.get('semantic_books'):
            print('Semantic matches:')
            for b in s['semantic_books'][:3]:
                print(f"- {b['title']} — {b.get('author') or 'Unknown'} ({b['rating']} stars, {b['similarity']})")
        print('')


def cmd_approve(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    row = conn.execute('SELECT * FROM candidates WHERE id = ?', (args.candidate_id,)).fetchone()
    if not row:
        raise SystemExit(f'Candidate {args.candidate_id} not found')
    client = LibrarrClient(cfg['librarr']['url'], resolve_librarr_api_key(cfg))
    result = client.add_to_wishlist(row['title'], row['author'] or '', cfg['librarr'].get('wishlist_media_type', 'audiobook'))
    librarr_id = None
    if isinstance(result, dict):
        librarr_id = str(result.get('id') or result.get('wishlist_id') or '') or None
    set_candidate_status(conn, args.candidate_id, 'approved', librarr_id=librarr_id, note='approved via CLI')
    record_feedback(conn, args.candidate_id, 'approve', args.note or '', 'cli')
    print(json.dumps({'candidate_id': args.candidate_id, 'librarr': result}, indent=2, ensure_ascii=False))


def cmd_reject(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    set_candidate_status(conn, args.candidate_id, 'rejected', note=args.note or 'rejected via CLI')
    record_feedback(conn, args.candidate_id, 'reject', args.note or '', 'cli')
    print(json.dumps({'candidate_id': args.candidate_id, 'status': 'rejected'}, indent=2))


def cmd_nightly(args):
    cfg = load_config(args.config)
    conn = connect(cfg['db_path'])
    init_db(conn)
    obs = import_obsidian(conn, cfg['vault_path'])
    dedupe = dedupe_books_by_title_author(conn)
    goodreads = sync_goodreads_rss(conn, args.goodreads_url or cfg['goodreads_rss_url'])
    scan = []
    for source in combined_sources(cfg):
        scan.append(_scan_one_source(conn, source, cfg))
    candidate_dedupe = _dedupe_candidates(conn)
    # Nightly should never resync all book embeddings, but it should repair
    # missing candidate authors/metadata before embedding and scoring.
    profile, changed = _score_pending(conn, cfg, sync_books=not args.no_sync_embeddings, explain=False, enrich=True)
    retired = _retire_low_scoring_candidates(conn, changed, cfg)
    # Filter out candidates whose title+author match an existing book.
    book_keys = _candidate_match_keys(conn)
    changed = [(c, s) for c, s in changed if normalize_candidate_match_key(c.get('title', ''), c.get('author', '')) not in book_keys]
    top = sorted(changed, key=lambda x: x[1]['score'], reverse=True)[: cfg.get('recommendation', {}).get('top_n', 10)]
    summary = {
        'obsidian': obs,
        'dedupe': dedupe,
        'goodreads': goodreads,
        'sources': scan,
        'candidate_dedupe': candidate_dedupe,
        'retired_low_score': retired,
        'scored': len(changed),
        'top': [{'id': c['id'], 'title': c['title'], 'author': c['author'], 'score': s['score']} for c, s in top],
    }
    if any((obs.get('imported', 0), goodreads.get('inserted', 0), goodreads.get('updated', 0), sum(s.get('inserted', 0) for s in scan), len(changed), dedupe.get('deleted_rows', 0))):
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(prog='books-recommender')
    p.add_argument('--config', default=None)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('init')
    s.add_argument('--import-obsidian', action='store_true')
    s.set_defaults(func=cmd_init)

    s = sub.add_parser('sync-goodreads')
    s.add_argument('--url', default=None)
    s.set_defaults(func=cmd_sync_goodreads)

    s = sub.add_parser('sync-goodreads-to-obsidian')
    s.add_argument('--goodreads-url', default=None)
    s.set_defaults(func=cmd_sync_goodreads_to_obsidian)

    s = sub.add_parser('preview-source')
    s.add_argument('url')
    s.add_argument('--kind', default=None)
    s.add_argument('--name', default=None)
    s.add_argument('--media-type', default='audiobook')
    s.add_argument('--weight', type=float, default=None)
    s.add_argument('--limit', type=int, default=10)
    s.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    s.set_defaults(func=cmd_preview_source)

    s = sub.add_parser('scan-sources')
    s.add_argument('--source', default=None)
    s.set_defaults(func=cmd_scan_sources)

    s = sub.add_parser('score')
    s.add_argument('--limit', type=int, default=20)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser('embed')
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser('digest')
    s.add_argument('--limit', type=int, default=10)
    s.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    s.set_defaults(func=cmd_digest)

    s = sub.add_parser('repair-candidates', help='Repair historical blank-author Goodreads candidates')
    s.add_argument('--limit', type=int, default=0, help='Maximum Goodreads IDs to repair; 0 means all')
    s.add_argument('--delay', type=float, default=0.2, help='Delay between metadata requests in seconds')
    s.add_argument('--no-reopen', action='store_true', help='Repair and rescore, but leave excluded rows excluded')
    s.set_defaults(func=cmd_repair_candidates)

    s = sub.add_parser('discord-post')
    s.add_argument('--limit', type=int, default=10)
    s.add_argument('--channel-id', default=None)
    s.set_defaults(func=cmd_discord_post)

    s = sub.add_parser('discord-sync-reactions')
    s.add_argument('--channel-id', default=None)
    s.set_defaults(func=cmd_discord_sync_reactions)

    s = sub.add_parser('approve')
    s.add_argument('candidate_id', type=int)
    s.add_argument('--note', default='')
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser('reject')
    s.add_argument('candidate_id', type=int)
    s.add_argument('--note', default='')
    s.set_defaults(func=cmd_reject)

    s = sub.add_parser('nightly')
    s.add_argument('--goodreads-url', default=None)
    s.add_argument('--no-sync-embeddings', action='store_true', help='Skip embedding sync (faster, use for daily cron)')
    s.set_defaults(func=cmd_nightly)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    main()
